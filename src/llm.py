"""Talking to the language model (used by B-06 now, and later by B-08 and B-09).

How it works:
- The provider is OpenRouter. It uses the same "OpenAI-style" API as many providers, so we
  talk to it with the standard openai library, pointed at OpenRouter's address.
  The model is chosen in .env (LLM_MODEL); names ending in ":free" cost nothing.
- temperature is 0, so the same ticket gets the same answer as far as the model allows.
- If you choose a "reasoning" model (one that thinks before answering), set
  LLM_REASONING_EFFORT=low: we then ask it to think briefly, keep the thinking out of the
  reply, and allow extra room (LLM_REASONING_BUDGET) for it. Leave it empty otherwise.
- Every answer is saved in a small database file (storage/llm_cache.sqlite). If we ask the
  exact same question again, we read the saved answer and make no call. This saves the
  free allowance and makes runs repeatable.
- If the provider says "too many requests", or times out, we wait and try again a few times.
  If it still fails, we raise LLMError. The caller decides what to do (usually: send the
  ticket to a person). One failing call never crashes a whole run.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from src import config

PROVIDER = "openrouter"
BASE_URL = "https://openrouter.ai/api/v1"


class LLMError(Exception):
    """The model could not give a usable answer (no key, provider down, or bad output)."""


# ------------------------------------------------------------------ prompts


def load_prompt(relative_path: str) -> tuple[str, str]:
    """Read a prompt file from prompts/ and return (system text, user template).
    The file has a line 'SYSTEM' and a line 'USER'; the text under each belongs to it."""
    text = (config.ROOT / "prompts" / relative_path).read_text(encoding="utf-8")
    match = re.match(r"\s*SYSTEM\s*\n(.*)\nUSER\s*\n(.*)\Z", text, flags=re.S)
    if not match:
        raise ValueError(f"{relative_path}: expected a SYSTEM part and a USER part")
    return match.group(1).strip(), match.group(2).strip()


def fill(template: str, values: dict[str, str]) -> str:
    """Put values into {placeholders}. We do not use str.format, because the prompts
    also contain JSON examples with { and } that must stay as they are."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def parse_json(text: str) -> dict:
    """Read the first JSON object in the model's reply. Models sometimes add words or
    ```json fences around it; we ignore those. Raises LLMError if there is no valid object."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        raise LLMError("reply contained no JSON object")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError as exc:
                raise LLMError(f"reply JSON could not be read: {exc}") from exc
    raise LLMError("reply JSON was not complete")


# ------------------------------------------------------------------ cache


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, reply TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT reply FROM replies WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, reply: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, reply))
            self._db.commit()


# ------------------------------------------------------------------ client


class LLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 cache_path: str | None = None, timeout_s: float = 30.0, max_attempts: int | None = None,
                 min_interval_s: float | None = None):
        self.provider = PROVIDER
        self.model = model or config.LLM_MODEL
        self._api_key = api_key if api_key is not None else config.OPENROUTER_API_KEY
        self._base_url = BASE_URL
        self._timeout = timeout_s
        self._max_attempts = max_attempts or config.LLM_MAX_ATTEMPTS
        # Wait at least this long between real calls, to stay inside the free tier's limits.
        self._min_interval = config.LLM_MIN_INTERVAL_S if min_interval_s is None else min_interval_s
        self._last_call = 0.0
        self._cache = _Cache(Path(cache_path or config.LLM_CACHE_PATH))
        self._client = None
        self.calls_made = 0       # real calls to the provider (for the metrics report)
        self.cache_hits = 0

    def _openai(self):
        if config.LLM_DISABLED:
            raise LLMError("language model calls are switched off (LLM_DISABLED)")
        if not self._api_key or self._api_key == "your_key_here":
            raise LLMError("no OpenRouter API key set; add OPENROUTER_API_KEY to .env")
        if self._client is None:
            from openai import OpenAI
            # max_retries=0: we do our own waiting and retrying below, so we can count it.
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key,
                                  timeout=self._timeout, max_retries=0)
        return self._client

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        """Ask the model one question and return its text reply (from the cache if seen before)."""
        key = hashlib.sha256(json.dumps([self.provider, self.model, system, user, max_tokens,
                                         config.LLM_REASONING_EFFORT]).encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        import openai
        from tenacity import (RetryError, retry, retry_if_exception_type, stop_after_attempt,
                              wait_exponential)

        retryable = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                     openai.InternalServerError)

        @retry(retry=retry_if_exception_type(retryable), stop=stop_after_attempt(self._max_attempts),
               wait=wait_exponential(multiplier=2, min=2, max=30), reraise=False)
        def _call() -> str:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            extra = {}
            limit = max_tokens
            if config.LLM_REASONING_EFFORT:      # reasoning models only (e.g. gpt-oss)
                extra = {"extra_body": {"reasoning": {"effort": config.LLM_REASONING_EFFORT, "exclude": True}}}
                limit = max_tokens + config.LLM_REASONING_BUDGET
            response = self._openai().chat.completions.create(
                model=self.model, temperature=0, max_tokens=limit,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **extra)
            self.calls_made += 1
            return response.choices[0].message.content or ""

        try:
            reply = _call()
        except RetryError as exc:
            raise LLMError(f"provider kept failing: {exc.last_attempt.exception()}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"provider error: {type(exc).__name__}: {exc}") from exc
        if not reply.strip():                     # e.g. all the room was used for thinking
            raise LLMError("the model returned an empty reply")
        self._cache.put(key, reply)
        return reply

    def complete_json(self, system: str, user: str, max_tokens: int = 400) -> dict:
        return parse_json(self.complete(system, user, max_tokens))


_default: LLMClient | None = None


def get_client() -> LLMClient:
    """One shared client for the whole run."""
    global _default
    if _default is None:
        _default = LLMClient()
    return _default
