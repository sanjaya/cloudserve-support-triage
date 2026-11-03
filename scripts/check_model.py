"""Check the language model (OpenRouter): one real call must return text.

    python scripts/check_model.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm import LLMClient, LLMError  # noqa: E402

# A fresh, empty cache each time, so this always makes a real call.
client = LLMClient(cache_path=str(Path(tempfile.mkdtemp()) / "cache.sqlite"))
try:
    reply = client.complete("You are a test.", "Reply with the single word ready.", max_tokens=5)
except LLMError as exc:
    sys.exit(f"Not working: {exc}")
print(f"{client.provider} / {client.model}: {reply!r}")
