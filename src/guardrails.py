"""Safety checks on every answer before it is sent (FR-07, backlog item B-09).

Only answers that passed the draft step's own checks (src/generate.py) come here. Every check
runs on every answer, and every result is recorded, passed or not (test T-07b). If ANY check
says "block", or a check could not run, the answer is not sent: the ticket goes to a person
with the draft attached. A wrongly blocked answer costs a person a few minutes; a wrongly sent
one reaches a customer.

The checks, cheap code checks first:

1. never_automate  The topic is on the never-automate list (FR-05). decide already checks
                   this; it is checked again here, just before release, as the spec asks.
2. private_data    The answer contains an email address, phone number, customer id or key that
                   is not in the help articles, or the name of another customer.
3. commitment      The answer promises a refund or credit, says the problem has been fixed on
                   CloudServe's side, or gives a date for a fix (code patterns).
4. injection       The ticket tries to give the system instructions. A code pattern catches the
                   obvious wording; prompt PR-06 (language model) catches the rest.
5. grounding       Every claim is supported by the passage it cites, and there is no forbidden
                   commitment: prompt PR-05 (language model).

Each check gives "pass", "block", or "error" (could not run). check_answer() never raises.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src import config
from src.classify import Classification
from src.generate import Draft, format_passages
from src.ingest import Ticket
from src.llm import LLMClient, LLMError, get_client, render
from src.retrieve import Passage

# Versions of the grounding prompt; which one is used is set by GROUNDING_PROMPT_VERSION (.env).
# scripts/evaluate_grounding.py compares them on development answers.
GROUNDING_PROMPTS = {"1.0": "build/PR-05_grounding_check_v1.0.md",
                     "1.1": "build/PR-05_grounding_check_v1.1.md"}
INJECTION_PROMPT = "build/PR-06_instruction_integrity_v1.0.md"
CHECKS = ("never_automate", "private_data", "commitment", "injection", "grounding")
INJECTION_MIN_CONFIDENCE = 0.5

# ---------------------------------------------------------------- patterns

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d ()-]{8,}\d(?!\w)(?!\.\d)")
_CUSTOMER_ID = re.compile(r"\bCUST-\d+\b", re.I)
_KEY = re.compile(r"\b(?:sk|pk|rk|key|api|token|secret)[-_][A-Za-z0-9_-]{12,}\b|\b[A-Za-z0-9]{32,}\b", re.I)

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
_DAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight)"
_COMMITMENTS = [
    ("refund or credit promised", re.compile(
        r"\b(?:refund|credit|reimburse\w*|compensat\w*)\b[^.\n]{0,60}\b(?:issued|processed|applied|approved|"
        r"granted|on its way|will (?:be|receive|see|get))\b|\bwe(?:'ll| will| have| are going to)\b[^.\n]{0,40}"
        r"\b(?:refund|credit|reimburse|compensate)\b", re.I)),
    ("fixed on our side", re.compile(
        r"\b(?:we|cloudserve|our (?:team|engineers))\b[^.\n]{0,40}\b(?:have|has)\b[^.\n]{0,20}"
        r"\b(?:fixed|resolved|deployed a fix|corrected)\b|\b(?:issue|problem|bug|incident)\b[^.\n]{0,30}"
        r"\bhas been (?:fixed|resolved)\b", re.I)),
    ("date for a fix", re.compile(
        rf"\b(?:fix|fixed|resolved|resolution|patch|release)\b[^.\n]{{0,50}}\b(?:by|on|before|within|until)\s+"
        rf"(?:{_DAY}|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}|{_MONTH}\s+\d{{1,2}}|the end of|next week|"
        rf"\d+\s+(?:hours?|days?|weeks?))\b", re.I)),
]
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget)\b[^.\n]{0,30}\b(?:previous|prior|above|your|all|these)\b[^.\n]{0,20}"
    r"\b(?:instructions?|rules|prompt)\b|\b(?:system prompt|developer message|reveal your (?:prompt|instructions)|"
    r"you are now|act as (?:an?|the) )", re.I)


# ---------------------------------------------------------------- customer names


_known_names: set[str] = set()
_loaded_files = False


def remember_customers(raw_tickets: list[Any]) -> None:
    """Add the customer names in a ticket file, so the private-data check can spot another
    customer's name in an answer. The harness calls this for every file it runs."""
    for r in raw_tickets:
        if isinstance(r, dict) and isinstance(r.get("customer_name"), str) and len(r["customer_name"].split()) >= 2:
            _known_names.add(r["customer_name"].strip())


def known_customer_names() -> set[str]:
    global _loaded_files
    if not _loaded_files:
        _loaded_files = True
        for name in ("validation_tickets.json", "development_tickets.json"):
            path = config.DATA_DIR / name
            if path.exists():
                try:
                    remember_customers(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    pass
    return _known_names


# ---------------------------------------------------------------- results


@dataclass
class GuardrailReport:
    results: dict[str, str] = field(default_factory=dict)    # check -> pass / block / error
    notes: dict[str, str] = field(default_factory=dict)      # check -> what was found
    passed: bool = False
    reason: str = ""
    prompt_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- the code checks


def check_private_data(answer: str, ticket: Ticket, passages: list[Passage]) -> tuple[str, str]:
    allowed = "\n".join(p.text for p in passages).lower()
    found = []
    for kind, pattern in [("email address", _EMAIL), ("phone number", _PHONE),
                          ("customer id", _CUSTOMER_ID), ("key or token", _KEY)]:
        for m in pattern.findall(answer):
            if kind == "phone number" and sum(ch.isdigit() for ch in m) < 9:
                continue
            if kind == "key or token" and not any(ch.isdigit() for ch in m):   # e.g. "token-based-login"
                continue
            if m.lower() not in allowed:
                found.append(f"{kind} '{m}'")
    own = str(ticket.original.get("customer_name") or "").strip().lower()
    text = answer.lower()
    found += [f"customer name '{n}'" for n in sorted(known_customer_names())
              if n.lower() != own and re.search(rf"\b{re.escape(n.lower())}\b", text)]
    return ("block", "; ".join(found)) if found else ("pass", "")


def check_commitment(answer: str, passages: list[Passage]) -> tuple[str, str]:
    """Wording that is copied word for word from a help article is policy, not a promise."""
    allowed = " ".join(" ".join(p.text.split()) for p in passages).lower()
    found = [f"{what}: '{m.group(0)}'" for what, pattern in _COMMITMENTS for m in pattern.finditer(answer)
             if " ".join(m.group(0).split()).lower() not in allowed]
    return ("block", "; ".join(found)) if found else ("pass", "")


# ---------------------------------------------------------------- the model checks


def check_injection(ticket: Ticket, client: LLMClient) -> tuple[str, str]:
    m = _INJECTION.search(ticket.text)
    if m:
        return "block", f"instruction-like text in the ticket: '{m.group(0)}'"
    system, user = render(INJECTION_PROMPT, {"subject": ticket.subject, "body": ticket.body})
    reply = client.complete_json(system, user, max_tokens=150)
    try:
        confidence = float(reply.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 1.0
    if reply.get("injection_suspected") is True and confidence >= INJECTION_MIN_CONFIDENCE:
        return "block", f"PR-06: {reply.get('evidence') or 'instructions to the system suspected'}"
    if not isinstance(reply.get("injection_suspected"), bool):
        raise LLMError("PR-06 reply had no injection_suspected value")
    return "pass", ""


def check_grounding(answer: str, passages: list[Passage], client: LLMClient,
                    version: str | None = None) -> tuple[str, str]:
    version = version or config.GROUNDING_PROMPT_VERSION
    if version not in GROUNDING_PROMPTS:
        raise LLMError(f"unknown GROUNDING_PROMPT_VERSION {version!r}; use one of {list(GROUNDING_PROMPTS)}")
    system, user = render(GROUNDING_PROMPTS[version], {"draft_answer": answer, "passages": format_passages(passages)})
    reply = client.complete_json(system, user, max_tokens=1200)
    # v1.0 lists every claim with a verdict; v1.1 lists only the unsupported ones.
    claims = reply.get("claims") if isinstance(reply.get("claims"), list) else []
    unsupported = [c for c in claims if isinstance(c, dict) and c.get("verdict") == "unsupported"]
    if isinstance(reply.get("unsupported"), list):
        unsupported += [c if isinstance(c, dict) else {"sentence": str(c)} for c in reply["unsupported"]]
    try:
        count = int(reply.get("unsupported_count", len(unsupported)))
    except (TypeError, ValueError):
        count = len(unsupported)
    forbidden = reply.get("forbidden_commitment") is True
    decision = reply.get("decision")
    if decision not in ("pass", "block"):
        raise LLMError("PR-05 reply had no pass/block decision")
    if decision == "block" or count > 0 or unsupported or forbidden:
        parts = []
        if unsupported or count:
            first = (unsupported[0].get("sentence") or "")[:120] if unsupported else ""
            parts.append(f"{max(count, len(unsupported))} unsupported claim(s)" + (f", e.g. '{first}'" if first else ""))
        if forbidden:
            parts.append("forbidden commitment")
        return "block", "PR-05: " + ("; ".join(parts) or "blocked")
    return "pass", ""


# ---------------------------------------------------------------- all checks


_LABELS = {"never_automate": "never-automate topic", "private_data": "private data",
           "commitment": "promise", "injection": "instructions in the ticket", "grounding": "unsupported claim"}


def check_answer(ticket: Ticket, label: Classification, draft: Draft, passages: list[Passage],
                 client: LLMClient | None = None, grounding_version: str | None = None) -> GuardrailReport:
    """Run every check on the answer and decide whether it may be sent. Never raises."""
    client = client or get_client()
    grounding_version = grounding_version or config.GROUNDING_PROMPT_VERSION
    report = GuardrailReport(prompt_versions={"grounding": f"PR-05 v{grounding_version}", "injection": "PR-06 v1.0"})
    answer = draft.answer

    def run(name: str, fn) -> None:
        try:
            outcome, note = fn()
        except Exception as exc:               # fail closed: a check that cannot run blocks
            outcome, note = "error", f"{type(exc).__name__}: {exc}"
        report.results[name] = outcome
        if note:
            report.notes[name] = note

    run("never_automate", lambda: ("block", f"intent {label.intent}") if label.intent in config.BLOCKED_INTENTS
        else ("pass", ""))
    run("private_data", lambda: check_private_data(answer, ticket, passages))
    run("commitment", lambda: check_commitment(answer, passages))
    run("injection", lambda: check_injection(ticket, client))
    run("grounding", lambda: check_grounding(answer, passages, client, grounding_version))

    failed = [n for n, r in report.results.items() if r != "pass"]
    report.passed = not failed
    if report.passed:
        report.reason = "All safety checks passed."
    else:
        what = "; ".join(f"{_LABELS[n]}{' (check could not run)' if report.results[n] == 'error' else ''}"
                         f"{': ' + report.notes[n] if report.notes.get(n) else ''}" for n in failed)
        report.reason = (f"The answer was stopped by the safety checks ({what}), so a person will handle "
                         "it, with the draft attached.")
    return report
