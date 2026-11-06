"""The hand-off package for a person (FR-08, backlog item B-16).

Every ticket that goes to a person (action "escalate" or "block") gets a hand-off note, so the
engineer does not have to re-read the thread or ask the customer again. It always has:

  summary                       what the customer needs and why it matters (60 words or fewer)
  intent, urgency, confidence   what the system decided, and how sure it was
  already_provided_by_customer  facts and attempts the customer has already given
  relevant_articles             help articles search found (only ids search really returned)
  system_uncertain_about        what the system could not decide
  escalation_reason             why it came to a person, in plain words
  suggested_first_question      a question the customer has not answered yet, or "none needed"
  draft                         the drafted answer, if one was written
  passages                      the passages search found (id and title)

The language model writes the summary, the already-provided list, the uncertainty and the first
question (prompt PR-04). Everything else is filled by code from what the system actually did, so
the model cannot change the intent, the reason, or which articles were found. If the model is
unavailable, a plain fallback note is built from the ticket text instead (B-17), so a hand-off is
never missing. escalate() never raises.
"""
from __future__ import annotations

from typing import Any

from src.llm import LLMClient, LLMError, get_client, render

PROMPT_FILE = "build/PR-04_escalation_summary_v1.0.md"
PROMPT_VERSION = "PR-04 v1.0"
FIELDS = ("summary", "intent", "urgency", "confidence", "already_provided_by_customer", "relevant_articles",
          "system_uncertain_about", "escalation_reason", "suggested_first_question", "draft", "passages")


def _first_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + ("…" if len(words) > n else "")


def _uncertainty(label: Any, reason: str) -> str:
    if label is None:
        return "Everything: the ticket could not be processed."
    if getattr(label, "method", "") == "fallback":
        return "What the ticket is about: it could not be classified."
    alts = [a.get("intent") for a in (getattr(label, "alternatives", None) or []) if isinstance(a, dict)]
    alt = f" (other possibilities: {', '.join(a.replace('_', ' ') for a in alts if a)})" if alts else ""
    return (f"Whether it is really {label.intent.replace('_', ' ')}: {label.confidence:.0%} sure{alt}.")


def escalate(ticket: Any, label: Any, passages: list[Any], draft: dict[str, Any] | None, reason: str,
             client: LLMClient | None = None, use_model: bool = True) -> dict[str, Any]:
    """Build the hand-off note. use_model=False (or a model failure) gives the plain fallback note."""
    found = [p.passage_id for p in passages]
    note: dict[str, Any] = {
        "summary": "",
        "intent": getattr(label, "intent", None) or "not classified",
        "urgency": getattr(label, "urgency", None) or "not stated",
        "confidence": getattr(label, "confidence", None),
        "already_provided_by_customer": [],
        "relevant_articles": sorted({pid.split("#")[0] for pid in found}),
        "system_uncertain_about": _uncertainty(label, reason),
        "escalation_reason": reason,
        "suggested_first_question": "none needed",
        "draft": (draft or {}).get("answer") or None,
        "passages": [{"id": p.passage_id, "title": p.title} for p in passages],
        "written_by": "code (fallback)",
        "prompt_version": None,
        "note": "",
    }
    text = " ".join(f"{getattr(ticket, 'subject', '')} {getattr(ticket, 'body', '')}".split())
    note["summary"] = (f"Customer wrote: {_first_words(text, 50)}" if text else "The ticket has no message text.")
    if not use_model or not text:
        return note

    findings = {k: note[k] for k in ("intent", "urgency", "confidence", "escalation_reason")}
    findings["passages"] = note["passages"]
    findings["draft"] = note["draft"]
    try:
        import json
        system, user = render(PROMPT_FILE, {"channel": getattr(ticket, "channel", ""), "subject": ticket.subject,
                                            "body": ticket.body, "findings_json": json.dumps(findings, indent=1)})
        reply = (client or get_client()).complete_json(system, user, max_tokens=500)
    except LLMError as exc:
        note["note"] = f"Summary written by code: the language model was unavailable ({exc})."
        return note
    except Exception as exc:          # last line of defence
        note["note"] = f"Summary written by code: {type(exc).__name__}: {exc}"
        return note

    summary = str(reply.get("summary") or "").strip()
    if summary:
        note["summary"] = _first_words(summary, 60)
    provided = reply.get("already_provided_by_customer")
    if isinstance(provided, list):
        note["already_provided_by_customer"] = [str(x) for x in provided if str(x).strip()][:10]
    uncertain = str(reply.get("system_uncertain_about") or "").strip()
    if uncertain:
        note["system_uncertain_about"] = uncertain
    question = str(reply.get("suggested_first_question") or "").strip()
    if question:
        note["suggested_first_question"] = question
    # Articles: keep only ids search actually found (T-08b); never add new ones from the model.
    listed = reply.get("relevant_articles")
    if isinstance(listed, list):
        allowed = {pid.split("#")[0] for pid in found} | set(found)
        kept = [str(a).split("#")[0] for a in listed if str(a) in allowed or str(a).split("#")[0] in allowed]
        dropped = [str(a) for a in listed if str(a).split("#")[0] not in allowed]
        if kept:
            note["relevant_articles"] = sorted(set(kept))
        if dropped:
            note["note"] = f"Removed article ids search did not find: {', '.join(dropped)}."
    note["written_by"] = "PR-04"
    note["prompt_version"] = PROMPT_VERSION
    return note


def is_complete(note: dict[str, Any] | None) -> bool:
    """T-08a: all the parts a person needs are there."""
    if not note:
        return False
    return all(k in note for k in FIELDS) and bool(note["summary"]) and bool(note["escalation_reason"]) \
        and note["intent"] is not None and note["urgency"] is not None \
        and isinstance(note["relevant_articles"], list) and bool(note["system_uncertain_about"])
