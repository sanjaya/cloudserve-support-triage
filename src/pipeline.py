"""The steps one ticket goes through.

Built so far: classifying the ticket (B-06) and searching the help articles (B-04).
Still to come: decide (B-07), write the answer (B-08) and check it (B-09).
Until then every ticket is sent to a person, with a reason, its labels and the passages found.
The `Result` shape only gains optional fields, so the harness and the metrics keep working
as each step is added.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.classify import Classifier
from src.ingest import Ticket
from src.retrieve import Retriever

ACTIONS = ("auto_respond", "escalate", "block")

_retriever: Retriever | None = None
_classifier: Classifier | None = None


def get_retriever() -> Retriever:
    """Create the search store once and reuse it for every ticket."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_classifier() -> Classifier:
    """Load the classifier once and reuse it for every ticket."""
    global _classifier
    if _classifier is None:
        _classifier = Classifier()
    return _classifier


@dataclass
class Result:
    ticket_id: str
    action: str                       # what we did: auto_respond, escalate or block
    reason: str                       # why, in words a support manager can read
    intent: str | None = None         # what the ticket is about (added in B-06)
    urgency: str | None = None        # high, medium or low (added in B-06)
    confidence: float | None = None   # how sure we are, from 0 to 1 (added in B-06)
    sources: list[str] = field(default_factory=list)   # document passages used (added in B-04)
    answer: str | None = None         # reply sent to the customer, only for auto_respond (B-08)
    guardrails: dict[str, str] = field(default_factory=dict)  # each safety check: pass or block (B-09)
    error: str | None = None          # filled in if a step failed and the ticket went to a person
    alternatives: list[dict[str, Any]] = field(default_factory=list)  # other intents considered (B-06)
    classifier_method: str | None = None  # how the intent was decided (B-06): see src/classify.py

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def process(ticket: Ticket) -> Result:
    """Run one ticket through the steps built so far. For now every ticket goes to a person."""
    if "empty_body" in ticket.issues:
        return Result(ticket.ticket_id, "escalate", "The ticket has no message text, so a person needs to look at it.")

    label = get_classifier().classify(ticket)
    passages = get_retriever().search(ticket.text)
    sources = [p.passage_id for p in passages]
    about = f"Looks like {label.intent.replace('_', ' ')} ({label.urgency} urgency, {label.confidence:.0%} sure). "
    if not passages:
        reason = about + "No help article matched closely enough, so a person will handle it."
    else:
        reason = (about + f"Found {len(passages)} matching help passages, but automatic answers are not "
                  "built yet, so a person will handle this ticket.")
    return Result(ticket.ticket_id, "escalate", reason, intent=label.intent, urgency=label.urgency,
                  confidence=label.confidence, sources=sources, alternatives=label.alternatives,
                  classifier_method=label.method)
