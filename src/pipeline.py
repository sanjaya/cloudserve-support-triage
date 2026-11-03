"""The steps one ticket goes through.

The order of the steps is a LangGraph graph, defined in src/graph.py. This file holds what
the steps share: the Result shape, and the classifier and search store (loaded once).

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
        if _classifier.trained is not None:
            _classifier.trained.predict("warm up")   # load the embedding model now, once
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
    classifier_note: str | None = None    # anything the classifier wants a person to know, e.g. why it fell back

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def process(ticket: Ticket) -> Result:
    """Run one ticket through the graph (src/graph.py). For now every ticket goes to a person."""
    from src.graph import get_graph
    get_retriever(), get_classifier()      # load both before the graph runs steps side by side
    return get_graph().invoke({"ticket": ticket})["result"]
