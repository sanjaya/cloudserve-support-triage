"""The steps one ticket goes through.

THIS IS A SKELETON (B-05). For now every ticket is sent to a person, with a reason.
The real steps are added later: classify (B-06), search documents (B-04),
decide (B-07), write the answer (B-08) and check the answer (B-09).
The `Result` shape will not change, so the harness and the metrics will keep working
as each step is added.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.ingest import Ticket

ACTIONS = ("auto_respond", "escalate", "block")


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def process(ticket: Ticket) -> Result:
    """Run one ticket through all the steps. For now (skeleton) we send every ticket to a person."""
    if "empty_body" in ticket.issues:
        return Result(ticket.ticket_id, "escalate", "The ticket has no message text, so a person needs to look at it.")
    return Result(ticket.ticket_id, "escalate", "Automatic answers are not built yet, so a person will handle this ticket.")
