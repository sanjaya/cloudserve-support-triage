"""Decide: answer automatically, or send to a person (FR-04, FR-05, backlog item B-07).

This step is plain code, with no model in it, so the same ticket always gets the same
decision (test T-04a). The rules are checked in this order; the first one that applies wins:

1. The ticket could not be classified                    -> a person
2. The ticket is about a topic we never answer
   automatically (FR-05: security incidents, compliance
   requests, feature requests, unclear requests)          -> a person, however sure we are
3. We are less sure than the threshold
   (CONFIDENCE_THRESHOLD, chosen with
   scripts/sweep_threshold.py)                            -> a person
4. No help-article passage matched closely enough         -> a person
5. Otherwise                                              -> answer automatically

"Answer automatically" only means the ticket goes on to the next steps: writing the answer
(B-08) and the safety checks (B-09). Either of those can still send it to a person.

The confidence is the measured accuracy of the classifier's agreement case (see
src/classify.py), so in practice the threshold decides which cases may be answered:
with 0.80, only tickets where both classification methods agree (about 97% right).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src import config
from src.classify import Classification
from src.retrieve import Passage

AUTO = "auto_respond"
ESCALATE = "escalate"

# Short names for the rule that decided, used in the metrics report and the decision log.
# ("no_text" is set by the graph for empty tickets, which never reach decide.)
RULES = ("no_text", "not_classified", "never_automate", "low_confidence", "no_passage", "all_conditions_met")


@dataclass(frozen=True)
class RouteDecision:
    route: str            # "auto_respond" or "escalate"
    rule: str             # which rule decided (one of RULES)
    reason: str           # in words a support manager can read
    threshold: float      # the confidence threshold that was applied

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _topic(intent: str) -> str:
    return intent.replace("_", " ")


def route(label: Classification, passages: list[Passage], threshold: float | None = None,
          blocked: frozenset[str] | None = None) -> RouteDecision:
    """Apply the rules above. Never raises for a normal Classification."""
    threshold = config.CONFIDENCE_THRESHOLD if threshold is None else threshold
    blocked = config.BLOCKED_INTENTS if blocked is None else blocked
    sure = f"{label.confidence:.0%} sure"
    needed = f"{threshold:.0%}"

    if label.method == "fallback":
        return RouteDecision(ESCALATE, "not_classified",
                             "The system could not work out what the ticket is about, so a person will handle it.",
                             threshold)
    if label.intent in blocked:
        return RouteDecision(ESCALATE, "never_automate",
                             f"Looks like {_topic(label.intent)}. This topic is never answered automatically "
                             "(FR-05), so a person will handle it.", threshold)
    if label.confidence < threshold:
        return RouteDecision(ESCALATE, "low_confidence",
                             f"Looks like {_topic(label.intent)}, but the system is only {sure} "
                             f"(it needs {needed} to answer on its own), so a person will handle it.",
                             threshold)
    if not passages:
        return RouteDecision(ESCALATE, "no_passage",
                             f"Looks like {_topic(label.intent)} ({sure}), but no help article matched "
                             "closely enough to answer from, so a person will handle it.", threshold)
    return RouteDecision(AUTO, "all_conditions_met",
                         f"Looks like {_topic(label.intent)} ({sure}, at least {needed} needed), and "
                         f"{len(passages)} help passage{'s' if len(passages) != 1 else ''} matched, so it "
                         "can be answered automatically.", threshold)
