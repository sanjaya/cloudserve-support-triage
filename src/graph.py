"""The ticket flow as a LangGraph graph (the orchestration layer named in the Project Brief).

Each step is a "node". The graph says which node runs after which:

    START -> check_input --(no text)--------------------------> hand_off -> END
                         \\-> classify --\\
                         \\-> search   ----> decide -> hand_off -> END

- check_input : a ticket with no message text goes straight to a person.
- classify    : what the ticket is about and how urgent (src/classify.py, B-06).
- search      : the matching help-article passages (src/retrieve.py, B-04).
  classify and search do not depend on each other, so they run side by side;
  decide waits until both have finished.
- decide      : answer automatically or hand off (B-07, not built yet: it always hands off).
- hand_off    : builds the Result a person will see, with the reason.

When B-07 to B-09 are built, decide gets a second route: draft -> safety checks -> reply,
with a route back to hand_off whenever a check fails. Only this file changes; the harness
keeps calling pipeline.process().

To print the graph as a Mermaid diagram:  python -m src.graph
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.classify import Classification
from src.ingest import Ticket
from src.retrieve import Passage


class TicketState(TypedDict, total=False):
    """What the graph knows about one ticket. Each node reads it and adds its own part."""
    ticket: Ticket
    classification: Classification
    passages: list[Passage]
    route: str                 # where decide sent the ticket: "hand_off" (later also "draft")
    reason: str                # why, in words a support manager can read
    result: Any                # the final pipeline.Result


# ------------------------------------------------------------------ nodes


def check_input(state: TicketState) -> dict:
    """Nothing to change here; the next step (an edge) looks at the ticket."""
    return {}


def has_text(state: TicketState) -> list[str] | str:
    """After check_input: an empty ticket goes to a person; otherwise classify and search."""
    if "empty_body" in state["ticket"].issues:
        return "hand_off"
    return ["classify", "search"]


def classify(state: TicketState) -> dict:
    from src.pipeline import get_classifier
    return {"classification": get_classifier().classify(state["ticket"])}


def search(state: TicketState) -> dict:
    from src.pipeline import get_retriever
    return {"passages": get_retriever().search(state["ticket"].text)}


def decide(state: TicketState) -> dict:
    """B-07 placeholder: every ticket goes to a person, with the reason."""
    label, passages = state["classification"], state["passages"]
    about = f"Looks like {label.intent.replace('_', ' ')} ({label.urgency} urgency, {label.confidence:.0%} sure). "
    if not passages:
        reason = about + "No help article matched closely enough, so a person will handle it."
    else:
        reason = (about + f"Found {len(passages)} matching help passages, but automatic answers are not "
                  "built yet, so a person will handle this ticket.")
    return {"route": "hand_off", "reason": reason}


def after_decide(state: TicketState) -> str:
    """Which node runs after decide. Only "hand_off" for now; B-07 adds "draft"."""
    return state["route"]


def hand_off(state: TicketState) -> dict:
    from src.pipeline import Result
    ticket = state["ticket"]
    label = state.get("classification")
    if label is None:              # came straight from check_input
        return {"result": Result(ticket.ticket_id, "escalate",
                                 "The ticket has no message text, so a person needs to look at it.")}
    return {"result": Result(ticket.ticket_id, "escalate", state["reason"], intent=label.intent,
                             urgency=label.urgency, confidence=label.confidence,
                             sources=[p.passage_id for p in state.get("passages", [])],
                             alternatives=label.alternatives, classifier_method=label.method,
                             classifier_note=label.note or None)}


# ------------------------------------------------------------------ the graph


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("check_input", check_input)
    graph.add_node("classify", classify)
    graph.add_node("search", search)
    graph.add_node("decide", decide)
    graph.add_node("hand_off", hand_off)

    graph.add_edge(START, "check_input")
    graph.add_conditional_edges("check_input", has_text, ["classify", "search", "hand_off"])
    graph.add_edge(["classify", "search"], "decide")       # decide waits for both
    graph.add_conditional_edges("decide", after_decide, ["hand_off"])
    graph.add_edge("hand_off", END)
    return graph.compile()


_graph = None


def get_graph():
    """Build the graph once and reuse it for every ticket."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


if __name__ == "__main__":
    print(get_graph().get_graph().draw_mermaid())
