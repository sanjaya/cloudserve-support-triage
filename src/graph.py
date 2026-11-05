"""The ticket flow as a LangGraph graph (the orchestration layer named in the Project Brief).

Each step is a "node". The graph says which node runs after which:

    START -> check_input --(no text)-----------------------------------------> hand_off -> END
                         \\-> classify --\\
                         \\-> search   ----> decide --(a person)--------------> hand_off -> END
                                                   \\--(automatic)-> draft --(not usable)-> hand_off -> END
                                                                         \\--(usable)----> reply -> END

- check_input : a ticket with no message text goes straight to a person.
- classify    : what the ticket is about and how urgent (src/classify.py, B-06).
- search      : the matching help-article passages (src/retrieve.py, B-04).
  classify and search do not depend on each other, so they run side by side;
  decide waits until both have finished.
- decide      : answer automatically or send to a person, by the rules in src/route.py (B-07).
- draft       : writes the answer from the passages and checks its citations (src/generate.py, B-08).
- reply       : the answer goes to the customer (action "auto_respond").
- hand_off    : the ticket goes to a person, with the reason and any draft attached.

B-09 adds the safety checks between draft and reply, with a route to hand_off whenever a
check fails. The harness keeps calling pipeline.process().

To print the graph as a Mermaid diagram:  python -m src.graph
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.classify import Classification
from src.generate import Draft, generate
from src.ingest import Ticket
from src.retrieve import Passage
from src.route import AUTO, RouteDecision, route


class TicketState(TypedDict, total=False):
    """What the graph knows about one ticket. Each node reads it and adds its own part."""
    ticket: Ticket
    classification: Classification
    passages: list[Passage]
    decision: RouteDecision    # what decide chose, with the rule and the reason
    answer_draft: Draft        # the written answer and whether it passed its checks
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
    """Apply the routing rules (src/route.py)."""
    decision = route(state["classification"], state["passages"])
    return {"decision": decision, "reason": decision.reason}


def after_decide(state: TicketState) -> str:
    """Which node runs after decide."""
    return "draft" if state["decision"].route == AUTO else "hand_off"


def draft(state: TicketState) -> dict:
    """Write the answer from the passages and check it (src/generate.py)."""
    from src.pipeline import get_llm_client
    d = generate(state["ticket"], state["classification"], state["passages"], client=get_llm_client())
    return {"answer_draft": d, "reason": d.reason}


def after_draft(state: TicketState) -> str:
    """A usable answer goes to the customer; anything else to a person."""
    return "reply" if state["answer_draft"].send else "hand_off"


def _result(state: TicketState, action: str, reason: str):
    from src.pipeline import Result
    label, decision, d = state["classification"], state["decision"], state.get("answer_draft")
    return Result(state["ticket"].ticket_id, action, reason, intent=label.intent,
                  urgency=label.urgency, confidence=label.confidence,
                  sources=[p.passage_id for p in state.get("passages", [])],
                  answer=d.answer if (d and action == AUTO) else None,
                  alternatives=label.alternatives, classifier_method=label.method,
                  classifier_note=label.note or None, routing_decision=decision.route,
                  routing_rule=decision.rule, threshold=decision.threshold,
                  draft=d.to_dict() if d else None)


def reply(state: TicketState) -> dict:
    d = state["answer_draft"]
    return {"result": _result(state, AUTO, state["decision"].reason + " " + d.reason)}


def hand_off(state: TicketState) -> dict:
    from src.pipeline import Result
    ticket = state["ticket"]
    if state.get("classification") is None:      # came straight from check_input
        return {"result": Result(ticket.ticket_id, "escalate",
                                 "The ticket has no message text, so a person needs to look at it.",
                                 routing_decision="escalate", routing_rule="no_text")}
    return {"result": _result(state, "escalate", state["reason"])}


# ------------------------------------------------------------------ the graph


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("check_input", check_input)
    graph.add_node("classify", classify)
    graph.add_node("search", search)
    graph.add_node("decide", decide)
    graph.add_node("draft", draft)
    graph.add_node("reply", reply)
    graph.add_node("hand_off", hand_off)

    graph.add_edge(START, "check_input")
    graph.add_conditional_edges("check_input", has_text, ["classify", "search", "hand_off"])
    graph.add_edge(["classify", "search"], "decide")       # decide waits for both
    graph.add_conditional_edges("decide", after_decide, ["draft", "hand_off"])
    graph.add_conditional_edges("draft", after_draft, ["reply", "hand_off"])   # B-09 goes in between
    graph.add_edge("reply", END)
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
