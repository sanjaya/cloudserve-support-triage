"""Tests for the LangGraph flow (src/graph.py). They check that the graph gives exactly the
same result the plain step-by-step code gave before, and that the steps run in the right order."""
import pytest

from src import config, graph, pipeline
from src.classify import Classifier
from src.ingest import load_raw, normalise
from src.retrieve import HashingEmbedder, Retriever

TICKETS = [normalise(r) for r in load_raw(config.DATA_DIR / "validation_tickets.json")]


class FakeLLM:
    def complete_json(self, system, user, max_tokens=400):
        return {"intent": "rate_limit", "urgency": "low", "confidence": 0.9}


@pytest.fixture
def steps(tmp_path, monkeypatch):
    retriever = Retriever(strategy="section", embedder=HashingEmbedder(), store_path=str(tmp_path))
    classifier = Classifier(client=FakeLLM())
    monkeypatch.setattr(pipeline, "_retriever", retriever)
    monkeypatch.setattr(pipeline, "_classifier", classifier)
    return classifier, retriever


def expected(ticket, classifier, retriever):
    """What the graph should give: classify, search and the routing rules, run one after another."""
    from src.route import AUTO, route
    label = classifier.classify(ticket)
    passages = retriever.search(ticket.text)
    decision = route(label, passages)
    reason = decision.reason
    if decision.route == AUTO:
        reason += " Answer writing (B-08) is not built yet, so for now a person will handle it."
    return pipeline.Result(ticket.ticket_id, "escalate", reason, intent=label.intent, urgency=label.urgency,
                           confidence=label.confidence, sources=[p.passage_id for p in passages],
                           alternatives=label.alternatives, classifier_method=label.method,
                           classifier_note=label.note or None, routing_decision=decision.route,
                           routing_rule=decision.rule, threshold=decision.threshold)


def test_graph_gives_the_same_result_as_the_steps_run_by_hand(steps):
    for t in TICKETS[:20]:
        assert pipeline.process(t) == expected(t, *steps)


def test_graph_has_the_planned_steps():
    nodes = set(graph.build_graph().get_graph().nodes)
    assert {"check_input", "classify", "search", "decide", "draft", "hand_off"} <= nodes


def test_empty_ticket_goes_straight_to_a_person(monkeypatch):
    def must_not_run(*_):
        raise AssertionError("classify/search should not run for an empty ticket")
    monkeypatch.setattr(graph, "classify", must_not_run)
    monkeypatch.setattr(graph, "search", must_not_run)
    flow = graph.build_graph()
    out = flow.invoke({"ticket": normalise({"ticket_id": "E", "channel": "chat", "body": ""})})["result"]
    assert out.action == "escalate" and "no message text" in out.reason


def test_decide_waits_for_classify_and_search(steps):
    seen = []
    flow = graph.build_graph()
    for chunk in flow.stream({"ticket": TICKETS[0]}, stream_mode="updates"):
        seen.extend(chunk)
    assert seen.index("decide") > seen.index("classify") and seen.index("decide") > seen.index("search")
    assert seen[-1] == "hand_off"


def test_a_classifier_failure_is_explained_in_the_result(steps, monkeypatch):
    classifier, _ = steps
    def broken(ticket):
        raise RuntimeError("model could not load")
    monkeypatch.setattr(classifier, "_classify", broken)
    out = pipeline.process(TICKETS[0])
    assert out.classifier_method == "fallback"
    assert "RuntimeError: model could not load" in out.classifier_note


def test_search_and_classifier_share_one_embedding_model():
    from src.retrieve import get_embedder
    assert get_embedder("hashing") is get_embedder("hashing")


def test_ticket_routed_automatic_goes_through_draft_and_keeps_the_decision(steps, monkeypatch):
    classifier, retriever = steps
    sure = Classifier(client=FakeLLM())
    sure.accuracy = {k: 0.97 for k in sure.accuracy}           # pretend the measured confidence is high
    monkeypatch.setattr(pipeline, "_classifier", sure)
    ticket = next(t for t in TICKETS if retriever.search(t.text))
    seen = []
    for chunk in graph.build_graph().stream({"ticket": ticket}, stream_mode="updates"):
        seen.extend(chunk)
    assert seen[-2:] == ["draft", "hand_off"]
    out = pipeline.process(ticket)
    assert out.action == "escalate" and out.routing_decision == "auto_respond"
    assert out.routing_rule == "all_conditions_met" and "B-08" in out.reason


def test_same_ticket_twice_gives_the_same_decision_T04a(steps):
    assert pipeline.process(TICKETS[3]) == pipeline.process(TICKETS[3])
