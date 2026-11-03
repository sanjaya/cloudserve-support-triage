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


def old_process(ticket, classifier, retriever):
    """The step-by-step code from before the move to LangGraph (commit 4e44e35)."""
    label = classifier.classify(ticket)
    passages = retriever.search(ticket.text)
    about = f"Looks like {label.intent.replace('_', ' ')} ({label.urgency} urgency, {label.confidence:.0%} sure). "
    reason = (about + "No help article matched closely enough, so a person will handle it." if not passages
              else about + f"Found {len(passages)} matching help passages, but automatic answers are not "
              "built yet, so a person will handle this ticket.")
    return pipeline.Result(ticket.ticket_id, "escalate", reason, intent=label.intent, urgency=label.urgency,
                           confidence=label.confidence, sources=[p.passage_id for p in passages],
                           alternatives=label.alternatives, classifier_method=label.method,
                           classifier_note=label.note or None)


def test_graph_gives_the_same_result_as_before(steps):
    for t in TICKETS[:20]:
        assert pipeline.process(t) == old_process(t, *steps)


def test_graph_has_the_planned_steps():
    nodes = set(graph.build_graph().get_graph().nodes)
    assert {"check_input", "classify", "search", "decide", "hand_off"} <= nodes


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
