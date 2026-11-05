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


class TwoPromptLLM:
    """Answers the classification prompt (PR-02) and the answer prompt (PR-03) differently.
    For PR-03 it cites the first passage it was given, or replies with `status`."""

    def __init__(self, status="answered"):
        self.status = status

    def complete_json(self, system, user, max_tokens=400):
        import re
        if "You draft replies" not in system:
            return {"intent": "rate_limit", "urgency": "low", "confidence": 0.9}
        first = re.search(r"\[(DOC-[^\]]+)\]", user).group(1)
        if self.status != "answered":
            return {"status": self.status, "answer": "", "citations": [], "unanswered_points": ["the plan limits"]}
        return {"status": "answered", "answer": f"Hello, please follow the steps [{first}].",
                "citations": [first], "unanswered_points": []}


@pytest.fixture
def sure(steps, monkeypatch):
    """A classifier whose measured confidence is high, so tickets with passages are routed automatic."""
    clf = Classifier(client=TwoPromptLLM())
    clf.accuracy = {k: 0.97 for k in clf.accuracy}
    monkeypatch.setattr(pipeline, "_classifier", clf)
    return next(t for t in TICKETS if steps[1].search(t.text))


def nodes_run(ticket):
    seen = []
    for chunk in graph.build_graph().stream({"ticket": ticket}, stream_mode="updates"):
        seen.extend(chunk)
    return seen


def test_usable_answer_is_sent_with_citation_and_disclosure(sure, monkeypatch):
    monkeypatch.setattr(pipeline, "_llm_client", TwoPromptLLM("answered"))
    assert nodes_run(sure)[-2:] == ["draft", "reply"]
    out = pipeline.process(sure)
    assert out.action == "auto_respond" and out.routing_decision == "auto_respond"
    assert out.answer.endswith("a member of the support team will follow up.")
    assert set(out.draft["citations"]) <= set(out.sources)              # T-06a
    assert out.draft["send"] is True and out.draft["prompt_version"].startswith("PR-03 v")


def test_insufficient_answer_goes_to_a_person_with_the_draft_T06b(sure, monkeypatch):
    monkeypatch.setattr(pipeline, "_llm_client", TwoPromptLLM("insufficient"))
    assert nodes_run(sure)[-2:] == ["draft", "hand_off"]
    out = pipeline.process(sure)
    assert out.action == "escalate" and out.routing_decision == "auto_respond"
    assert out.answer is None and out.draft["status"] == "insufficient"
    assert "the plan limits" in out.reason


def test_model_down_while_writing_goes_to_a_person(sure, monkeypatch):
    from src.llm import LLMError

    class Down(TwoPromptLLM):
        def complete_json(self, system, user, max_tokens=400):
            raise LLMError("provider down")
    monkeypatch.setattr(pipeline, "_llm_client", Down())
    out = pipeline.process(sure)
    assert out.action == "escalate" and out.draft["check"] == "model_failed" and "provider down" in out.reason


def test_same_ticket_twice_gives_the_same_decision_T04a(steps):
    assert pipeline.process(TICKETS[3]) == pipeline.process(TICKETS[3])
