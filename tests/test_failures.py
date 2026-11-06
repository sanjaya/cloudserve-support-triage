"""Tests for failure handling (FR-12, B-17). Test ids T-12a, T-12b and T-12c come from the
Stage 3 specification. The provider is simulated, so no internet is needed."""
import json

import pytest

from evaluation import harness
from src import config, pipeline
from src.classify import Classifier
from src.llm import LLMClient, LLMError
from src.retrieve import HashingEmbedder, Retriever

TICKETS = json.loads((config.DATA_DIR / "validation_tickets.json").read_text())


class Provider:
    """A simulated provider for the whole pipeline. It answers every prompt sensibly, but after
    `works_for` questions it is 'disconnected' and every call fails."""

    def __init__(self, works_for=10**9):
        self.works_for, self.asked = works_for, 0

    def complete_json(self, system, user, max_tokens=400):
        import re
        self.asked += 1
        if self.asked > self.works_for:
            raise LLMError("provider error: APIConnectionError: Connection error.")
        if "security filter" in system:
            return {"injection_suspected": False, "confidence": 0.05, "evidence": ""}
        if "strict fact-checker" in system:
            return {"claims": [], "forbidden_commitment": False, "unsupported_count": 0, "decision": "pass"}
        if "hand-off notes" in system:
            return {"summary": "s", "already_provided_by_customer": [], "relevant_articles": [],
                    "system_uncertain_about": "u", "suggested_first_question": "q"}
        if "You draft replies" in system:
            first = re.search(r"\[(DOC-[^\]]+)\]", user)
            if not first:
                return {"status": "insufficient", "answer": "", "citations": [], "unanswered_points": []}
            return {"status": "answered", "answer": f"Hello [{first.group(1)}].", "citations": [first.group(1)],
                    "unanswered_points": []}
        return {"intent": "rate_limit", "urgency": "low", "confidence": 0.9}


@pytest.fixture
def system(tmp_path, monkeypatch):
    def setup(provider):
        clf = Classifier(client=provider)
        clf.accuracy = {k: 0.97 for k in clf.accuracy}
        monkeypatch.setattr(pipeline, "_classifier", clf)
        monkeypatch.setattr(pipeline, "_retriever",
                            Retriever(strategy="section", embedder=HashingEmbedder(), store_path=str(tmp_path / "c")))
        monkeypatch.setattr(pipeline, "_llm_client", provider)
    return setup


def _run(tmp_path, tickets):
    src = tmp_path / "t.json"
    src.write_text(json.dumps(tickets))
    out = harness.run(str(src), str(tmp_path / "out"))
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    return rows, json.loads((out / "metrics.json").read_text())


def test_T12a_provider_disconnected_mid_run_every_ticket_still_finishes(system, tmp_path):
    system(Provider(works_for=25))
    rows, metrics = _run(tmp_path, TICKETS[:20])
    assert len(rows) == 20 and metrics["volume"]["stage_errors"] == 0          # the run completed
    assert metrics["governance"]["decision_log_reconciles"] is True               # and every ticket is logged
    after = [r["result"] for r in rows[-8:]]
    assert all(r["action"] in ("escalate", "block") for r in after)              # affected tickets go to a person
    assert all(r["handoff"] and r["handoff"]["summary"] for r in after)          # with a hand-off note
    assert all("unavailable" in r["reason"] for r in after)                     # the reason says why


def test_T12c_malformed_tickets_are_escalated_with_a_reason(system, tmp_path):
    system(Provider())
    bad = [None, "just a string", {"channel": "chat"}, {"ticket_id": "B4", "body": 12345, "subject": ["x"]},
           {"ticket_id": "B5", "channel": "pigeon", "body": "   "}]
    rows, metrics = _run(tmp_path, bad + TICKETS[:2])
    assert len(rows) == 7 and metrics["governance"]["decision_log_reconciles"] is True
    for r in rows[:5]:
        assert r["result"]["action"] == "escalate" and r["result"]["reason"] and r["result"]["handoff"]


def test_search_failure_sends_the_ticket_to_a_person_with_the_reason(system, tmp_path, monkeypatch):
    system(Provider())

    def broken(text, *a, **k):
        raise RuntimeError("vector store unreachable")
    monkeypatch.setattr(pipeline._retriever, "search", broken)
    out = pipeline.process(harness.normalise(TICKETS[0]))
    assert out.action == "escalate" and out.routing_rule == "search_failed"
    assert "vector store unreachable" in out.reason and out.handoff["summary"]


class _Down:
    """Pretends to be ChatOpenAI with the provider unreachable."""

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        import httpx
        import openai
        self.calls += 1
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))


def test_circuit_breaker_stops_calling_a_dead_provider(tmp_path, monkeypatch):
    client = LLMClient(api_key="test", cache_path=str(tmp_path / "c.sqlite"), max_attempts=1, min_interval_s=0)
    down = _Down()
    monkeypatch.setattr(client, "_chat_model", lambda max_tokens: down)
    for n in range(5):
        with pytest.raises(LLMError):
            client.complete("s", f"question {n}")
    assert down.calls == config.LLM_CIRCUIT_FAILURES             # later questions never reach the provider
    assert client.usage()["circuit_breaker_opened"] == 1 and client.usage()["failed_calls"] == config.LLM_CIRCUIT_FAILURES


def test_cached_answers_still_work_while_the_circuit_is_open(tmp_path, monkeypatch):
    import hashlib
    client = LLMClient(api_key="test", cache_path=str(tmp_path / "c.sqlite"), max_attempts=1, min_interval_s=0)
    key = hashlib.sha256(json.dumps([client.provider, client.model, "s", "cached", 400,
                                     config.LLM_REASONING_EFFORT]).encode()).hexdigest()
    client._cache.put(key, "saved")
    monkeypatch.setattr(client, "_chat_model", lambda max_tokens: _Down())
    for n in range(3):
        with pytest.raises(LLMError):
            client.complete("s", f"q{n}")
    assert client.complete("s", "cached") == "saved"


def test_T12b_rate_limits_are_counted(tmp_path, monkeypatch):
    from tests.test_classify import _FakeChatModel
    client = LLMClient(api_key="test", cache_path=str(tmp_path / "c.sqlite"), max_attempts=3, min_interval_s=0)
    monkeypatch.setattr(client, "_chat_model", lambda max_tokens: _FakeChatModel(fails=2))
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)     # do not really wait in the test
    assert client.complete("s", "u") == "ok"
    assert client.usage()["rate_limit_events"] == 2 and client.usage()["failed_calls"] == 0


def test_metrics_report_includes_model_usage(system, tmp_path):
    system(Provider())
    src = tmp_path / "t.json"
    src.write_text(json.dumps(TICKETS[:2]))
    out = harness.run(str(src), str(tmp_path / "out"))
    usage = json.loads((out / "metrics.json").read_text())["meta"]["model_usage"]
    assert set(usage) == {"model_calls", "cache_hits", "rate_limit_events", "failed_calls", "circuit_breaker_opened"}
