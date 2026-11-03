"""Tests for classifying tickets (FR-02, B-06). Test ids T-02a, T-02b and T-02c come from the
Stage 3 specification. They use the offline stand-in embedder and a fake language model,
so they need no internet and no API key.

Design under test (see src/classify.py): the language model decides the intent, the trained
classifier checks it, and the confidence is the measured accuracy of the agreement case."""
import json

import pytest

from evaluation.metrics import calibration
from src import config
from src.classify import (LEVELS, UNCALIBRATED, Classifier, TrainedClassifier, agreement_level,
                          near_duplicate_groups)
from src.ingest import load_raw, normalise
from src.llm import LLMClient, LLMError, fill, load_prompt, parse_json, prompt_template, render
from src.retrieve import HashingEmbedder

RAW = load_raw(config.DATA_DIR / "validation_tickets.json")
TICKETS = [normalise(r) for r in RAW]
MEASURED = {"agree": 0.93, "in_top3": 0.71, "disagree": 0.42, "llm_only": 0.76, "trained_only": 0.47}


class FakeLLM:
    """Stands in for the language model. Returns a fixed reply, or raises."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def complete_json(self, system, user, max_tokens=400):
        self.calls += 1
        assert "<ticket" in user                      # customer text stays inside the ticket tags
        if self.error:
            raise self.error
        return self.reply


def says(intent, urgency="low", confidence=0.99):
    return FakeLLM({"intent": intent, "urgency": urgency, "confidence": confidence})


@pytest.fixture(scope="module")
def trained():
    return TrainedClassifier.fit([t.text for t in TICKETS], [r["labels"]["intent"] for r in RAW],
                                 [r["labels"]["urgency"] for r in RAW], embedder=HashingEmbedder())


@pytest.fixture
def calib(tmp_path):
    path = tmp_path / "agreement_calibration.json"
    path.write_text(json.dumps({"urgency_source": "llm",
                                "levels": {k: {"accuracy": v} for k, v in MEASURED.items()}}))
    return str(path)


def ranked_for(trained, i=0):
    return trained.predict(TICKETS[i].text)[0]


def test_T02a_every_ticket_gets_valid_labels_and_confidence(trained, calib):
    clf = Classifier(trained=trained, client=says("rate_limit"), calibration_path=calib)
    for t in TICKETS:
        c = clf.classify(t)
        assert c.intent in config.INTENTS and c.urgency in config.URGENCIES
        assert 0.0 <= c.confidence <= 1.0
        assert c.method in LEVELS
        assert len(c.alternatives) <= 2 and all(a["intent"] != c.intent for a in c.alternatives)


def test_agreement_cases_get_their_measured_confidence(trained, calib):
    ranked = ranked_for(trained)
    for llm_choice, level in [(ranked[0][0], "agree"), (ranked[1][0], "in_top3"), (ranked[-1][0], "disagree")]:
        c = Classifier(trained=trained, client=says(llm_choice), calibration_path=calib).classify(TICKETS[0])
        assert (c.intent, c.method, c.confidence) == (llm_choice, level, MEASURED[level])
        assert c.llm_intent == llm_choice and c.trained_intent == ranked[0][0]
        assert c.prompt_version == "PR-02 v1.0"


def test_the_language_model_decides_even_when_it_disagrees(trained, calib):
    outsider = ranked_for(trained)[-1][0]
    c = Classifier(trained=trained, client=says(outsider), calibration_path=calib).classify(TICKETS[0])
    assert c.intent == outsider and c.confidence == MEASURED["disagree"]


def test_language_model_down_uses_trained_answer(trained, calib):
    c = Classifier(trained=trained, client=FakeLLM(error=LLMError("provider down")),
                   calibration_path=calib).classify(TICKETS[0])
    assert c.method == "trained_only" and c.intent == ranked_for(trained)[0][0]
    assert c.confidence == MEASURED["trained_only"] and "provider down" in c.note


def test_no_trained_model_uses_language_model_only(calib):
    c = Classifier(client=says("rate_limit"), calibration_path=calib).classify(TICKETS[0])
    assert (c.method, c.intent, c.confidence) == ("llm_only", "rate_limit", MEASURED["llm_only"])


def test_T02b_neither_method_available_falls_back_safely():
    c = Classifier(client=FakeLLM(error=LLMError("provider down"))).classify(TICKETS[0])
    assert (c.intent, c.confidence, c.method) == ("unclear_request", 0.0, "fallback")


def test_T02b_empty_ticket_falls_back():
    c = Classifier(client=says("rate_limit")).classify(normalise({"ticket_id": "E", "channel": "chat", "body": ""}))
    assert c.method == "fallback" and c.intent == "unclear_request"


def test_T02b_unknown_label_from_language_model_is_treated_as_unavailable(trained, calib):
    c = Classifier(trained=trained, client=says("made_up"), calibration_path=calib).classify(TICKETS[0])
    assert c.method == "trained_only" and "unknown intent" in c.note


def test_without_measured_confidence_everything_is_low(trained):
    clf = Classifier(trained=trained, client=says(ranked_for(trained)[0][0]))   # no calibration file
    c = clf.classify(TICKETS[0])
    assert c.method == "agree" and c.confidence == UNCALIBRATED["agree"] <= 0.3
    assert "evaluate_classifier.py" in c.note


def test_urgency_source_follows_the_measurement(trained, tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"urgency_source": "trained", "levels": {}}))
    c = Classifier(trained=trained, client=says(ranked_for(trained)[0][0], urgency="high"),
                   calibration_path=str(path)).classify(TICKETS[0])
    assert c.urgency == trained.predict(TICKETS[0].text)[1]


def test_agreement_level():
    ranked = [("a", 0.5), ("b", 0.3), ("c", 0.1), ("d", 0.1)]
    assert [agreement_level(x, ranked) for x in "abcd"] == ["agree", "in_top3", "in_top3", "disagree"]


def test_saved_model_gives_the_same_ranking(trained, tmp_path, monkeypatch):
    path = tmp_path / "clf.joblib"
    trained.save(path)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "hashing")
    loaded = Classifier(model_path=str(path), client=says("rate_limit"))
    assert loaded.trained is not None
    for t in TICKETS[:5]:
        assert loaded.trained.predict(t.text) == trained.predict(t.text)


def test_model_trained_with_another_embedder_is_not_used(trained, tmp_path, monkeypatch):
    path = tmp_path / "clf.joblib"
    trained.save(path)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    clf = Classifier(model_path=str(path), client=says("rate_limit"))
    assert clf.trained is None and "not used" in clf.load_note


def test_near_duplicates_share_a_group():
    groups = near_duplicate_groups([
        "builds that work last week are now fail during dependency resolution",
        "builds that worked last week now fail during dependency resolution",
        "please add per-project spend caps",
    ])
    assert groups[0] == groups[1] != groups[2]


def test_T02c_calibration_table():
    table = calibration([(0.9, True)] * 9 + [(0.9, False)] + [(0.3, False)] * 2)
    top = next(b for b in table if b["band"] == "0.8-1.0")
    assert top["stated_pct"] == 90.0 and top["observed_pct"] == 90.0 and top["gap_points"] == 0.0


# ---- the language-model helpers ------------------------------------------------------


def test_prompt_file_has_system_and_user_parts():
    system, user = load_prompt("build/PR-02_classify_intent_urgency_v1.0.md")
    assert "ROLE:" in system and "{body}" in user


def test_langchain_prompt_gives_exactly_the_same_text_as_plain_filling():
    # Same text means the same cache key, so answers saved before the LangChain move still count.
    system, template = load_prompt("build/PR-02_classify_intent_urgency_v1.0.md")
    for t in TICKETS:
        values = {"channel": t.channel, "subject": t.subject, "body": t.body}
        assert render("build/PR-02_classify_intent_urgency_v1.0.md", values) == (system, fill(template, values))


def test_every_prompt_file_loads_as_a_langchain_template():
    import re
    for path in sorted((config.ROOT / "prompts").rglob("PR-*.md")):
        rel = str(path.relative_to(config.ROOT / "prompts"))
        _, user = load_prompt(rel)
        assert set(prompt_template(rel).input_variables) == set(re.findall(r"\{([a-z_]+)\}", user))


def test_customer_text_with_braces_is_kept_as_written():
    values = {"channel": "chat", "subject": "", "body": 'my config is {"region": "eu"} and {body}'}
    _, user = render("build/PR-02_classify_intent_urgency_v1.0.md", values)
    assert '{"region": "eu"} and {body}' in user


def test_fill_keeps_json_braces():
    out = fill('{"intent": "x"} <ticket>{body}</ticket>', {"body": "hello"})
    assert out == '{"intent": "x"} <ticket>hello</ticket>'


def test_parse_json_ignores_fences_and_extra_words():
    assert parse_json('Sure!\n```json\n{"intent": "rate_limit", "n": {"a": 1}}\n```') == {"intent": "rate_limit", "n": {"a": 1}}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_cached_reply_needs_no_call(tmp_path):
    client = LLMClient(api_key="test", cache_path=str(tmp_path / "c.sqlite"))
    with pytest.raises(LLMError):                          # calls are switched off in tests
        client.complete("s", "u")
    import hashlib
    key = hashlib.sha256(json.dumps([client.provider, client.model, "s", "u", 400, config.LLM_REASONING_EFFORT]).encode()).hexdigest()
    client._cache.put(key, "saved answer")
    assert client.complete("s", "u") == "saved answer"
    assert client.cache_hits == 1 and client.calls_made == 0


def _rate_limited():
    import httpx
    import openai
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)


class _FakeChatModel:
    """Pretends to be LangChain's ChatOpenAI: fails `fails` times with 'too many requests',
    then answers."""

    def __init__(self, fails):
        self.fails, self.calls = fails, 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        self.calls += 1
        assert [m.type for m in messages] == ["system", "human"]
        if self.calls <= self.fails:
            raise _rate_limited()
        return AIMessage(content="ok")


def test_rate_limit_is_retried_then_succeeds(tmp_path, monkeypatch):
    client = LLMClient(api_key="test", cache_path=str(tmp_path / "c.sqlite"),
                       max_attempts=2, min_interval_s=0)
    fake = _FakeChatModel(fails=1)
    monkeypatch.setattr(client, "_chat_model", lambda max_tokens: fake)
    assert client.complete("s", "u2") == "ok"
    assert fake.calls == 2 and client.calls_made == 1


def test_rate_limit_that_never_ends_raises_llm_error(tmp_path, monkeypatch):
    client = LLMClient(api_key="test", cache_path=str(tmp_path / "c.sqlite"),
                       max_attempts=2, min_interval_s=0)
    monkeypatch.setattr(client, "_chat_model", lambda max_tokens: _FakeChatModel(fails=99))
    with pytest.raises(LLMError, match="kept failing"):
        client.complete("s", "u3")


def test_chat_model_points_at_openrouter_with_temperature_0(monkeypatch):
    monkeypatch.setattr(config, "LLM_DISABLED", False)
    model = LLMClient(api_key="test", min_interval_s=0)._chat_model(200)
    assert type(model).__name__ == "ChatOpenAI"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.temperature == 0 and model.max_retries == 0
    assert model.extra_body["max_tokens"] == 200
