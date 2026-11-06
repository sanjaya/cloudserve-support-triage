"""Tests for the single-ticket API (B-23). No real model: the same fake model as tests/test_graph.py."""
import json

import pytest
from fastapi.testclient import TestClient

from src import api, config, kill_switch, pipeline
from tests.test_graph import TwoPromptLLM, steps, sure  # noqa: F401  (steps and sure are fixtures)

RAW = {t["ticket_id"]: t for t in json.loads((config.DATA_DIR / "validation_tickets.json").read_text(encoding="utf-8"))}


@pytest.fixture()
def client():
    kill_switch.set_switch(False, "test")
    yield TestClient(api.app)
    kill_switch.set_switch(False, "test")


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["kill_switch_on"] is False
    assert body["confidence_threshold"] == config.CONFIDENCE_THRESHOLD
    assert body["decision_log_run_id"].startswith("api_")


def test_ticket_is_answered_logged_and_can_be_read_back(client, sure, monkeypatch):  # noqa: F811
    monkeypatch.setattr(pipeline, "_llm_client", TwoPromptLLM("answered"))
    out = client.post("/tickets", json=RAW[sure.ticket_id]).json()
    assert out["action"] == "auto_respond" and out["answer"]
    assert out["sources"] and set(out["safety_checks"].values()) == {"pass"}
    assert out["decision_log"]["logged"] is True

    back = client.get(f"/tickets/{sure.ticket_id}").json()
    assert back["run_id"] == api.RUN_ID
    assert [s["step"] for s in back["steps"]][-1] == "final"


def test_kill_switch_applies_to_the_api_too(client, sure, monkeypatch):  # noqa: F811
    monkeypatch.setattr(pipeline, "_llm_client", TwoPromptLLM("answered"))
    kill_switch.set_switch(True, "test", "api drill")
    assert client.get("/health").json()["kill_switch_on"] is True
    out = client.post("/tickets", json=RAW[sure.ticket_id]).json()
    assert out["action"] == "escalate" and out["routing_rule"] == "kill_switch" and out["handoff"]["summary"]


@pytest.mark.parametrize("bad", [{}, {"ticket_id": "BROKEN-1", "body": ""}, ["not", "a", "ticket"], "just text"])
def test_broken_ticket_goes_to_a_person_never_an_error(client, bad):
    r = client.post("/tickets", json=bad)
    assert r.status_code == 200
    out = r.json()
    assert out["action"] == "escalate" and out["reason"] and out["handoff"]


def test_unknown_ticket_history_is_404(client):
    assert client.get("/tickets/NO-SUCH-TICKET").status_code == 404


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200 and "tickets_processed" in r.text
