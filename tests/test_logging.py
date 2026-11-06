"""Tests for the decision log (FR-09, NFR-05, B-10). Test id T-09a comes from the Stage 3
specification: twenty tickets run; every step logged with reason, prompt version and requirement
ids; the counts reconcile."""
import json

from evaluation import harness
from src import config, logging_store
from src.ingest import normalise
from src.pipeline import Result


def _run(tmp_path, tickets):
    src = tmp_path / "tickets.json"
    src.write_text(json.dumps(tickets))
    return harness.run(str(src), str(tmp_path / "out"))


def test_T09a_twenty_tickets_logged_and_counts_reconcile(tmp_path):
    tickets = json.loads((config.DATA_DIR / "validation_tickets.json").read_text())[:20]
    out = _run(tmp_path, tickets)
    report = json.loads((out / "metrics.json").read_text())
    run_id = report["meta"]["run_id"]
    assert run_id.startswith(out.name)
    assert report["governance"]["decisions_logged"] == 20
    assert report["governance"]["decision_log_reconciles"] is True
    for t in tickets:
        rows = logging_store.history(t["ticket_id"], run_id)
        steps = [r["step"] for r in rows]
        assert steps[-1] == "final" and set(steps) <= set(logging_store.STEPS)
        for r in rows:
            assert r["requirement_ids"] and r["input_summary"] and r["system_version"] is not None
        assert rows[-1]["reason"] and rows[-1]["action"] in ("auto_respond", "escalate", "block")


def test_a_failed_ticket_still_gets_a_final_record(tmp_path, monkeypatch):
    def boom(ticket):
        raise RuntimeError("provider down")
    monkeypatch.setattr(harness, "process", boom)
    out = _run(tmp_path, [{"ticket_id": "FAIL-1", "channel": "chat", "body": "hi"}])
    rows = logging_store.history("FAIL-1", json.loads((out / "metrics.json").read_text())["meta"]["run_id"])
    assert [r["step"] for r in rows] == ["handoff", "final"]          # a person still gets a note (B-16)
    assert rows[1]["action"] == "escalate" and "provider down" in rows[1]["details"]["error"]
    assert "provider down" in rows[0]["details"]["system_uncertain_about"] or rows[0]["reason"]


def test_every_step_of_an_answered_ticket_is_recorded_with_prompt_versions():
    ticket = normalise({"ticket_id": "LOG-1", "channel": "chat", "subject": "Key", "body": "401", "customer_id": "C1"})
    result = Result("LOG-1", "auto_respond", "sent", intent="api_key_issue", urgency="medium", confidence=0.97,
                    sources=["DOC-AUTH-004#resolution"], answer="Hello", classifier_method="agree",
                    routing_decision="auto_respond", routing_rule="all_conditions_met", threshold=0.8,
                    draft={"status": "answered", "send": True, "check": "ok", "citations": ["DOC-AUTH-004#resolution"],
                           "prompt_version": "PR-03 v1.1", "reason": "ok", "unanswered_points": []},
                    guardrails={"grounding": "pass", "injection": "pass"})
    assert logging_store.log_decision("run_test_steps", ticket, result, "abc1234") == 6
    rows = logging_store.history("LOG-1", "run_test_steps")
    by = {r["step"]: r for r in rows}
    assert list(by) == ["classify", "search", "route", "draft", "safety", "final"]
    assert by["classify"]["prompt_version"] == "PR-02 v1.0" and by["classify"]["requirement_ids"] == "FR-02"
    assert by["route"]["threshold"] == 0.8 and by["route"]["requirement_ids"] == "FR-04, FR-05"
    assert by["draft"]["prompt_version"] == "PR-03 v1.1"
    assert by["safety"]["guardrail_results"] == {"grounding": "pass", "injection": "pass"}
    assert by["final"]["action"] == "auto_respond" and by["final"]["system_version"] == "abc1234"
    assert "customer C1" in by["final"]["input_summary"]


def test_ticket_handling_can_be_read_back(capsys):
    logging_store.main(["--ticket", "LOG-1", "--run", "run_test_steps"])
    out = capsys.readouterr().out
    assert "[route] FR-04, FR-05" in out and "PR-03 v1.1" in out and "[final]" in out
