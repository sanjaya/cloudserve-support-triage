"""Tests for the run harness (B-05). Test ids T-11a and T-11b come from the Stage 3 specification."""
import json

from evaluation import harness
from evaluation.metrics import per_class
from src import config


def _run(tmp_path, tickets):
    src = tmp_path / "unseen_tickets.json"   # a new file name, like the hidden test file
    src.write_text(json.dumps(tickets))
    return harness.run(str(src), str(tmp_path / "out"))


def test_T11a_every_ticket_processed_from_any_path(tmp_path):
    tickets = json.loads((config.DATA_DIR / "validation_tickets.json").read_text())[:10]
    tickets += [{"ticket_id": "BAD-1", "channel": "email", "body": ""}, None, "garbage"]
    out = _run(tmp_path, tickets)
    lines = (out / "results.jsonl").read_text().splitlines()
    assert len(lines) == len(tickets)          # every ticket has a result; none is lost
    for line in lines:
        rec = json.loads(line)
        assert rec["result"]["action"] in ("auto_respond", "escalate", "block")
        assert rec["result"]["reason"]


def test_T11b_metrics_report_written_with_required_groups(tmp_path):
    out = _run(tmp_path, json.loads((config.DATA_DIR / "validation_tickets.json").read_text())[:5])
    report = json.loads((out / "metrics.json").read_text())
    for group in ("volume", "business", "technical", "governance"):
        assert group in report
    assert report["volume"]["tickets_processed"] == 5
    assert report["meta"]["input_file"].endswith("unseen_tickets.json")
    assert (out / "metrics.md").read_text().startswith("# Metrics report")


def test_a_failing_stage_escalates_and_the_run_continues(tmp_path, monkeypatch):
    def boom(ticket):
        raise RuntimeError("provider down")
    monkeypatch.setattr(harness, "process", boom)
    out = _run(tmp_path, [{"ticket_id": "A", "channel": "chat", "body": "hi"}] * 3)
    recs = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert len(recs) == 3
    assert all(r["result"]["action"] == "escalate" and "provider down" in r["result"]["error"] for r in recs)


def test_per_class_precision_recall():
    m = per_class([("a", "a"), ("a", "b"), ("b", "b")])
    assert m["a"] == {"precision": 1.0, "recall": 0.5, "support": 2}
    assert m["b"]["precision"] == 0.5
