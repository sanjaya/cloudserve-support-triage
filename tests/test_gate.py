"""Tests for the gate check (B-11) and for running a file shaped like the hidden set."""
import importlib.util
import json

from evaluation import harness
from src import config

spec = importlib.util.spec_from_file_location("check_gate", config.ROOT / "scripts" / "check_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
TICKETS = json.loads((config.DATA_DIR / "validation_tickets.json").read_text())


def _run(tmp_path, tickets, name="t.json"):
    src = tmp_path / name
    if name.endswith(".jsonl"):
        src.write_text("\n".join(json.dumps(t) for t in tickets) + "\n")
    else:
        src.write_text(json.dumps(tickets))
    return src, harness.run(str(src), str(tmp_path / "out"))


def test_a_complete_run_passes_every_check_except_possibly_the_commit_one(tmp_path):
    src, out = _run(tmp_path, TICKETS[:6])
    checks, _ = gate.check(out, src)
    failed = [k for k, (ok, _) in checks.items() if not ok and "committed code" not in k]
    assert failed == []
    meta = json.loads((out / "metrics.json").read_text())["meta"]
    assert "code_matches_commit" in meta


def test_a_dropped_ticket_fails_A9(tmp_path):
    src, out = _run(tmp_path, TICKETS[:4])
    lines = (out / "results.jsonl").read_text().splitlines()
    (out / "results.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    checks, _ = gate.check(out, src)
    assert checks["A9 every ticket processed, none dropped or doubled"][0] is False


def test_unlabelled_jsonl_file_runs_like_the_hidden_set(tmp_path):
    unlabelled = [{k: v for k, v in t.items() if k not in ("labels", "history")} for t in TICKETS[:5]]
    src, out = _run(tmp_path, unlabelled, "hidden_style.jsonl")
    checks, metrics = gate.check(out, src)
    assert checks["A9 every ticket processed, none dropped or doubled"][0]
    assert checks["A10 metrics report written with all four sections"][0]
    assert metrics["governance"]["decision_log_reconciles"] is True
