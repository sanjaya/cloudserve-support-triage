"""Tests for the fairness audit (B-14) and the human review kit (B-22). No model calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import fairness_audit  # noqa: E402
import prepare_review  # noqa: E402
import score_review  # noqa: E402
from src.retrieve import load_articles, split_by_section  # noqa: E402


def _row(tid, action, fluency="fluent", intent="billing", right=True, text="word " * 10):
    return {"ticket_id": tid, "customer_tier": "standard", "customer_region": "europe", "channel": "email",
            "language_fluency": fluency, "_text": text,
            "labels": {"intent": "billing", "expected_doc_ids": ["KB-1"]},
            "result": {"action": action, "routing_decision": action, "intent": intent if right else "other",
                       "sources": ["KB-1#s1"] if right else ["KB-9#s1"]}}


# ---- fairness audit ------------------------------------------------------------------

def test_fairness_verdicts():
    # 10 fluent tickets all right and answered; 10 non-fluent, none answered, half right.
    rows = [_row(f"F{i}", "auto_respond") for i in range(10)]
    rows += [_row(f"N{i}", "escalate", fluency="non_fluent", right=i % 2 == 0) for i in range(10)]
    out = fairness_audit.audit(rows)
    segs = out["dimensions"]["language_fluency"]["segments"]
    assert segs["fluent"]["resolution"]["value"] == 100.0
    assert segs["non_fluent"]["resolution"]["gap_points"] == 100.0
    assert segs["non_fluent"]["resolution"]["verdict"] == "real gap"
    assert segs["non_fluent"]["quality_intent"]["value"] == 50.0
    assert segs["fluent"]["quality_intent"]["verdict"] == "within 5 points"


def test_fairness_small_gap_cannot_be_called():
    rows = [_row(f"F{i}", "auto_respond" if i < 3 else "escalate") for i in range(5)]
    rows += [_row(f"N{i}", "auto_respond" if i < 2 else "escalate", fluency="non_fluent") for i in range(5)]
    segs = fairness_audit.audit(rows)["dimensions"]["language_fluency"]["segments"]
    assert segs["non_fluent"]["resolution"]["verdict"].startswith("can't tell")


def test_fairness_report_has_every_dimension():
    rows = [_row(f"F{i}", "auto_respond") for i in range(4)]
    md = fairness_audit.to_markdown(fairness_audit.audit(rows), "run_x", 4, None)
    for title in ("Customer tier", "Region", "English fluency", "Ticket length", "Channel"):
        assert f"## {title}" in md


# ---- human review kit ----------------------------------------------------------------

@pytest.fixture()
def run_folder(tmp_path):
    """A small finished run: 3 sent answers, 4 stopped by the grounding check, 1 escalated."""
    passage = next(p for a in load_articles() for p in split_by_section(a))
    tickets = [{"ticket_id": f"T-{i}", "channel": "email", "subject": "Help", "body": f"Question number {i}?",
                "received_at": "2026-03-02T08:32:00Z", "customer_id": "C1", "customer_name": "A",
                "customer_tier": "standard", "customer_region": "europe", "language_fluency": "fluent"}
               for i in range(8)]
    (tmp_path / "tickets.json").write_text(json.dumps(tickets), encoding="utf-8")
    run = tmp_path / "run_test"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"meta": {"input_file": str(tmp_path / "tickets.json")}}))
    lines = []
    for i in range(8):
        action = "auto_respond" if i < 3 else ("block" if i < 7 else "escalate")
        draft = None if action == "escalate" else {"send": True, "answer": f"Answer {i} [{passage.passage_id}]",
                                                   "citations": [passage.passage_id]}
        lines.append(json.dumps({"ticket_id": f"T-{i}", "result": {
            "ticket_id": f"T-{i}", "action": action, "draft": draft, "sources": [passage.passage_id],
            "guardrails": {"grounding": "pass" if action == "auto_respond" else "block"}}}))
    (run / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path, run


class FakeJudge:
    def complete_json(self, system, user, max_tokens=400):
        assert "Answer" in user  # the system answer is in the prompt
        return {"correctness": 4, "completeness": 4, "clarity": 5, "tone": 5, "overall": 4}


def test_prepare_review_picks_sent_first_and_hides_verdicts(run_folder, monkeypatch):
    tmp, run = run_folder
    monkeypatch.setattr(prepare_review, "LLMClient", FakeJudge)
    out = tmp / "review"
    monkeypatch.setattr(sys, "argv", ["prepare_review.py", "--runs", str(run), "--n", "5", "--out", str(out),
                                      "--ground-truth", str(tmp / "none.json")])
    prepare_review.main()
    items = json.loads((out / "review_items.json").read_text(encoding="utf-8"))
    assert len(items) == 5
    assert [i["sent"] for i in items[:3]] == [True, True, True]      # every sent answer first
    assert all(i["ticket_id"] != "T-7" for i in items)                # escalated: no answer to review
    assert items[0]["pr08"]["overall"] == 4 and items[0]["ticket"].startswith("Help")
    assert items[0]["passage_text"]                                     # full passage text for the reviewer

    from openpyxl import load_workbook
    wb = load_workbook(out / "review_workbook.xlsx")
    assert wb.sheetnames == ["How to review", "Reviewer A", "Reviewer B"]
    header = [c.value for c in wb["Reviewer A"][1]]
    assert "Unsupported claim? (Y/N)" in header
    text = " ".join(str(c.value) for row in wb["Reviewer A"].iter_rows() for c in row if c.value)
    assert "grounding" not in text and "pr08" not in text.lower()   # reviewers are not shown the verdicts


def test_score_review(run_folder, monkeypatch, capsys):
    tmp, run = run_folder
    monkeypatch.setattr(prepare_review, "LLMClient", FakeJudge)
    out = tmp / "review"
    monkeypatch.setattr(sys, "argv", ["prepare_review.py", "--runs", str(run), "--n", "7", "--out", str(out),
                                      "--ground-truth", str(tmp / "none.json")])
    prepare_review.main()
    from openpyxl import load_workbook
    wb = load_workbook(out / "review_workbook.xlsx")
    # Both reviewers: answers 4 and 5 (stopped by the grounding check) have an unsupported claim;
    # reviewer B also flags answer 1 (a sent one). Everything else is fine.
    for sheet, flagged in (("Reviewer A", {4, 5}), ("Reviewer B", {1, 4, 5})):
        for row in wb[sheet].iter_rows(min_row=2):
            n = row[0].value
            row[6].value = "Y" if n in flagged else "N"
            row[7].value = "Y"
            for c in row[8:13]:
                c.value = 4
    wb.save(out / "review_workbook.xlsx")
    monkeypatch.setattr(sys, "argv", ["score_review.py", "--review", str(out)])
    score_review.main()
    report = (out / "review_results.md").read_text(encoding="utf-8")
    assert "7 of 7 answers" in report
    assert "both reviewers found an unsupported claim | 28.6% (2)" in report
    assert "at least one reviewer did | 42.9% (3)" in report
    assert "Answers PR-05 stopped: 4; of these, both reviewers found an unsupported claim in 2" in report
    assert "No 1 (T-0)" in report      # the disagreement is listed


def test_kappa():
    assert score_review.kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert score_review.kappa([True, True, False, False], [True, False, True, False]) == 0.0
