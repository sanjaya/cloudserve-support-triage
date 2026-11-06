"""B-22: prepare the human review of 50 replies (NFR-03; test T-06c).

    python scripts/prepare_review.py --runs evaluation/results/run_<gate> evaluation/results/run_<rehearsal>

Picks 50 answers the system wrote, from finished runs:
  1. every answer that was actually sent to a customer;
  2. then answers for tickets that have a senior agent's reference answer
     (data/ground_truth_responses.json), whether sent or stopped by the safety checks;
  3. then other answers stopped by the safety checks, to reach 50 (fixed random order).
Including stopped answers lets the review also say whether the grounding check (PR-05) was
right to stop them.

For each it runs the answer-quality judge, prompt PR-08 (one model call each; cached), and
writes a review folder evaluation/results/review_<date>/ with:
  review_workbook.xlsx  sheets "How to review", "Reviewer A", "Reviewer B": the ticket, the
                        passages the answer cites (full text), the answer and any reference
                        answer, with empty columns for each reviewer. PR-08's scores and the
                        safety-check verdicts are NOT shown, so they cannot sway the reviewers.
  review_items.json     everything, including PR-08's scores and the safety-check verdicts,
                        for scripts/score_review.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.ingest import load_raw, normalise  # noqa: E402
from src.llm import LLMClient, LLMError, render  # noqa: E402
from src.retrieve import load_articles, split_by_section  # noqa: E402

JUDGE = "evaluation/PR-08_answer_quality_judge_v1.0.md"
DEFAULT_MUST_NOT = ["a refund has been issued", "the issue has been fixed on our side", "a specific delivery date for a fix"]
REVIEW_COLUMNS = ["Unsupported claim? (Y/N)", "Citations point to passages that support it? (Y/N)",
                  "Correctness 1-5", "Completeness 1-5", "Clarity 1-5", "Tone 1-5", "Overall 1-5", "Notes"]
GUIDE = [
    "How to review (B-22, about 2 minutes per answer)",
    "",
    "Each row is one answer the system wrote. Read the ticket, then the cited passages, then the answer.",
    "Use the reference answer (where there is one) only as a guide to what a good answer covers.",
    "",
    "Unsupported claim? Y if ANY sentence states a fact, step, number, cause or promise that the cited",
    "  passages do not say (a hallucination). Rewording or putting steps in a list is fine. Otherwise N.",
    "Citations point to passages that support it? Y if the passages cited are the ones the answer's",
    "  statements come from. N if a citation points to a passage that does not support its statement.",
    "Correctness: 5 = nothing wrong; 1 = wrong or harmful. Completeness: does it answer what was asked?",
    "Clarity: plain and usable by a non-fluent reader. Tone: polite, no promises about refunds, fixes or dates.",
    "Overall: your overall judgement, 1-5.",
    "",
    "Two people review the same 50 answers independently (Reviewer A and Reviewer B sheets).",
    "Do not discuss answers until both sheets are done. Then run: python scripts/score_review.py --review <folder>",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--ground-truth", default=str(config.DATA_DIR / "ground_truth_responses.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--no-judge", action="store_true", help="skip PR-08 (no model calls)")
    args = ap.parse_args()

    truth = {}
    if Path(args.ground_truth).exists():
        truth = {g["ticket_id"]: g for g in json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))}
    passages = {p.passage_id: p for a in load_articles() for p in split_by_section(a)}

    candidates = []
    for run in map(Path, args.runs):
        meta = json.loads((run / "metrics.json").read_text(encoding="utf-8"))["meta"]
        input_path = Path(meta["input_file"])
        if not input_path.is_absolute():
            input_path = config.ROOT / input_path
        tickets = {normalise(r).ticket_id: normalise(r) for r in load_raw(input_path)}
        for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines():
            res = json.loads(line)["result"]
            d = res.get("draft") or {}
            if not (d.get("send") and d.get("answer")):
                continue
            t = tickets.get(res["ticket_id"])
            candidates.append({"ticket_id": res["ticket_id"], "run": run.name, "ticket": t.text if t else "",
                               "answer": d["answer"], "citations": d.get("citations") or [],
                               "sources": res.get("sources") or [], "sent": res["action"] == "auto_respond",
                               "action": res["action"], "guardrails": res.get("guardrails") or {},
                               "guardrail_notes": res.get("guardrail_notes") or {},
                               "has_reference": res["ticket_id"] in truth})
    seen, chosen = set(), []
    rng = random.Random(args.seed)
    rest = [c for c in candidates if not c["sent"]]
    rng.shuffle(rest)
    for c in [c for c in candidates if c["sent"]] + sorted(rest, key=lambda c: not c["has_reference"]):
        if c["ticket_id"] not in seen and len(chosen) < args.n:
            seen.add(c["ticket_id"])
            chosen.append(c)
    print(f"{len(candidates)} written answers found; {len(chosen)} chosen "
          f"({sum(c['sent'] for c in chosen)} sent, {sum(c['has_reference'] for c in chosen)} with a reference answer).")

    client = LLMClient()
    for n, c in enumerate(chosen, 1):
        g = truth.get(c["ticket_id"], {})
        c["reference"] = g.get("reference_response") or ""
        c["passage_text"] = "\n\n".join(f"[{pid}]\n{passages[pid].text}" for pid in c["citations"] if pid in passages)
        c["pr08"] = None
        if args.no_judge:
            continue
        try:
            system, user = render(JUDGE, {
                "body": c["ticket"], "system_reply": c["answer"],
                "reference_response": c["reference"] or "No reference answer is available; judge against the ticket.",
                "must_mention": json.dumps(g.get("must_mention") or []),
                "must_not_claim": json.dumps(g.get("must_not_claim") or DEFAULT_MUST_NOT)})
            c["pr08"] = client.complete_json(system, user, max_tokens=400)
        except LLMError as exc:
            c["pr08"] = {"error": str(exc)}
        if n % 10 == 0:
            print(f"  PR-08: {n}/{len(chosen)}")

    out = Path(args.out) if args.out else config.ROOT / "evaluation" / "results" / \
        datetime.now(timezone.utc).strftime("review_%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_items.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False), encoding="utf-8")
    write_workbook(out / "review_workbook.xlsx", chosen)
    print(f"Saved {out / 'review_workbook.xlsx'} and review_items.json")


def write_workbook(path: Path, items: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    guide = wb.active
    guide.title = "How to review"
    for i, line in enumerate(GUIDE, 1):
        guide.cell(row=i, column=1, value=line).font = Font(bold=(i == 1), size=13 if i == 1 else 11)
    guide.column_dimensions["A"].width = 110
    head = ["No", "Ticket ID", "Ticket", "Cited passages", "System answer", "Reference answer (senior agent)"] + REVIEW_COLUMNS
    widths = [5, 13, 45, 60, 60, 50, 14, 16, 12, 12, 12, 12, 12, 30]
    for name in ("Reviewer A", "Reviewer B"):
        ws = wb.create_sheet(name)
        ws.append(head)
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F3A5F")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for n, c in enumerate(items, 1):
            ws.append([n, c["ticket_id"], c["ticket"], c["passage_text"], c["answer"], c["reference"] or "(none)"]
                      + [None] * len(REVIEW_COLUMNS))
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for cell in row[6:13]:
                cell.fill = PatternFill("solid", fgColor="FFF7E0")
        yn = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
        score = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True)
        ws.add_data_validation(yn)
        ws.add_data_validation(score)
        yn.add(f"G2:H{len(items) + 1}")
        score.add(f"I2:M{len(items) + 1}")
        ws.freeze_panes = "C2"
    wb.save(path)


if __name__ == "__main__":
    main()
