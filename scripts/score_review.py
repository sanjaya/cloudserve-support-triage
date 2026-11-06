"""B-22: score the human review once both reviewers have filled in the workbook.

    python scripts/score_review.py --review evaluation/results/review_<...>

Reads review_workbook.xlsx (sheets "Reviewer A" and "Reviewer B") and review_items.json, and
writes review_results.md with:
- hallucination rate: share of answers with an unsupported claim, per reviewer and where both
  agree (NFR-03 target: 5% or less), with the 95% range;
- citation accuracy: share whose citations support the statements (NFR-03 target: 95% or more);
- agreement between the two reviewers (percent and Cohen's kappa) - NFR-03 asks for it;
- agreement with the automatic judge PR-08 (overall score within 1 point; and whether PR-08's
  correctness of 2 or less matches a reviewer finding an unsupported claim);
- how the grounding check (PR-05) compares with the reviewers: of the answers it stopped, how
  many really had an unsupported claim, and of those it let through, how many did.
Only rows both reviewers completed are counted; the number is stated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

COLS = {"unsupported": 6, "citations_ok": 7, "correctness": 8, "completeness": 9, "clarity": 10, "tone": 11,
        "overall": 12}


def wilson(k, n, z=1.96):
    if not n:
        return (None, None)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1))


def pct(k, n):
    return round(100 * k / n, 1) if n else None


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 2) if pe < 1 else 1.0


def read_sheet(ws) -> dict[int, dict]:
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        rec = {}
        for k, i in COLS.items():
            v = row[i]
            if k in ("unsupported", "citations_ok"):
                v = str(v).strip().upper()[:1] if v is not None else None
                v = {"Y": True, "N": False}.get(v)
            else:
                try:
                    v = int(v) if v is not None else None
                except (TypeError, ValueError):
                    v = None
            rec[k] = v
        out[int(row[0])] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--review", required=True)
    args = ap.parse_args()
    folder = Path(args.review)
    from openpyxl import load_workbook
    wb = load_workbook(folder / "review_workbook.xlsx", data_only=True)
    items = json.loads((folder / "review_items.json").read_text(encoding="utf-8"))
    A, B = read_sheet(wb["Reviewer A"]), read_sheet(wb["Reviewer B"])
    done = [n for n in range(1, len(items) + 1)
            if A.get(n, {}).get("unsupported") is not None and B.get(n, {}).get("unsupported") is not None]
    if not done:
        sys.exit("No row has been completed by both reviewers yet.")
    ua = [A[n]["unsupported"] for n in done]
    ub = [B[n]["unsupported"] for n in done]
    both = sum(x and y for x, y in zip(ua, ub))
    either = sum(x or y for x, y in zip(ua, ub))
    cit = [n for n in done if A[n]["citations_ok"] is not None and B[n]["citations_ok"] is not None]
    cit_ok = sum(A[n]["citations_ok"] and B[n]["citations_ok"] for n in cit)
    lines = [f"# Human review results (B-22): {folder.name}", "",
             f"{len(done)} of {len(items)} answers completed by both reviewers "
             f"({sum(items[n - 1]['sent'] for n in done)} were sent to customers, the rest stopped by the safety checks).", "",
             "| Measure | Result | 95% range | Target |", "|---|---|---|---|",
             f"| Hallucination rate – both reviewers found an unsupported claim | {pct(both, len(done))}% ({both}) | "
             f"{wilson(both, len(done))[0]}–{wilson(both, len(done))[1]}% | ≤ 5% (NFR-03) |",
             f"| Hallucination rate – at least one reviewer did | {pct(either, len(done))}% ({either}) | "
             f"{wilson(either, len(done))[0]}–{wilson(either, len(done))[1]}% | |",
             f"| Reviewer A / Reviewer B alone | {pct(sum(ua), len(done))}% / {pct(sum(ub), len(done))}% | | |",
             f"| Citation accuracy – both reviewers say citations support the answer | {pct(cit_ok, len(cit))}% "
             f"({cit_ok} of {len(cit)}) | {wilson(cit_ok, len(cit))[0]}–{wilson(cit_ok, len(cit))[1]}% | ≥ 95% (NFR-03) |",
             f"| Agreement between reviewers on 'unsupported claim' | {pct(sum(x == y for x, y in zip(ua, ub)), len(done))}% "
             f"(kappa {kappa(ua, ub)}) | | reported (NFR-03) |"]
    for who, sheet in (("A", A), ("B", B)):
        means = {k: round(sum(sheet[n][k] for n in done if sheet[n][k]) / max(1, sum(1 for n in done if sheet[n][k])), 2)
                 for k in ("correctness", "completeness", "clarity", "tone", "overall")}
        lines.append(f"| Reviewer {who} mean scores (1-5) | {means} | | |")

    judged = [n for n in done if isinstance(items[n - 1].get("pr08"), dict) and "overall" in items[n - 1]["pr08"]]
    if judged:
        close = sum(abs(int(items[n - 1]["pr08"]["overall"]) - (A[n]["overall"] or 0)) <= 1 and
                    abs(int(items[n - 1]["pr08"]["overall"]) - (B[n]["overall"] or 0)) <= 1
                    for n in judged if A[n]["overall"] and B[n]["overall"])
        flag = [int(items[n - 1]["pr08"].get("correctness", 5)) <= 2 for n in judged]
        human = [A[n]["unsupported"] and B[n]["unsupported"] for n in judged]
        lines += ["", "## Agreement with the automatic judge (PR-08)", "",
                  f"- Overall score within 1 point of both reviewers: {pct(close, len(judged))}% ({close} of {len(judged)})",
                  f"- PR-08 correctness ≤ 2 matches 'both reviewers found an unsupported claim': "
                  f"{pct(sum(x == y for x, y in zip(flag, human)), len(judged))}% (kappa {kappa(flag, human)})"]

    stopped = [n for n in done if items[n - 1]["guardrails"].get("grounding") in ("block", "error")]
    passed = [n for n in done if items[n - 1]["guardrails"].get("grounding") == "pass"]
    real = lambda ns: sum(A[n]["unsupported"] and B[n]["unsupported"] for n in ns)  # noqa: E731
    lines += ["", "## How the grounding check (PR-05) compares with the reviewers", "",
              f"- Answers PR-05 stopped: {len(stopped)}; of these, both reviewers found an unsupported claim in "
              f"{real(stopped)} ({pct(real(stopped), len(stopped))}%). The rest were good answers stopped wrongly.",
              f"- Answers PR-05 let through: {len(passed)}; of these, both reviewers found an unsupported claim in "
              f"{real(passed)} ({pct(real(passed), len(passed))}%).", "",
              "## Disagreements to discuss", ""]
    lines += [f"- No {n} ({items[n - 1]['ticket_id']}): A says {'unsupported' if A[n]['unsupported'] else 'supported'}, "
              f"B says {'unsupported' if B[n]['unsupported'] else 'supported'}" for n, x, y in zip(done, ua, ub) if x != y] or ["- none"]
    report = "\n".join(lines) + "\n"
    (folder / "review_results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {folder / 'review_results.md'}")


if __name__ == "__main__":
    main()
