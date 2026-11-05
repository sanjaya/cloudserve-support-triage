"""B-09: compare versions of the grounding check (PR-05) on development answers.

    python scripts/evaluate_grounding.py                 # compares PR-05 v1.0 and v1.1
    python scripts/evaluate_grounding.py --versions 1.1  # one version only

Uses development data only – never the validation or hidden tickets.

Two sets of answers:
1. Real answers: the development answers that PR-03 v1.1 wrote and the draft step would send, from
   evaluation/results/answer_comparison.csv (written by scripts/evaluate_answers.py). Search is run
   again for each ticket, so the check sees the same passages the drafter saw. A good version
   passes most of these. They are not all correct, so the pass rate is reported separately for
   tickets the labels say are answerable from the articles and those they say are not; any
   must_not_claim phrase from data/ground_truth_responses.json is also reported.
2. Engineered answers from scripts/check_guardrails.py: the unsupported claim, the refund promise
   and the fix-and-date promise must each be blocked by the grounding check; the two control
   answers must pass.

Each check costs one model call; results are cached. Writes evaluation/results/grounding_comparison.md
and grounding_comparison.csv (every verdict, for reading).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_guardrails import CASES, PASSAGES as ALL_PASSAGES  # noqa: E402

from src import config  # noqa: E402
from src.generate import DISCLOSURE  # noqa: E402
from src.guardrails import GROUNDING_PROMPTS, check_grounding  # noqa: E402
from src.ingest import normalise  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieve import Retriever  # noqa: E402

GROUNDING_CASES = {"Unsupported claim", "Refund promise", "Fix and date promise"}


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def run_check(answer, passages, client, version):
    try:
        outcome, note = check_grounding(answer, passages, client, version)
    except Exception as exc:          # in the system, an error also stops the answer
        outcome, note = "error", f"{type(exc).__name__}: {exc}"
    return outcome, note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--answers", default=str(config.ROOT / "evaluation" / "results" / "answer_comparison.csv"))
    ap.add_argument("--tickets", default=str(config.DATA_DIR / "development_tickets.json"))
    ap.add_argument("--answer-version", default="1.1", help="which PR-03 version's answers to check")
    ap.add_argument("--versions", nargs="+", default=list(GROUNDING_PROMPTS))
    ap.add_argument("--pace", type=float, default=config.LLM_MIN_INTERVAL_S)
    args = ap.parse_args()
    if not Path(args.answers).exists():
        sys.exit(f"{args.answers} not found. Run: python scripts/evaluate_answers.py")

    raw = {r["ticket_id"]: r for r in json.loads(Path(args.tickets).read_text(encoding="utf-8"))}
    real = [r for r in csv.DictReader(open(args.answers, encoding="utf-8"))
            if r["version"] == args.answer_version and r["sent"] == "True"]
    retriever = Retriever()
    real_cases = [(r, retriever.search(normalise(raw[r["ticket_id"]]).text)) for r in real]
    print(f"{len(real_cases)} real development answers, {len(CASES)} engineered answers.")

    client = LLMClient(min_interval_s=args.pace)
    rows, summary = [], {}
    for version in args.versions:
        print(f"Checking with PR-05 v{version} ...")
        mine = []
        for r, passages in real_cases:
            outcome, note = run_check(r["answer"], passages, client, version)
            mine.append({"version": version, "set": "real", "ticket_id": r["ticket_id"],
                         "answerable_from_docs": r["answerable_from_docs"],
                         "must_not_claim_breached": r["must_not_claim_breached"],
                         "outcome": outcome, "note": note, "answer": r["answer"]})
        for name, expected, _body, answer, cited, _intent in CASES:
            outcome, note = run_check(f"{answer}\n\n{DISCLOSURE}", [ALL_PASSAGES[c] for c in cited], client, version)
            mine.append({"version": version, "set": "engineered", "ticket_id": name, "answerable_from_docs": "",
                         "must_not_claim_breached": "", "outcome": outcome, "note": note, "answer": answer})
        rows += mine
        realr = [x for x in mine if x["set"] == "real"]
        ans = [x for x in realr if x["answerable_from_docs"] == "True"]
        unans = [x for x in realr if x["answerable_from_docs"] == "False"]
        eng = {x["ticket_id"]: x["outcome"] for x in mine if x["set"] == "engineered"}
        bad_ok = sum(eng[n] == "block" for n in GROUNDING_CASES)
        controls = [n for n in eng if n.startswith("Control")]
        summary[version] = {
            "real answers passed (want high)": f"{sum(x['outcome'] == 'pass' for x in realr)} of {len(realr)} "
                                               f"({pct(sum(x['outcome'] == 'pass' for x in realr), len(realr))}%)",
            "…answerable tickets passed": f"{sum(x['outcome'] == 'pass' for x in ans)} of {len(ans)}",
            "…'not answerable' tickets passed": f"{sum(x['outcome'] == 'pass' for x in unans)} of {len(unans)}",
            "real answers: outcomes": dict(Counter(x["outcome"] for x in realr)),
            "engineered bad answers blocked (must be 3 of 3)": f"{bad_ok} of {len(GROUNDING_CASES)}",
            "control answers passed (must be 2 of 2)": f"{sum(eng[n] == 'pass' for n in controls)} of {len(controls)}",
            "passed answers breaking must_not_claim": sum(1 for x in realr if x["outcome"] == "pass"
                                                          and x["must_not_claim_breached"]),
        }

    lines = [f"# Grounding check comparison ({len(real_cases)} real development answers, "
             f"{len(CASES)} engineered answers)", "",
             "| Measure | " + " | ".join(f"PR-05 v{v}" for v in args.versions) + " |",
             "|---|" + "---|" * len(args.versions)]
    for measure in next(iter(summary.values())):
        lines.append(f"| {measure} | " + " | ".join(str(summary[v][measure]) for v in args.versions) + " |")
    lines += ["", "Pick the version that passes the most real answers while still blocking all 3 engineered bad",
              "answers. Set it in .env as GROUNDING_PROMPT_VERSION and record the choice in Stage 3 (PR-05",
              "change history) and the Stage 5 revision log. Every verdict is in grounding_comparison.csv."]
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    out = config.ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "grounding_comparison.md").write_text(report, encoding="utf-8")
    with open(out / "grounding_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {out / 'grounding_comparison.md'} and grounding_comparison.csv")


if __name__ == "__main__":
    main()
