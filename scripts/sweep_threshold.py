"""B-07 / FR-04: choose the confidence threshold from the development tickets (test T-04b).

    python scripts/sweep_threshold.py

Uses development tickets only – never the validation or hidden tickets.

What it uses:
- evaluation/results/classifier_per_ticket.csv, written by
  `python scripts/evaluate_classifier.py --llm-sample 160`. For 160 development tickets it
  holds the language model's intent and the agreement case, measured on groups of tickets the
  trained classifier had never seen (the honest setting). No new model calls are made.
- models/agreement_calibration.json: the confidence for each agreement case.
- the help-article search, run on the same 160 tickets (needs the embedding model).

For each threshold it applies the real routing rules (src/route.py) and reports:
- how many tickets would be answered automatically,
- how many of those had the right intent (a wrong intent means a wrong answer),
- how many of those the labels say should have gone to a person, and why
  (not answerable from the articles, or a never-automate topic),
- routing accuracy against labels.expected_route.

Writes evaluation/results/threshold_sweep.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.classify import Classification, load_calibration  # noqa: E402
from src.ingest import normalise  # noqa: E402
from src.retrieve import Retriever  # noqa: E402
from src.route import AUTO, route  # noqa: E402

THRESHOLDS = [round(x / 100, 2) for x in range(25, 100, 5)]


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickets", default=str(config.DATA_DIR / "development_tickets.json"))
    ap.add_argument("--per-ticket", default=str(config.ROOT / "evaluation" / "results" / "classifier_per_ticket.csv"))
    args = ap.parse_args()

    for path, how in [(args.tickets, "Copy development_tickets.json from Capstone_Pack/05_Datasets/ into data/."),
                      (args.per_ticket, "Run: python scripts/evaluate_classifier.py --llm-sample 160")]:
        if not Path(path).exists():
            sys.exit(f"{path} not found. {how}")

    raw = {r["ticket_id"]: r for r in json.loads(Path(args.tickets).read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(open(args.per_ticket, encoding="utf-8")))
    accuracy, _, note = load_calibration()
    if note:
        sys.exit(f"Measured confidence missing: {note}")

    retriever = Retriever()
    print(f"Searching the help articles for {len(rows)} development tickets ...")
    cases = []
    for row in rows:
        r = raw[row["ticket_id"]]
        level = row["agreement"]
        label = Classification(row["llm_intent"], row["trained_urgency"] or "medium", accuracy[level], [], level)
        cases.append({"label": label, "passages": retriever.search(normalise(r).text),
                      "labels": r["labels"], "correct": row["correct"] == "True"})

    lines = [f"# Confidence threshold sweep ({len(cases)} development tickets, {config.EMBEDDING_MODEL})", "",
             "Each ticket's confidence is the measured accuracy of its agreement case:",
             ", ".join(f"{k} {round(100 * v, 1)}%" for k, v in sorted(accuracy.items(), key=lambda kv: -kv[1])) + ".",
             "So the threshold decides which agreement cases may be answered automatically.", "",
             "| Threshold | Cases allowed | Routed to an automatic answer | Right intent among them (%) "
             "| Label says a person should handle it | of which: not answerable from articles | of which: never-automate "
             "| Routing accuracy vs label (%) |",
             "|---|---|---|---|---|---|---|---|"]
    expected_auto = [c["labels"]["expected_route"] == "auto_respond" for c in cases]
    for th in THRESHOLDS:
        auto = [c for c in cases if route(c["label"], c["passages"], threshold=th).route == AUTO]
        wrong_route = [c for c in auto if c["labels"]["expected_route"] != "auto_respond"]
        right = sum(c["correct"] for c in auto)
        ok = sum((route(c["label"], c["passages"], threshold=th).route == AUTO) == e
                 for c, e in zip(cases, expected_auto))
        allowed = ", ".join(k for k, v in sorted(accuracy.items(), key=lambda kv: -kv[1])
                            if v >= th and k in ("agree", "in_top3", "disagree")) or "none"
        lines.append(f"| {th:.2f} | {allowed} | {len(auto)} ({pct(len(auto), len(cases))}%) | {pct(right, len(auto))} "
                     f"| {len(wrong_route)} | {sum(not c['labels']['answerable_from_docs'] for c in wrong_route)} "
                     f"| {sum(bool(c['labels']['must_not_auto_respond']) for c in wrong_route)} "
                     f"| {pct(ok, len(cases))} |")

    th = config.CONFIDENCE_THRESHOLD
    by_intent = defaultdict(lambda: [0, 0, 0])
    for c in cases:
        if route(c["label"], c["passages"], threshold=th).route == AUTO:
            v = by_intent[c["label"].intent]
            v[0] += 1
            v[1] += c["correct"]
            v[2] += c["labels"]["expected_route"] != "auto_respond"
    lines += ["", f"## At the current setting (CONFIDENCE_THRESHOLD={th:.2f}): automatic answers by intent", "",
              "| Intent | Routed automatic | Right intent | Label says a person |", "|---|---|---|---|"]
    lines += [f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in sorted(by_intent.items(), key=lambda kv: -kv[1][0])]
    lines += ["", "Read it like this:",
              "- 'Right intent among them' is the share that would get an answer about the right topic. "
              "Pick the lowest threshold that keeps it high.",
              "- 'Label says a person should handle it' are tickets routed automatically that should not be. "
              "The never-automate column must be 0. Tickets not answerable from the articles are meant to be "
              "caught next, when the answer is written (B-08 replies 'insufficient' and escalates).",
              "- Routing accuracy compares this step alone with labels.expected_route.",
              "", "Put the chosen value in .env as CONFIDENCE_THRESHOLD and record it in the Stage 3 / Stage 5 documents."]

    report = "\n".join(lines) + "\n"
    print("\n" + report)
    out = config.ROOT / "evaluation" / "results" / "threshold_sweep.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
