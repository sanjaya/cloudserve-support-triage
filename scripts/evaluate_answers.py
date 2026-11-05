"""B-08: compare versions of the answer prompt (PR-03) on development tickets.

    python scripts/evaluate_answers.py                    # compares PR-03 v1.0 and v1.1
    python scripts/evaluate_answers.py --versions 1.1     # one version only

Uses development tickets only – never the validation or hidden tickets.

Which tickets: the 160 development tickets in evaluation/results/classifier_per_ticket.csv
(from `python scripts/evaluate_classifier.py --llm-sample 160`). Each is classified as in that
evaluation (the unseen-groups setting), searched, and routed with the real rules
(src/route.py). The ones routed to an automatic answer are drafted with each prompt version.

What a good version does, using the labels:
- for tickets the articles CAN answer (labels.answerable_from_docs = true): says "answered",
  so the answer is sent;
- for tickets the articles CANNOT answer: says "partial" or "insufficient", so a person
  handles them.
If data/ground_truth_responses.json is present (copy it from Capstone_Pack/05_Datasets/), sent
answers are also checked for the phrases in must_not_claim.

Each answer costs one model call; answers are cached, so re-running is free.
Writes evaluation/results/answer_comparison.md and answer_comparison.csv (every draft, for review).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.classify import Classification, load_calibration  # noqa: E402
from src.generate import PROMPTS, generate  # noqa: E402
from src.ingest import normalise  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.retrieve import Retriever  # noqa: E402
from src.route import AUTO, route  # noqa: E402


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickets", default=str(config.DATA_DIR / "development_tickets.json"))
    ap.add_argument("--per-ticket", default=str(config.ROOT / "evaluation" / "results" / "classifier_per_ticket.csv"))
    ap.add_argument("--ground-truth", default=str(config.DATA_DIR / "ground_truth_responses.json"))
    ap.add_argument("--versions", nargs="+", default=list(PROMPTS))
    ap.add_argument("--pace", type=float, default=config.LLM_MIN_INTERVAL_S, help="seconds between model calls")
    args = ap.parse_args()

    for path, how in [(args.tickets, "Copy development_tickets.json from Capstone_Pack/05_Datasets/ into data/."),
                      (args.per_ticket, "Run: python scripts/evaluate_classifier.py --llm-sample 160")]:
        if not Path(path).exists():
            sys.exit(f"{path} not found. {how}")
    raw = {r["ticket_id"]: r for r in json.loads(Path(args.tickets).read_text(encoding="utf-8"))}
    truth = {}
    if Path(args.ground_truth).exists():
        truth = {g["ticket_id"]: g for g in json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))}
    accuracy, _, note = load_calibration()
    if note:
        sys.exit(f"Measured confidence missing: {note}")

    retriever = Retriever()
    cases = []
    for row in csv.DictReader(open(args.per_ticket, encoding="utf-8")):
        r = raw[row["ticket_id"]]
        ticket = normalise(r)
        label = Classification(row["llm_intent"], row["trained_urgency"] or "medium", accuracy[row["agreement"]],
                               [], row["agreement"])
        passages = retriever.search(ticket.text)
        if route(label, passages).route == AUTO:
            cases.append((ticket, label, passages, r["labels"]))
    answerable = [c for c in cases if c[3]["answerable_from_docs"]]
    print(f"{len(cases)} development tickets routed to an automatic answer "
          f"({len(answerable)} answerable from the articles, {len(cases) - len(answerable)} not).")

    client = LLMClient(min_interval_s=args.pace)
    rows, summary = [], {}
    for version in args.versions:
        print(f"Drafting with PR-03 v{version} ...")
        drafts = []
        for n, (ticket, label, passages, labels) in enumerate(cases, 1):
            d = generate(ticket, label, passages, client=client, version=version)
            drafts.append(d)
            gt = truth.get(ticket.ticket_id, {})
            breached = [p for p in gt.get("must_not_claim", []) if d.send and p.lower() in d.answer.lower()]
            rows.append({"version": version, "ticket_id": ticket.ticket_id, "intent": label.intent,
                         "answerable_from_docs": labels["answerable_from_docs"],
                         "expected_route": labels["expected_route"], "status": d.status, "sent": d.send,
                         "check": d.check, "cites_expected_article": bool(
                             {c.split("#")[0] for c in d.citations} & set(labels.get("expected_doc_ids") or [])),
                         "must_not_claim_breached": "; ".join(breached),
                         "unanswered_points": " | ".join(d.unanswered_points), "answer": d.answer})
            if n % 20 == 0:
                print(f"  {n}/{len(cases)} ({client.calls_made} calls, {client.cache_hits} from cache)")
        mine = [x for x in rows if x["version"] == version]
        ans = [x for x in mine if x["answerable_from_docs"]]
        unans = [x for x in mine if not x["answerable_from_docs"]]
        sent = [x for x in mine if x["sent"]]
        summary[version] = {
            "answerable tickets sent (want high)": f"{sum(x['sent'] for x in ans)} of {len(ans)} ({pct(sum(x['sent'] for x in ans), len(ans))}%)",
            "unanswerable tickets sent (want low)": f"{sum(x['sent'] for x in unans)} of {len(unans)} ({pct(sum(x['sent'] for x in unans), len(unans))}%)",
            "statuses": dict(Counter(x["status"] for x in mine)),
            "checks": dict(Counter(x["check"] for x in mine)),
            "sent answers citing an expected article": f"{pct(sum(x['cites_expected_article'] for x in sent), len(sent))}%",
            "sent answers breaking must_not_claim": (sum(bool(x["must_not_claim_breached"]) for x in sent)
                                                     if truth else "no ground-truth file"),
        }

    lines = [f"# Answer prompt comparison ({len(cases)} development tickets routed to an automatic answer)", "",
             f"{len(answerable)} are answerable from the articles (labels), {len(cases) - len(answerable)} are not.", "",
             "| Measure | " + " | ".join(f"PR-03 v{v}" for v in args.versions) + " |",
             "|---|" + "---|" * len(args.versions)]
    for measure in next(iter(summary.values())):
        lines.append(f"| {measure} | " + " | ".join(str(summary[v][measure]) for v in args.versions) + " |")
    lines += ["", "Pick the version that sends the most answerable tickets without sending more unanswerable ones.",
              "Set it in .env as ANSWER_PROMPT_VERSION and record the choice in Stage 3 (PR-03 change history)",
              "and the Stage 5 revision log. Every draft is in answer_comparison.csv for reading."]
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    out = config.ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer_comparison.md").write_text(report, encoding="utf-8")
    with open(out / "answer_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {out / 'answer_comparison.md'} and answer_comparison.csv")


if __name__ == "__main__":
    main()
