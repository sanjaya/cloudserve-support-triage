"""B-06: measure the classifier fairly on the development tickets, and save the measured
confidence for each agreement case.

    python scripts/evaluate_classifier.py                  # trained classifier only, no API calls
    python scripts/evaluate_classifier.py --llm-sample 160 # full evaluation (needs OpenRouter)

Uses development tickets only – never the validation or hidden tickets.

1. The trained classifier is tested on WHOLE GROUPS it has never seen (5-fold, by group).
   The development tickets are about 80 originals, each reworded several times; testing on
   unseen groups shows how it copes with new wording. A normal random split is shown for contrast.
2. With --llm-sample N, the language model classifies N tickets (about 2 from every group).
   Each ticket is put in an agreement case – agree / in_top3 / disagree – by comparing the
   language model's answer with the trained classifier's ranking (from the unseen-groups test,
   so the trained classifier had never seen that wording). The accuracy of each case becomes
   the confidence the system reports for it, saved in models/agreement_calibration.json.
   Language-model answers are cached, so re-running costs nothing.
3. Urgency is measured for both methods; the better one is used.

Writes evaluation/results/classifier_evaluation.md and classifier_per_ticket.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import calibration, macro_precision, per_class  # noqa: E402
from src import config  # noqa: E402
from src.classify import LEVELS, TrainedClassifier, agreement_level, ask_llm, near_duplicate_groups  # noqa: E402
from src.ingest import normalise  # noqa: E402
from src.llm import LLMClient, LLMError  # noqa: E402
from src.retrieve import get_embedder  # noqa: E402


class CachedEmbedder:
    """Embed each text once, then reuse it across the test folds."""

    def __init__(self, inner):
        self.inner, self.name, self._seen = inner, inner.name, {}

    def embed(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self._seen]
        if todo:
            self._seen.update(zip(todo, self.inner.embed(todo)))
        return [self._seen[t] for t in texts]


def out_of_fold(texts, intents, urgencies, groups, embedder, grouped=True):
    """Train on 4/5, predict the other 1/5, five times. Every ticket is predicted once,
    by a model that did not train on it (or, when grouped, on any reworded copy of it)."""
    from sklearn.model_selection import GroupKFold, KFold
    splitter = GroupKFold(5) if grouped else KFold(5, shuffle=True, random_state=0)
    preds = [None] * len(texts)
    for train, test in splitter.split(texts, intents, groups if grouped else None):
        clf = TrainedClassifier.fit([texts[i] for i in train], [intents[i] for i in train],
                                    [urgencies[i] for i in train], embedder=embedder,
                                    groups=[groups[i] for i in train])
        for i in test:
            preds[i] = clf.predict(texts[i])
    return preds


def pct(n, d):
    return round(100 * n / d, 1) if d else None


def wilson(k, n, z=1.96):
    """95% range for an accuracy measured on n tickets (small samples are uncertain)."""
    if not n:
        return (None, None)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * (centre - half), 1), round(100 * (centre + half), 1))


def pick_sample(groups, n):
    """About the same number of tickets from every group, in a fixed order."""
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    order, round_ = [], 0
    while len(order) < min(n, len(groups)):
        for g in sorted(by_group):
            if round_ < len(by_group[g]) and len(order) < n:
                order.append(by_group[g][round_])
        round_ += 1
    return sorted(order)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickets", default=str(config.DATA_DIR / "development_tickets.json"))
    ap.add_argument("--llm-sample", type=int, default=0, help="ask the language model about N tickets (0 = skip)")
    ap.add_argument("--pace", type=float, default=3.0, help="seconds between model calls (stay inside rate limits)")
    ap.add_argument("--embedder", default=None)
    args = ap.parse_args()

    path = Path(args.tickets)
    if not path.exists():
        sys.exit(f"{path} not found. Copy development_tickets.json from Capstone_Pack/05_Datasets/ into data/.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    tickets = [normalise(r) for r in raw]
    texts = [t.text for t in tickets]
    intents = [r["labels"]["intent"] for r in raw]
    urgencies = [r["labels"]["urgency"] for r in raw]
    groups = near_duplicate_groups(texts)
    embedder = CachedEmbedder(get_embedder(args.embedder))

    print(f"{len(texts)} tickets, {len(set(groups))} groups. Testing on unseen groups ...")
    oof = out_of_fold(texts, intents, urgencies, groups, embedder, grouped=True)
    print("Testing with a normal random split ...")
    rnd = out_of_fold(texts, intents, urgencies, groups, embedder, grouped=False)

    def trained_summary(preds):
        pairs = [(t, p[0][0][0]) for t, p in zip(intents, preds)]
        return (pct(sum(a == b for a, b in pairs), len(pairs)), macro_precision(per_class(pairs)),
                pct(sum(u == p[1] for u, p in zip(urgencies, preds)), len(urgencies)))

    g, r = trained_summary(oof), trained_summary(rnd)
    trained_ok = sum(t == p[0][0][0] for t, p in zip(intents, oof))
    lines = [f"# Classifier evaluation ({embedder.name}, {len(texts)} development tickets, {len(set(groups))} groups)", "",
             f"Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "## Trained classifier (the checker)", "",
             "| Measure | Unseen groups (new wording) | Random split (copies seen) |", "|---|---|---|",
             f"| intent accuracy (%) | {g[0]} | {r[0]} |", f"| macro precision (%) | {g[1]} | {r[1]} |",
             f"| urgency accuracy (%) | {g[2]} | {r[2]} |", "",
             "The unseen-groups column is the honest figure for tickets worded in a new way."]

    if not args.llm_sample:
        lines += ["", "Run again with --llm-sample 160 to measure the language model and the agreement cases."]
    else:
        sample = pick_sample(groups, args.llm_sample)
        client = LLMClient(min_interval_s=args.pace)
        print(f"Asking the language model about {len(sample)} tickets (cached answers are free) ...")
        llm, failed = {}, 0
        for n, i in enumerate(sample, 1):
            try:
                llm[i] = ask_llm(tickets[i], client)
            except LLMError as exc:
                failed += 1
                print(f"  {raw[i]['ticket_id']}: {exc}")
            if n % 20 == 0:
                print(f"  {n}/{len(sample)} ({client.calls_made} calls, {client.cache_hits} from cache)")
        asked = [i for i in sample if i in llm]

        rows = []
        for i in asked:
            ranked, t_urg = oof[i]
            level = agreement_level(llm[i]["intent"], ranked)
            rows.append({"ticket_id": raw[i]["ticket_id"], "true_intent": intents[i],
                         "llm_intent": llm[i]["intent"], "trained_intent": ranked[0][0],
                         "trained_top3": "|".join(n for n, _ in ranked[:3]), "agreement": level,
                         "correct": llm[i]["intent"] == intents[i], "llm_stated_confidence": llm[i]["confidence"],
                         "true_urgency": urgencies[i], "llm_urgency": llm[i]["urgency"], "trained_urgency": t_urg})

        levels = {}
        for level in ("agree", "in_top3", "disagree"):
            g_rows = [x for x in rows if x["agreement"] == level]
            k = sum(x["correct"] for x in g_rows)
            levels[level] = {"tickets": len(g_rows), "correct": k,
                             "accuracy": round(k / len(g_rows), 4) if g_rows else 0.0,
                             "range_95_pct": wilson(k, len(g_rows))}
        llm_ok = sum(x["correct"] for x in rows)
        levels["llm_only"] = {"tickets": len(rows), "correct": llm_ok,
                              "accuracy": round(llm_ok / len(rows), 4) if rows else 0.0,
                              "range_95_pct": wilson(llm_ok, len(rows))}
        levels["trained_only"] = {"tickets": len(texts), "correct": trained_ok,
                                  "accuracy": round(trained_ok / len(texts), 4),
                                  "range_95_pct": wilson(trained_ok, len(texts))}
        llm_urg = sum(x["llm_urgency"] == x["true_urgency"] for x in rows)
        tr_urg = sum(x["trained_urgency"] == x["true_urgency"] for x in rows)
        urgency_source = "llm" if llm_urg >= tr_urg else "trained"

        lines += ["", f"## Language model ({client.model}, OpenRouter) on {len(rows)} tickets", "",
                  f"- intent accuracy: {pct(llm_ok, len(rows))}% (failed calls: {failed})",
                  f"- urgency accuracy: language model {pct(llm_urg, len(rows))}%, trained classifier "
                  f"{pct(tr_urg, len(rows))}% -> urgency taken from: **{urgency_source}**", "",
                  "### The model's own stated confidence (why we do not use it)", "",
                  "| Confidence band | Tickets | Said (%) | Right (%) | Gap (points) |", "|---|---|---|---|---|"]
        lines += [f"| {b['band']} | {b['tickets']} | {b['stated_pct']} | {b['observed_pct']} | {b['gap_points']} |"
                  for b in calibration([(x["llm_stated_confidence"], x["correct"]) for x in rows])]
        lines += ["", "## Agreement cases: the confidence the system will report", "",
                  "| Case | Tickets | Right | Accuracy = confidence (%) | 95% range (%) |", "|---|---|---|---|---|"]
        for level in LEVELS:
            v = levels[level]
            lines.append(f"| {level} | {v['tickets']} | {v['correct']} | {round(100 * v['accuracy'], 1)} | "
                         f"{v['range_95_pct'][0]}-{v['range_95_pct'][1]} |")
        lines += ["", "agree / in_top3 / disagree: language model answer vs trained classifier ranking.",
                  "llm_only: language model alone (used if the trained model is missing).",
                  "trained_only: trained classifier alone on unseen groups (used if the language model is down).",
                  "", "These accuracies are measured on the same tickets they describe, so the validation run",
                  "(harness calibration table) is the independent check. Small cases have wide 95% ranges."]

        calib = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "model": client.model, "embedder": embedder.name, "development_tickets": len(texts),
                 "sample": len(rows), "urgency_source": urgency_source, "levels": levels}
        cpath = Path(config.CLASSIFIER_CALIBRATION_PATH)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        shown = cpath.relative_to(config.ROOT) if cpath.is_relative_to(config.ROOT) else cpath
        lines += ["", f"Saved the measured confidence to {shown} (commit this file)."]

        out_csv = config.ROOT / "evaluation" / "results" / "classifier_per_ticket.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    report = "\n".join(lines) + "\n"
    print("\n" + report)
    out = config.ROOT / "evaluation" / "results" / "classifier_evaluation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
