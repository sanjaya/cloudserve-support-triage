"""B-14 / NFR-06: fairness audit of a finished run (Governance Framework section 3).

    python scripts/fairness_audit.py --run evaluation/results/run_<...> --input data/validation_tickets.json
    # optional larger sample for classification, from the development tickets (no model calls):
    python scripts/fairness_audit.py --run ... --input ... --dev-classifier evaluation/results/classifier_per_ticket.csv

It reads a finished run (no model calls) and compares segments of customers:
  customer tier, region and English fluency (NFR-06), short against long tickets (split at the
  median number of words; Governance Framework section 3), and channel.

For each segment it reports:
  tickets           how many tickets are in the segment (small segments are uncertain)
  resolution rate   share answered automatically
  routed automatic  share decide sent to an automatic answer (before the safety checks)
  quality: intent   share classified with the right intent (needs labels)
  quality: article  share where search found an article the labels expect (needs labels)
For resolution and both quality measures it gives the 95% range, the gap to the best segment in
points, and a verdict:
  within 5 points    the gap is under the NFR-06 limit
  real gap           over 5 points and the 95% ranges do not overlap
  can't tell         over 5 points, but the ranges overlap: too few tickets to call it

Writes fairness.md and fairness.json into the run folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest import load_raw, normalise  # noqa: E402

DIMENSIONS = [("customer_tier", "Customer tier"), ("customer_region", "Region"),
              ("language_fluency", "English fluency"), ("length", "Ticket length"), ("channel", "Channel")]
LIMIT = 5.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1))


def rate(k: int, n: int) -> float | None:
    return round(100 * k / n, 1) if n else None


def verdict(value, best, rng, best_rng) -> str:
    if value is None or best is None:
        return "no data"
    gap = best - value
    if gap <= LIMIT:
        return "within 5 points"
    if rng and best_rng and rng[1] < best_rng[0]:
        return "real gap"
    return "can't tell (ranges overlap)"


def segment_rows(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or "unknown")].append(r)
    return dict(sorted(groups.items()))


def measures(group: list[dict]) -> dict:
    n = len(group)
    auto = sum(r["result"]["action"] == "auto_respond" for r in group)
    routed = sum((r["result"].get("routing_decision") or r["result"]["action"]) == "auto_respond" for r in group)
    lab = [r for r in group if r.get("labels", {}).get("intent")]
    right = sum(r["result"].get("intent") == r["labels"]["intent"] for r in lab)
    docs = [r for r in group if r.get("labels", {}).get("expected_doc_ids")]
    found = sum(bool({s.split("#")[0] for s in r["result"].get("sources") or []} & set(r["labels"]["expected_doc_ids"]))
                for r in docs)
    return {"tickets": n,
            "resolution": {"value": rate(auto, n), "k": auto, "n": n, "range": wilson(auto, n)},
            "routed_automatic": {"value": rate(routed, n), "k": routed, "n": n},
            "quality_intent": {"value": rate(right, len(lab)), "k": right, "n": len(lab), "range": wilson(right, len(lab))},
            "quality_article": {"value": rate(found, len(docs)), "k": found, "n": len(docs),
                                "range": wilson(found, len(docs))}}


def audit(rows: list[dict]) -> dict:
    words = [len(r["_text"].split()) for r in rows]
    median = statistics.median(words) if words else 0
    for r, w in zip(rows, words):
        r["length"] = f"short (≤{int(median)} words)" if w <= median else f"long (>{int(median)} words)"
    out = {"median_words": median, "dimensions": {}}
    for key, title in DIMENSIONS:
        segs = {name: measures(g) for name, g in segment_rows(rows, key).items()}
        for m in ("resolution", "quality_intent", "quality_article"):
            values = [(s[m]["value"], s[m].get("range"), name) for name, s in segs.items() if s[m]["value"] is not None]
            if not values:
                continue
            best_value, best_range, best_name = max(values, key=lambda v: v[0])
            for name, s in segs.items():
                v = s[m]["value"]
                s[m]["gap_points"] = round(best_value - v, 1) if v is not None else None
                s[m]["verdict"] = verdict(v, best_value, s[m].get("range"), best_range)
                s[m]["best_segment"] = best_name
        out["dimensions"][key] = {"title": title, "segments": segs}
    return out


def dev_classifier(csv_path: Path, tickets_path: Path) -> dict:
    """Intent accuracy by segment on the development tickets classified in the unseen-groups test."""
    raw = {t["ticket_id"]: t for t in json.loads(tickets_path.read_text(encoding="utf-8"))}
    rows = []
    for row in csv.DictReader(open(csv_path, encoding="utf-8")):
        t = raw.get(row["ticket_id"])
        if t:
            rows.append({**{k: t.get(k) for k in ("customer_tier", "customer_region", "language_fluency", "channel")},
                         "correct": row["correct"] == "True"})
    out = {}
    for key in ("customer_tier", "customer_region", "language_fluency", "channel"):
        segs = {}
        for name, g in segment_rows(rows, key).items():
            k = sum(r["correct"] for r in g)
            segs[name] = {"tickets": len(g), "intent_accuracy": rate(k, len(g)), "range": wilson(k, len(g))}
        best = max(s["intent_accuracy"] for s in segs.values())
        for s in segs.values():
            s["gap_points"] = round(best - s["intent_accuracy"], 1)
        out[key] = segs
    return {"tickets": len(rows), "by": out}


def fmt(m: dict) -> str:
    if m.get("value") is None:
        return "–"
    rng = f" ({m['range'][0]}–{m['range'][1]})" if m.get("range") else ""
    return f"{m['value']}%{rng}"


def to_markdown(result: dict, run: str, n: int, dev: dict | None) -> str:
    lines = [f"# Fairness audit (B-14, NFR-06): {run}", "",
             f"{n} tickets. Short/long split at the median of {result['median_words']:g} words. "
             "Percentages with their 95% range in brackets. Gap = points behind the best segment.", "",
             "Verdicts: **within 5 points** meets NFR-06; **real gap** = over 5 points and the ranges do not "
             "overlap; **can't tell** = over 5 points but the ranges overlap (too few tickets to call).", ""]
    for key, d in result["dimensions"].items():
        lines += [f"## {d['title']}", "",
                  "| Segment | Tickets | Resolution rate | Gap | Routed automatic | Quality: right intent | Gap | "
                  "Quality: right article found | Gap | Verdicts (resolution / intent / article) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, s in d["segments"].items():
            v = " / ".join(s[m].get("verdict", "–") for m in ("resolution", "quality_intent", "quality_article"))
            lines.append(f"| {name} | {s['tickets']} | {fmt(s['resolution'])} | {s['resolution'].get('gap_points', '–')} | "
                         f"{s['routed_automatic']['value']}% | {fmt(s['quality_intent'])} | "
                         f"{s['quality_intent'].get('gap_points', '–')} | {fmt(s['quality_article'])} | "
                         f"{s['quality_article'].get('gap_points', '–')} | {v} |")
        lines.append("")
    if dev:
        lines += [f"## Larger sample: intent accuracy on {dev['tickets']} development tickets "
                  "(unseen-groups classifier test, no model calls)", "",
                  "| Dimension | Segment | Tickets | Right intent | Gap |", "|---|---|---|---|---|"]
        for key, segs in dev["by"].items():
            for name, s in segs.items():
                lines.append(f"| {key.replace('_', ' ')} | {name} | {s['tickets']} | {s['intent_accuracy']}% "
                             f"({s['range'][0]}–{s['range'][1]}) | {s['gap_points']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--dev-classifier", default=None, help="classifier_per_ticket.csv for the larger sample")
    ap.add_argument("--dev-tickets", default=str(Path(__file__).resolve().parent.parent / "data" / "development_tickets.json"))
    args = ap.parse_args()
    run = Path(args.run)
    raw = load_raw(args.input)
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    texts = {normalise(r).ticket_id: normalise(r).text for r in raw}
    for r in rows:
        r["_text"] = texts.get(r["ticket_id"], "")
    result = audit(rows)
    dev = dev_classifier(Path(args.dev_classifier), Path(args.dev_tickets)) if args.dev_classifier else None
    (run / "fairness.json").write_text(json.dumps({"audit": result, "development": dev}, indent=2), encoding="utf-8")
    report = to_markdown(result, run.name, len(rows), dev)
    (run / "fairness.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {run / 'fairness.md'}")


if __name__ == "__main__":
    main()
