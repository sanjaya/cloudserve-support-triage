"""B-03 / B-04: compare the two ways of splitting articles, and help choose the minimum score.

    python scripts/compare_chunking.py
    python scripts/compare_chunking.py --tickets data/development_tickets.json

Uses the 500 development tickets (never the validation or hidden tickets).
For each splitting method ("section" and "fixed") it measures:
  - hit@1, hit@3, hit@5: how often one of the articles a correct answer should cite
    is among the first 1, 3 or 5 articles found (answerable tickets only)
  - MRR: on average, how high up the first correct article appears (1.0 = always first)
  - hit@5 for fluent and non-fluent customers separately (an early fairness check)
Then, for the better method, a table of minimum scores: how many answerable tickets
would still find their article, and how many unanswerable tickets would correctly get nothing.

The results are printed and saved to evaluation/results/chunking_comparison.md.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.ingest import normalise  # noqa: E402
from src.retrieve import Retriever, get_embedder  # noqa: E402

K = 10


def ranked_docs(passages) -> list[tuple[str, float]]:
    """Article ids in the order found, each once, with its best passage score."""
    seen, out = set(), []
    for p in passages:
        if p.doc_id not in seen:
            seen.add(p.doc_id)
            out.append((p.doc_id, p.score))
    return out


def evaluate(retriever: Retriever, tickets: list[dict]) -> dict:
    rows = []
    for raw in tickets:
        t = normalise(raw)
        docs = ranked_docs(retriever.search(t.text, top_k=K, min_score=0.0))
        labels = raw.get("labels", {})
        expected = set(labels.get("expected_doc_ids") or [])
        rank = next((i + 1 for i, (d, _) in enumerate(docs) if d in expected), None)
        rows.append({
            "answerable": bool(labels.get("answerable_from_docs")) and bool(expected),
            "fluency": t.language_fluency,
            "rank": rank,
            "top_score": docs[0][1] if docs else 0.0,
            "hit_score": next((s for d, s in docs if d in expected), None),
        })
    ans = [r for r in rows if r["answerable"]]

    def hit(k, group=None):
        g = [r for r in ans if group is None or r["fluency"] == group]
        return round(100 * sum(1 for r in g if r["rank"] and r["rank"] <= k) / len(g), 1) if g else None

    return {
        "passages": retriever.passage_count,
        "answerable_tickets": len(ans),
        "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5),
        "mrr": round(statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in ans), 3),
        "hit@5 fluent": hit(5, "fluent"), "hit@5 non_fluent": hit(5, "non_fluent"),
        "_rows": rows,
    }


def threshold_table(rows: list[dict]) -> list[tuple]:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    table = []
    for t in [x / 100 for x in range(20, 75, 5)]:
        kept = sum(1 for r in ans if r["rank"] and r["rank"] <= 5 and r["hit_score"] >= t)
        empty_ok = sum(1 for r in unans if r["top_score"] < t)
        table.append((t, round(100 * kept / len(ans), 1), round(100 * empty_ok / len(unans), 1) if unans else None))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickets", default=str(config.DATA_DIR / "development_tickets.json"))
    ap.add_argument("--embedder", default=None, help="default: EMBEDDING_MODEL from .env")
    args = ap.parse_args()

    path = Path(args.tickets)
    if not path.exists():
        sys.exit(f"{path} not found. Copy development_tickets.json from Capstone_Pack/05_Datasets/ into data/.")
    tickets = json.loads(path.read_text(encoding="utf-8"))
    embedder = get_embedder(args.embedder)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for strategy in ("section", "fixed"):
            print(f"Building and testing '{strategy}' ...")
            r = Retriever(strategy=strategy, embedder=embedder, store_path=tmp, min_score=0.0)
            results[strategy] = evaluate(r, tickets)

    best = max(results, key=lambda s: (results[s]["hit@5"], results[s]["mrr"]))
    cols = ["passages", "answerable_tickets", "hit@1", "hit@3", "hit@5", "mrr", "hit@5 fluent", "hit@5 non_fluent"]
    lines = [f"# Chunking comparison ({embedder.name}, {len(tickets)} development tickets)", "",
             "| Measure | section | fixed |", "|---|---|---|"]
    lines += [f"| {c} | {results['section'][c]} | {results['fixed'][c]} |" for c in cols]
    lines += ["", f"Better method on hit@5 (then MRR): **{best}**", "",
              f"## Minimum score for '{best}'", "",
              "| Minimum score | Answerable tickets that still find a correct article (%) | Unanswerable tickets that correctly get nothing (%) |",
              "|---|---|---|"]
    lines += [f"| {t:.2f} | {a} | {u} |" for t, a, u in threshold_table(results[best]["_rows"])]
    lines += ["", "Pick the minimum score where the first column is still high and the second column has risen.",
              "Put it in .env as RETRIEVAL_MIN_SCORE and record the choice in the Stage 3 / Stage 5 documents."]
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    out = Path(config.ROOT / "evaluation" / "results" / "chunking_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
