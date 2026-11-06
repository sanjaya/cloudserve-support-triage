"""B-11: make a practice file shaped like the hidden assessment set, to rehearse "re-run against a
file you have never seen" (Build Specification A9).

    python scripts/make_rehearsal_set.py --out ~/workspace/rehearsal/rehearsal_120.jsonl

It takes 120 development tickets the harness has never been run on as a file (a fixed random
sample), removes the answers (labels) and the history, and writes them as JSON Lines – a
different format and a different path from the validation file. Running the harness on it
checks that nothing depends on the validation file's name, format or labels.

Because the trained classifier learned from the development tickets, the outcome figures from this
file are not a fair estimate of quality; the point is that the run completes unattended.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()
    tickets = json.loads((config.DATA_DIR / "development_tickets.json").read_text(encoding="utf-8"))
    sample = random.Random(args.seed).sample(tickets, args.n)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for t in sample:
            fh.write(json.dumps({k: v for k, v in t.items() if k not in ("labels", "history")}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sample)} unlabelled tickets to {out}")


if __name__ == "__main__":
    main()
