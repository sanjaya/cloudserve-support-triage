"""B-06: train the classifier on the 500 development tickets and save it.

    python scripts/train_classifier.py

Writes models/classifier.joblib. That file IS committed, so a clean checkout (and the
assessor's run) can classify tickets without the development data.
Retrain whenever the embedding model or the development data changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402
from src.classify import TrainedClassifier, near_duplicate_groups  # noqa: E402
from src.ingest import normalise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tickets", default=str(config.DATA_DIR / "development_tickets.json"))
    ap.add_argument("--out", default=config.CLASSIFIER_MODEL_PATH)
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

    print(f"Training on {len(texts)} tickets ({len(set(groups))} groups of reworded copies) ...")
    clf = TrainedClassifier.fit(texts, intents, urgencies, groups=groups)
    clf.save(args.out)
    print(f"Saved to {args.out}")
    print(json.dumps(clf.info, indent=2))


if __name__ == "__main__":
    main()
