"""B-21: kill switch drill (Governance Framework section 5, "How is it tested?").

    python scripts/kill_switch_drill.py --input data/validation_tickets.json --limit 20

What it does:
  1. turns the kill switch on (the file switch, the same one an operator uses);
  2. runs the normal harness over the tickets;
  3. turns the kill switch off again, even if the run fails;
  4. checks that NO ticket was answered automatically, that every ticket went to a person with
     the kill-switch reason and a complete hand-off note, and that the decision log reconciles;
  5. writes kill_switch_drill.md into the run folder.
Re-running over tickets already in the answer cache costs nothing and takes seconds.
Exits with 0 if the drill passed, 1 if not.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import harness  # noqa: E402
from src import kill_switch  # noqa: E402
from src.escalate import is_complete  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="data/validation_tickets.json")
    ap.add_argument("--output", default="evaluation/results/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--by", default="kill switch drill")
    args = ap.parse_args()

    was_on = kill_switch.is_on()
    started = time.time()
    print(kill_switch.set_switch(True, args.by, "drill"))
    try:
        run = harness.run(args.input, args.output, limit=args.limit)
    finally:
        if not was_on:
            print(kill_switch.set_switch(False, args.by, "drill finished"))
    took = time.time() - started

    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    meta = json.loads((run / "metrics.json").read_text(encoding="utf-8")).get("meta", {})
    sent = [r["ticket_id"] for r in rows if r["result"]["action"] == "auto_respond"]
    other_reason = [r["ticket_id"] for r in rows if r["result"].get("routing_rule") not in ("kill_switch", "no_text")]
    no_note = [r["ticket_id"] for r in rows if not is_complete(r["result"].get("handoff"))]
    checks = {
        "No ticket answered automatically": (not sent, f"{len(sent)} sent"),
        "Every ticket sent to a person because of the kill switch": (not other_reason, f"{len(other_reason)} with another rule"),
        "Every ticket has a complete hand-off note": (not no_note, f"{len(no_note)} without"),
        "Kill switch is off again after the drill": (kill_switch.is_on() == was_on, "off" if not kill_switch.is_on() else "ON"),
    }
    passed = all(ok for ok, _ in checks.values())
    lines = [f"# Kill switch drill: {run.name}", "",
             f"{len(rows)} tickets from {args.input}; switch turned on, run, switched off in {took:.1f} s. "
             f"Model usage: {meta.get('model_usage')}.", "",
             "| Check | Passed | Detail |", "|---|---|---|"]
    lines += [f"| {name} | {'yes' if ok else '**NO**'} | {detail} |" for name, (ok, detail) in checks.items()]
    lines += ["", f"Outcome: {'PASSED' if passed else 'FAILED'}. Switch changes are in storage/kill_switch_log.txt."]
    report = "\n".join(lines) + "\n"
    (run / "kill_switch_drill.md").write_text(report, encoding="utf-8")
    print(report)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
