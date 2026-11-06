"""B-17 / FR-12 / A11: prove the system keeps going when things break, in a real run.

    python scripts/induce_failures.py                     # disconnect after 30 tickets
    python scripts/induce_failures.py --disconnect-after 10

It runs the normal harness over data/validation_tickets.json plus five broken tickets (not a
record, missing fields, wrong types, unknown channel, empty text), with the real configuration.
After --disconnect-after tickets it cuts the language-model provider off: the address is
switched to one that refuses connections, and the answer cache is replaced by an empty one, so
every later question really has to reach the provider and fails.

What should happen (test T-12a, T-12c; acceptance A11):
- the run finishes: every ticket has a result and a decision-log record (log reconciles);
- tickets after the cut go to a person with a reason and a hand-off note written by code;
- the circuit breaker opens, so the remaining tickets do not each wait through every retry;
- the broken tickets go to a person with a reason.

No money is spent: before the cut, answers come from the cache; after it, nothing gets through.
Writes the run folder as usual plus failure_test.md inside it.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import harness  # noqa: E402
from src import config  # noqa: E402
from src.llm import _Cache, get_client  # noqa: E402

BROKEN = [None, "just a string", {"channel": "chat", "subject": "no id or body"},
          {"ticket_id": "BROKEN-4", "channel": 7, "subject": ["a", "list"], "body": 12345},
          {"ticket_id": "BROKEN-5", "channel": "pigeon", "body": "   "}]
UNREACHABLE = "http://127.0.0.1:9"     # nothing listens here, so every connection is refused


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default=str(config.DATA_DIR / "validation_tickets.json"))
    ap.add_argument("--output", default=str(config.ROOT / "evaluation" / "results"))
    ap.add_argument("--disconnect-after", type=int, default=30)
    args = ap.parse_args()

    tickets = json.loads(Path(args.input).read_text(encoding="utf-8")) + BROKEN
    work = Path(tempfile.mkdtemp(prefix="induce_failures_"))
    file = work / "tickets_with_broken_records.json"
    file.write_text(json.dumps(tickets), encoding="utf-8")

    client = get_client()
    real_process, seen = harness.process, {"n": 0}

    def process(ticket):
        seen["n"] += 1
        if seen["n"] == args.disconnect_after + 1:
            print(f"--- cutting the provider off before ticket {seen['n']} ({ticket.ticket_id}) ---")
            client._base_url = UNREACHABLE
            client._cache = _Cache(work / "empty_cache.sqlite")
            client._timeout = 5.0
        return real_process(ticket)

    harness.process = process
    out = harness.run(str(file), args.output)
    harness.process = real_process

    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    before, after = rows[:args.disconnect_after], rows[args.disconnect_after:-len(BROKEN)]
    broken = rows[-len(BROKEN):]
    to_person = lambda rs: sum(r["result"]["action"] in ("escalate", "block") for r in rs)  # noqa: E731
    by_code = sum((r["result"].get("handoff") or {}).get("written_by") == "code (fallback)" for r in after)
    usage = metrics["meta"]["model_usage"]
    checks = {
        "every ticket has a result": len(rows) == len(tickets),
        "decision log reconciles": metrics["governance"]["decision_log_reconciles"] is True,
        "no ticket failed outright (stage errors)": metrics["volume"]["stage_errors"] == 0,
        "every ticket after the cut went to a person": to_person(after) == len(after),
        "every ticket after the cut has a hand-off note": all((r["result"].get("handoff") or {}).get("summary")
                                                              for r in after),
        "circuit breaker opened": usage["circuit_breaker_opened"] >= 1,
        "every broken ticket went to a person with a reason": all(
            r["result"]["action"] == "escalate" and r["result"]["reason"] for r in broken),
    }
    reasons = Counter(r["result"].get("routing_rule") or "error" for r in after)
    lines = [f"# Failure test (FR-12, A11): provider cut off after {args.disconnect_after} tickets", "",
             f"Run: {out.name}; {len(tickets)} tickets ({len(tickets) - len(BROKEN)} validation + {len(BROKEN)} broken); "
             f"duration {metrics['meta']['duration_s']} s.", "",
             "| Check | Result |", "|---|---|"]
    lines += [f"| {k} | {'yes' if v else '**NO**'} |" for k, v in checks.items()]
    lines += ["", "| | Before the cut | After the cut | Broken tickets |", "|---|---|---|---|",
              f"| tickets | {len(before)} | {len(after)} | {len(broken)} |",
              f"| answered automatically | {sum(r['result']['action'] == 'auto_respond' for r in before)} | "
              f"{sum(r['result']['action'] == 'auto_respond' for r in after)} | 0 |",
              f"| went to a person | {to_person(before)} | {to_person(after)} | {to_person(broken)} |",
              f"| hand-off note written by code (model unavailable) | – | {by_code} | – |", "",
              f"Routing rule for tickets after the cut: {dict(reasons)}", "",
              f"Model usage in the run: {usage}", "",
              "Example reason after the cut: " + (after[0]["result"]["reason"] if after else "–")]
    report = "\n".join(lines) + "\n"
    (out / "failure_test.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved to {out / 'failure_test.md'}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
