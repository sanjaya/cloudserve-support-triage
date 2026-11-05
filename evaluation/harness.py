"""Run the system on a whole ticket file, with nobody watching (FR-11; checks A9 and A10).

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/

- You give it the path of ANY ticket file. No file name is fixed in the code,
  because the assessor will run it on a hidden file we have never seen.
- Every ticket gets a result. If something fails for one ticket, that ticket is sent
  to a person, the error is written down, and the run continues. No ticket is lost.
- Each run creates a new folder named with the date and time, containing:
    results.jsonl  one line per ticket
    metrics.json   all the numbers the Build Specification asks for
    metrics.md     the same numbers, easy to read
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from evaluation import metrics as M
from src.guardrails import remember_customers
from src.ingest import load_raw, normalise
from src.pipeline import Result, process


def _git_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run(input_path: str, output_dir: str, limit: int | None = None) -> Path:
    started = datetime.now(timezone.utc)
    raw_tickets = load_raw(input_path)
    if limit:
        raw_tickets = raw_tickets[:limit]

    remember_customers(raw_tickets)      # so the private-data check knows every customer name in the file

    out = Path(output_dir) / started.strftime("run_%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)

    records = []
    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for raw in tqdm(raw_tickets, desc="tickets", unit="ticket"):
            ticket = normalise(raw)
            t0 = time.perf_counter()
            try:
                result = process(ticket)
            except Exception as exc:  # FR-12: one bad ticket must never stop the whole run
                result = Result(ticket.ticket_id, "escalate",
                                "One of the steps failed, so a person will handle this ticket.",
                                error=f"{type(exc).__name__}: {exc}")
            latency = time.perf_counter() - t0
            labels = raw.get("labels", {}) if isinstance(raw, dict) else {}
            record = {
                "ticket_id": ticket.ticket_id,
                "channel": ticket.channel,
                "customer_tier": ticket.customer_tier,
                "customer_region": ticket.customer_region,
                "language_fluency": ticket.language_fluency,
                "ingest_issues": ticket.issues,
                "latency_s": round(latency, 4),
                "result": result.to_dict(),
                "labels": labels if isinstance(labels, dict) else {},
            }
            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    finished = datetime.now(timezone.utc)
    meta = {
        "input_file": str(input_path),
        "tickets_in_file": len(raw_tickets),
        "started_utc": started.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds(), 1),
        "system_version": _git_version(),
    }
    report = M.compute(records, decisions_logged=None)  # the decision log is added in B-10
    (out / "metrics.json").write_text(json.dumps({"meta": meta, **report}, indent=2), encoding="utf-8")
    (out / "metrics.md").write_text(M.to_markdown(report, meta), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the system on a ticket file and write a metrics report.")
    parser.add_argument("--input", required=True, help="the ticket file to process")
    parser.add_argument("--output", required=True, help="folder for the results (a new dated folder is made inside it)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tickets (for quick tests while building)")
    args = parser.parse_args(argv)
    out = run(args.input, args.output, args.limit)
    print(f"Done. Results and metrics report in {out}")


if __name__ == "__main__":
    sys.exit(main())
