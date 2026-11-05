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
import uuid
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from evaluation import metrics as M
from src.guardrails import remember_customers
from src.logging_store import log_decision, tickets_logged
from src.monitoring import get_metrics
from src.ingest import load_raw, normalise
from src.pipeline import Result, process


def _git_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run(input_path: str, output_dir: str, limit: int | None = None, metrics_port: int | None = None,
        pace_s: float = 0.0, keep_serving: bool = False) -> Path:
    """Process every ticket in the file and write the results and the metrics report.

    metrics_port: also publish live metrics for Prometheus at http://localhost:<port>/metrics (B-12).
    pace_s:       wait this long between tickets, so a dashboard shows traffic over time.
    keep_serving: after the run, keep the metrics endpoint open until Ctrl+C, so Prometheus can
                  read the final numbers."""
    started = datetime.now(timezone.utc)
    raw_tickets = load_raw(input_path)
    if limit:
        raw_tickets = raw_tickets[:limit]

    remember_customers(raw_tickets)      # so the private-data check knows every customer name in the file

    out = Path(output_dir) / started.strftime("run_%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)

    metrics = get_metrics()
    if metrics_port is not None:
        port = metrics.serve(metrics_port)
        print(f"Live metrics at http://localhost:{port}/metrics")

    records, log_errors = [], []
    version = _git_version()
    # The run's id in the decision log: the folder name plus a short random part, so two runs
    # started in the same second never share an id.
    run_id = f"{out.name}_{uuid.uuid4().hex[:6]}"
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
            metrics.record(ticket, result, latency)          # B-12: live monitoring
            if pace_s:
                time.sleep(pace_s)
            try:                     # FR-09: every decision goes to the decision log
                log_decision(run_id, ticket, result, version)
            except Exception as exc:  # a logging failure is reported, never hidden
                log_errors.append(f"{ticket.ticket_id}: {type(exc).__name__}: {str(exc)[:200]}")

    finished = datetime.now(timezone.utc)
    meta = {
        "input_file": str(input_path),
        "tickets_in_file": len(raw_tickets),
        "started_utc": started.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds(), 1),
        "system_version": version,
        "run_id": run_id,
    }
    try:
        logged = tickets_logged(run_id)
    except Exception as exc:
        logged, log_errors = None, log_errors + [f"count failed: {type(exc).__name__}: {exc}"]
    report = M.compute(records, decisions_logged=logged)
    report["governance"]["decision_log_reconciles"] = (logged == len(records)) if logged is not None else False
    report["governance"]["decision_log_errors"] = log_errors[:20] or "none"
    (out / "metrics.json").write_text(json.dumps({"meta": meta, **report}, indent=2), encoding="utf-8")
    (out / "metrics.md").write_text(M.to_markdown(report, meta), encoding="utf-8")
    if metrics_port is not None and keep_serving:
        print(f"Run finished; results in {out}. Metrics endpoint stays open for Prometheus - press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the system on a ticket file and write a metrics report.")
    parser.add_argument("--input", required=True, help="the ticket file to process")
    parser.add_argument("--output", required=True, help="folder for the results (a new dated folder is made inside it)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tickets (for quick tests while building)")
    parser.add_argument("--metrics-port", type=int, default=None,
                        help="publish live metrics for Prometheus on this port, e.g. 8001 (B-12)")
    parser.add_argument("--pace", type=float, default=0.0,
                        help="seconds to wait between tickets, so the dashboard shows traffic over time")
    parser.add_argument("--keep-serving", action="store_true",
                        help="keep the metrics endpoint open after the run, until Ctrl+C")
    args = parser.parse_args(argv)
    out = run(args.input, args.output, args.limit, args.metrics_port, args.pace, args.keep_serving)
    print(f"Done. Results and metrics report in {out}")


if __name__ == "__main__":
    sys.exit(main())
