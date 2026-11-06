"""A small web API for one ticket at a time (backlog item B-23, a Should).

    python -m src.api                    # http://127.0.0.1:8000 ; open /docs to try it in the browser
    python -m src.api --port 8080

The harness (evaluation/harness.py) stays the way to process a whole file unattended. This API
runs the SAME steps on a single ticket, for a live demo or for another system to call:

  POST /tickets            send one ticket (the same JSON as one entry in the ticket files); get back
                           what the system did and why: action, reason, intent, confidence, the answer
                           (if sent), the sources, each safety check and the hand-off note.
                           Every decision is written to the decision log, like in a harness run.
  GET  /tickets/{id}       how that ticket was handled, step by step, read back from the decision log.
  GET  /health             is it up; is the kill switch on; which model and threshold are in use.
  GET  /metrics            the same live metrics as the harness (B-12), for Prometheus.

Like the harness, it never fails on a bad ticket: an empty or broken ticket goes to a person with a
reason (FR-12). It listens on this computer only (127.0.0.1) by default; it has no login, so do not
open it to a network without putting authentication in front of it.
"""
from __future__ import annotations

import argparse
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from src import config, kill_switch
from src.escalate import escalate
from src.guardrails import remember_customers
from src.ingest import normalise
from src.logging_store import history, log_decision
from src.monitoring import get_metrics
from src.pipeline import Result, process


def _version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=config.ROOT, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


VERSION = _version()
# One id in the decision log for everything this server handles, so its records stay apart from
# harness runs: api_<start time>_<random part>.
RUN_ID = f"api_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"

app = FastAPI(title="CloudServe support triage", version=VERSION,
              description="Answer a support ticket from the help articles, or hand it to a person with the reason. "
                          "Backlog item B-23.")

EXAMPLE = {
    "ticket_id": "DEMO-0001",
    "channel": "email",
    "subject": "Webhooks failing signature check",
    "body": "Signature verification fails for every delivery. We compute the signature over the parsed body.",
    "customer_id": "CUST-0001",
    "customer_name": "Demo Customer",
    "customer_tier": "business",
    "customer_region": "europe",
    "language_fluency": "fluent",
}


# One ticket at a time: the model client spaces its calls for the provider's rate limit and the
# answer cache is a single file, so requests that arrive together simply wait their turn.
_ONE_AT_A_TIME = threading.Lock()


def _safe_process(ticket) -> Result:
    """Run the graph; if anything breaks, send the ticket to a person with the reason (same as the harness)."""
    try:
        with _ONE_AT_A_TIME:
            return process(ticket)
    except Exception as exc:  # FR-12
        reason = "One of the steps failed, so a person will handle this ticket."
        return Result(ticket.ticket_id, "escalate", reason, error=f"{type(exc).__name__}: {exc}",
                      handoff=escalate(ticket, None, [], None, f"{reason} ({type(exc).__name__}: {exc})",
                                       use_model=False))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "system_version": VERSION, "kill_switch_on": kill_switch.is_on(),
            "model": "disabled" if config.LLM_DISABLED else config.LLM_MODEL,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD, "decision_log_run_id": RUN_ID}


@app.post("/tickets")
def triage(raw: Any = Body(..., examples=[EXAMPLE])) -> dict[str, Any]:
    """Process one ticket. The body is one ticket as in the ticket files; missing fields are allowed."""
    if isinstance(raw, dict):
        remember_customers([raw])      # so the private-data check knows this customer's name
    ticket = normalise(raw)
    t0 = time.perf_counter()
    result = _safe_process(ticket)
    latency = time.perf_counter() - t0
    get_metrics().record(ticket, result, latency)
    try:
        log_decision(RUN_ID, ticket, result, VERSION)       # FR-09: every decision is logged
        logged, log_error = True, None
    except Exception as exc:  # reported, never hidden
        logged, log_error = False, f"{type(exc).__name__}: {str(exc)[:200]}"
    r = result.to_dict()
    return {
        "ticket_id": r["ticket_id"],
        "action": r["action"],
        "reason": r["reason"],
        "intent": r["intent"],
        "urgency": r["urgency"],
        "confidence": r["confidence"],
        "routing_rule": r["routing_rule"],
        "answer": r["answer"],
        "sources": r["sources"],
        "safety_checks": r["guardrails"],
        "safety_check_notes": r["guardrail_notes"],
        "handoff": r["handoff"],
        "error": r["error"],
        "ingest_issues": ticket.issues,
        "latency_s": round(latency, 3),
        "decision_log": {"run_id": RUN_ID, "logged": logged, "error": log_error},
    }


@app.get("/tickets/{ticket_id}")
def ticket_history(ticket_id: str) -> dict[str, Any]:
    """How a ticket was handled, step by step, from the decision log (its latest run)."""
    rows = history(ticket_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No decision log records for {ticket_id}.")
    steps = [{k: row[k] for k in ("step", "requirement_ids", "model", "prompt_version", "prediction", "confidence",
                                  "sources", "threshold", "action", "reason", "guardrail_results")}
             for row in rows]
    return {"ticket_id": ticket_id, "run_id": rows[0]["run_id"], "system_version": rows[0]["system_version"],
            "input_summary": rows[0]["input_summary"], "steps": steps}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    from prometheus_client import generate_latest
    return generate_latest(get_metrics().registry).decode("utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the single-ticket API (B-23).")
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    import uvicorn

    from src.pipeline import get_classifier, get_retriever
    print("Loading the search store and the classifier (first time takes about a minute)...")
    get_retriever()
    get_classifier()
    print(f"CloudServe triage API on http://{args.host}:{args.port}  (try it at /docs; decision log run id {RUN_ID})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
