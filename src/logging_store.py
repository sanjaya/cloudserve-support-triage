"""The decision log (FR-09, NFR-05, backlog item B-10).

Every automated decision is written to a database table, so any ticket's handling can be
explained later from the log alone. The database is set by DATABASE_URL in .env: SQLite by
default (storage/decisions.db, never committed); PostgreSQL works too.

For each ticket there is one record per step it went through:

  step       requirement   what is recorded
  classify   FR-02         intent, urgency, confidence, alternatives, method, model, prompt PR-02
  search     FR-03         passages found, minimum score
  route      FR-04, FR-05  automatic or a person, the rule that decided, the threshold
  draft      FR-06         answer status, citations, which check decided, prompt PR-03
  safety     FR-07         every safety-check result and what it found, prompts PR-05 / PR-06
  final      FR-08, FR-09  what happened in the end (auto_respond / escalate / block) and why

Every record has the run, the ticket, an input summary, the action and a plain-language
reason. A ticket that failed part-way still gets its "final" record. The harness checks that the
number of tickets with a "final" record equals the number of tickets processed (test T-09a).

To see how one ticket was handled:   python -m src.logging_store --ticket VAL-0001
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from src import config

STEPS = ("classify", "search", "route", "draft", "safety", "final")
REQUIREMENTS = {"classify": "FR-02", "search": "FR-03", "route": "FR-04, FR-05", "draft": "FR-06",
                "safety": "FR-07", "final": "FR-08, FR-09"}

_engine = None
_table = None


def _db():
    """Connect once and create the table if it is not there yet."""
    global _engine, _table
    if _engine is None:
        from sqlalchemy import (JSON, Column, DateTime, Float, Integer, MetaData, String, Table, Text,
                                create_engine)
        url = config.DATABASE_URL
        if url.startswith("sqlite:///"):
            path = url[len("sqlite:///"):]
            if path and path != ":memory:":
                from pathlib import Path
                p = Path(path)
                if not p.is_absolute():
                    p = config.ROOT / p
                p.parent.mkdir(parents=True, exist_ok=True)
                url = f"sqlite:///{p}"
        _engine = create_engine(url, future=True)
        meta = MetaData()
        _table = Table(
            "decisions", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String(64), index=True),
            Column("ticket_id", String(64), index=True),
            Column("step", String(16)),
            Column("recorded_at", DateTime(timezone=True)),
            Column("requirement_ids", String(64)),
            Column("input_summary", Text),
            Column("model", String(200)),
            Column("prompt_version", String(64)),
            Column("prediction", String(64)),
            Column("confidence", Float),
            Column("alternatives", JSON),
            Column("sources", JSON),
            Column("threshold", Float),
            Column("action", String(32)),
            Column("reason", Text),
            Column("guardrail_results", JSON),
            Column("details", JSON),
            Column("system_version", String(64)),
        )
        meta.create_all(_engine)
    return _engine, _table


def reset() -> None:
    """Forget the connection (used by tests after changing DATABASE_URL)."""
    global _engine, _table
    if _engine is not None:
        _engine.dispose()
    _engine = _table = None


def _summary(ticket: Any) -> str:
    """Channel, customer and the first part of the ticket text."""
    text = " ".join(f"{getattr(ticket, 'subject', '')} {getattr(ticket, 'body', '')}".split())
    return (f"[{getattr(ticket, 'channel', '')}] customer {getattr(ticket, 'customer_id', '') or 'unknown'}: "
            f"{text[:200]}{'…' if len(text) > 200 else ''}")


def records_for(run_id: str, ticket: Any, result: Any, system_version: str = "") -> list[dict[str, Any]]:
    """Turn one ticket's Result into log records, one per step it went through."""
    r = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    now = datetime.now(timezone.utc)
    base = {"run_id": run_id, "ticket_id": r["ticket_id"], "recorded_at": now,
            "input_summary": _summary(ticket), "system_version": system_version}

    def rec(step: str, **kw: Any) -> dict[str, Any]:
        return {**base, "step": step, "requirement_ids": REQUIREMENTS[step], **kw}

    out = []
    if r.get("intent"):
        out.append(rec("classify", model=f"{config.LLM_MODEL} + trained classifier ({config.EMBEDDING_MODEL})",
                       prompt_version="PR-02 v1.0" if r.get("classifier_method") not in ("trained_only", "fallback") else None,
                       prediction=r["intent"], confidence=r.get("confidence"), alternatives=r.get("alternatives"),
                       action=None, reason=r.get("classifier_note") or f"method: {r.get('classifier_method')}",
                       details={"urgency": r.get("urgency"), "method": r.get("classifier_method")}))
        out.append(rec("search", model=config.EMBEDDING_MODEL, sources=r.get("sources"),
                       threshold=config.RETRIEVAL_MIN_SCORE,
                       reason=f"{len(r.get('sources') or [])} passage(s) at or above the minimum score"))
    if r.get("routing_decision"):
        out.append(rec("route", prediction=r.get("intent"), confidence=r.get("confidence"),
                       threshold=r.get("threshold"), action=r["routing_decision"],
                       reason=r.get("routing_rule"), details={"rule": r.get("routing_rule")}))
    d = r.get("draft")
    if d:
        out.append(rec("draft", model=config.LLM_MODEL, prompt_version=d.get("prompt_version"),
                       prediction=d.get("status"), sources=d.get("citations"),
                       action="send" if d.get("send") else "do not send", reason=d.get("reason"),
                       details={"check": d.get("check"), "unanswered_points": d.get("unanswered_points")}))
    if r.get("guardrails"):
        out.append(rec("safety", model=config.LLM_MODEL, prompt_version="PR-05 v{} + PR-06 v1.0".format(
                           config.GROUNDING_PROMPT_VERSION),
                       action="pass" if all(v == "pass" for v in r["guardrails"].values()) else "block",
                       guardrail_results=r["guardrails"], details={"notes": r.get("guardrail_notes")},
                       reason="; ".join(f"{k}: {v}" for k, v in (r.get("guardrail_notes") or {}).items()) or
                       "all checks passed"))
    out.append(rec("final", prediction=r.get("intent"), confidence=r.get("confidence"),
                   alternatives=r.get("alternatives"), sources=r.get("sources"), threshold=r.get("threshold"),
                   action=r["action"], reason=r["reason"], guardrail_results=r.get("guardrails") or None,
                   details={"error": r.get("error"), "answer": r.get("answer"),
                            "urgency": r.get("urgency")}))
    return out


def log_decision(run_id: str, ticket: Any, result: Any, system_version: str = "") -> int:
    """Write one ticket's records. Returns how many were written."""
    engine, table = _db()
    columns = [c.name for c in table.columns if c.name != "id"]
    rows = [{c: r.get(c) for c in columns} for r in records_for(run_id, ticket, result, system_version)]
    with engine.begin() as conn:
        conn.execute(table.insert(), rows)
    return len(rows)


def tickets_logged(run_id: str) -> int:
    """How many tickets in this run have a final record (for the reconciliation check)."""
    from sqlalchemy import func, select
    engine, table = _db()
    with engine.connect() as conn:
        return conn.execute(select(func.count(func.distinct(table.c.ticket_id)))
                            .where(table.c.run_id == run_id, table.c.step == "final")).scalar_one()


def history(ticket_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
    """Every record for one ticket (the latest run unless run_id is given), in step order."""
    from sqlalchemy import select
    engine, table = _db()
    with engine.connect() as conn:
        if run_id is None:
            last = conn.execute(select(table.c.run_id).where(table.c.ticket_id == ticket_id)
                                .order_by(table.c.id.desc()).limit(1)).scalar()
            if last is None:
                return []
            run_id = last
        rows = conn.execute(select(table).where(table.c.ticket_id == ticket_id, table.c.run_id == run_id)
                            .order_by(table.c.id)).mappings().all()
    return [dict(r) for r in rows]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Show how one ticket was handled, from the decision log.")
    ap.add_argument("--ticket", required=True)
    ap.add_argument("--run", default=None, help="run id (default: the latest run with this ticket)")
    args = ap.parse_args(argv)
    rows = history(args.ticket, args.run)
    if not rows:
        print(f"No records for {args.ticket}.")
        return
    print(f"{args.ticket} in {rows[0]['run_id']} (system version {rows[0]['system_version']})")
    print(f"Input: {rows[0]['input_summary']}\n")
    for r in rows:
        bits = [f"{k}={r[k]}" for k in ("prediction", "confidence", "threshold", "action", "prompt_version")
                if r[k] not in (None, "")]
        print(f"[{r['step']}] {r['requirement_ids']}  " + "  ".join(bits))
        if r["sources"]:
            print(f"    sources: {', '.join(r['sources'])}")
        if r["guardrail_results"]:
            print(f"    checks: {json.dumps(r['guardrail_results'])}")
        if r["reason"]:
            print(f"    reason: {r['reason']}")


if __name__ == "__main__":
    main()
