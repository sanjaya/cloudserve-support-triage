"""B-11: check a finished run against the gate (acceptance criteria A9, A10 and A8, with spot
checks of A3, A6 and the escalation rules).

    python scripts/check_gate.py --run evaluation/results/run_<...> --input data/validation_tickets.json

It reads only what the run left behind (results.jsonl, metrics.json, metrics.md and the
decision log), so it shows what an assessor would see. It writes gate_check.md into the run
folder and exits with 0 if every check passed, 1 if not.

Checks (Build Specification sections 02 and 04):
- A9  every ticket in the input file has a result, in the same order, none dropped or doubled
- A9  every result is a sent answer or an escalation (escalate / block)
- A9  the run was made on committed code (so it can be repeated exactly)
- A10 the metrics report exists with all four sections (volume, business, technical, governance)
- A8  the decision log has one final record per ticket for this run (reconciles), with a reason
- A3  every classified ticket has an intent and a confidence between 0 and 1
- A6  every sent answer cites only passages search returned for that ticket
- Route: every escalation carries a hand-off note with a summary and the retrieved sources
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.escalate import is_complete  # noqa: E402
from src.ingest import load_raw, normalise  # noqa: E402

ACTIONS = {"auto_respond", "escalate", "block"}


def check(run: Path, input_path: Path) -> tuple[dict[str, tuple[bool, str]], dict]:
    raw = load_raw(input_path)
    expected = [normalise(r).ticket_id for r in raw]
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    md = (run / "metrics.md").read_text(encoding="utf-8") if (run / "metrics.md").exists() else ""
    meta = metrics.get("meta", {})
    results = [r["result"] for r in rows]
    c: dict[str, tuple[bool, str]] = {}

    got = [r["ticket_id"] for r in rows]
    c["A9 every ticket processed, none dropped or doubled"] = (
        got == expected, f"{len(got)} results for {len(expected)} tickets in {input_path.name}")
    bad = [r["ticket_id"] for r in results if r.get("action") not in ACTIONS]
    c["A9 every result is a sent answer or an escalation"] = (not bad, f"{len(bad)} without a valid action")
    c["A9 run made on committed code"] = (
        meta.get("code_matches_commit") is True,
        f"version {meta.get('system_version')}, code matches commit: {meta.get('code_matches_commit')}")
    sections = all(f"## {s}" in md for s in ("Volume", "Business", "Technical", "Governance"))
    c["A10 metrics report written with all four sections"] = (
        sections and all(k in metrics for k in ("volume", "business", "technical", "governance")),
        "metrics.md and metrics.json present" if sections else "metrics report missing or incomplete")
    gov = metrics.get("governance", {})
    c["A8 decision log reconciles with tickets processed"] = (
        gov.get("decision_log_reconciles") is True and gov.get("decisions_logged") == len(rows),
        f"{gov.get('decisions_logged')} logged for {len(rows)} processed; errors: {gov.get('decision_log_errors')}")
    try:
        from src.logging_store import history
        run_id = meta.get("run_id")
        sample = [history(t, run_id) for t in expected[:: max(1, len(expected) // 10)]]
        ok = all(h and h[-1]["step"] == "final" and h[-1]["reason"] for h in sample)
        c["A8 log records read back with a reason (sample of 10)"] = (ok, f"run id {run_id}")
    except Exception as exc:
        c["A8 log records read back with a reason (sample of 10)"] = (False, f"{type(exc).__name__}: {exc}")
    classified = [r for r in results if r.get("intent")]
    bad_conf = [r["ticket_id"] for r in classified
                if not isinstance(r.get("confidence"), (int, float)) or not 0.0 <= r["confidence"] <= 1.0]
    c["A3 every classified ticket has a confidence between 0 and 1"] = (
        not bad_conf, f"{len(classified)} classified, {len(bad_conf)} with a bad confidence")
    sent = [r for r in results if r.get("action") == "auto_respond"]
    bad_cite = [r["ticket_id"] for r in sent
                if not set((r.get("draft") or {}).get("citations") or ["none"]) <= set(r.get("sources") or [])]
    c["A6 sent answers cite only passages search returned"] = (
        not bad_cite, f"{len(sent)} sent, {len(bad_cite)} with a citation search did not return")
    to_person = [r for r in results if r.get("action") in ("escalate", "block")]
    no_note = [r["ticket_id"] for r in to_person if not is_complete(r.get("handoff"))]
    c["Every escalation carries a complete hand-off note"] = (
        not no_note, f"{len(to_person)} escalations, {len(no_note)} without a complete note")
    return c, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="the run folder made by the harness")
    ap.add_argument("--input", required=True, help="the ticket file that run processed")
    args = ap.parse_args()
    run, inp = Path(args.run), Path(args.input)
    c, metrics = check(run, inp)
    meta, vol = metrics.get("meta", {}), metrics.get("volume", {})
    passed = all(ok for ok, _ in c.values())
    lines = [f"# Gate check (B-11): {run.name}", "",
             f"Input: {inp} – {meta.get('tickets_in_file')} tickets; started {meta.get('started_utc')}; "
             f"duration {meta.get('duration_s')} s; version {meta.get('system_version')}.", "",
             f"**Result: {'PASSED' if passed else 'NOT PASSED'}** ({sum(ok for ok, _ in c.values())} of {len(c)} checks)", "",
             "| Check | Passed | Detail |", "|---|---|---|"]
    lines += [f"| {k} | {'yes' if ok else '**NO**'} | {d} |" for k, (ok, d) in c.items()]
    lines += ["", f"Outcome: {vol.get('answered_automatically')} answered automatically, {vol.get('escalated')} sent to a "
              f"person, {vol.get('blocked_by_guardrails')} stopped by safety checks, {vol.get('stage_errors')} stage errors.",
              f"Model usage: {meta.get('model_usage')}."]
    report = "\n".join(lines) + "\n"
    (run / "gate_check.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {run / 'gate_check.md'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
