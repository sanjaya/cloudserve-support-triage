"""Metrics report (FR-11, Build Specification §04).
All numbers are worked out here by code from the run's results. Nothing is added by hand.

Each record given to `compute()` has:
  result    - what the system decided for the ticket (from src.pipeline)
  latency_s - how many seconds the system took for the ticket
  labels    - the correct answers for the ticket, if the file has them (otherwise empty)
"""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any


def _pct(n: int, d: int) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return round(ordered[k], 3)


def per_class(pairs: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
    """Precision and recall for each intent.

    precision: of the tickets we labelled as X, how many really were X
    recall:    of the tickets that really were X, how many we labelled as X
    """
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    for true, pred in pairs:
        support[true] += 1
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    out = {}
    for c in sorted(set(support) | set(fp)):
        p_den, r_den = tp[c] + fp[c], tp[c] + fn[c]
        out[c] = {
            "precision": round(tp[c] / p_den, 3) if p_den else None,
            "recall": round(tp[c] / r_den, 3) if r_den else None,
            "support": support[c],
        }
    return out


def compute(records: list[dict[str, Any]], decisions_logged: int | None = None) -> dict[str, Any]:
    n = len(records)
    results = [r["result"] for r in records]
    actions = Counter(r["action"] for r in results)
    latencies = [r["latency_s"] for r in records]
    labelled = [r for r in records if r.get("labels")]

    # Classification: compare our intent with the correct intent, where we have both
    intent_pairs = [(r["labels"]["intent"], r["result"]["intent"])
                    for r in labelled if r["result"].get("intent") and r["labels"].get("intent")]
    # Document search "hit": we used at least one of the documents a correct answer should cite
    hit_base = [r for r in labelled if r["labels"].get("expected_doc_ids")]
    hits = sum(1 for r in hit_base
               if {s.split("#")[0] for s in r["result"]["sources"]} & set(r["labels"]["expected_doc_ids"]))
    route_base = [r for r in labelled if r["labels"].get("expected_route")]
    route_ok = sum(1 for r in route_base
                   if (r["result"]["action"] == "auto_respond") == (r["labels"]["expected_route"] == "auto_respond"))
    unsafe_auto = sum(1 for r in labelled
                      if r["labels"].get("must_not_auto_respond") and r["result"]["action"] == "auto_respond")

    guardrail_blocks = Counter()
    for res in results:
        for name, outcome in res.get("guardrails", {}).items():
            if outcome == "block":
                guardrail_blocks[name] += 1

    # For the fairness audit (B-14): how often each customer group got an automatic answer
    segments: dict[str, dict[str, Any]] = {}
    for key in ("customer_tier", "customer_region", "language_fluency"):
        groups = defaultdict(list)
        for r in records:
            groups[r.get(key, "unknown")].append(r["result"]["action"] == "auto_respond")
        segments[key] = {g: {"tickets": len(v), "auto_resolved_pct": _pct(sum(v), len(v))} for g, v in sorted(groups.items())}

    return {
        "volume": {
            "tickets_processed": n,
            "answered_automatically": actions["auto_respond"],
            "escalated": actions["escalate"],
            "blocked_by_guardrails": actions["block"],
            "stage_errors": sum(1 for r in results if r.get("error")),
        },
        "business": {
            "first_contact_resolution_pct": _pct(actions["auto_respond"], n),
            "escalation_rate_pct": _pct(actions["escalate"] + actions["block"], n),
            "response_time_mean_s": round(statistics.mean(latencies), 3) if latencies else None,
            "response_time_median_s": round(statistics.median(latencies), 3) if latencies else None,
        },
        "technical": {
            "classification_per_class": per_class(intent_pairs) if intent_pairs else None,
            "retrieval_hit_rate_pct": _pct(hits, len(hit_base)) if any(r["result"]["sources"] for r in hit_base) else None,
            "routing_accuracy_pct": _pct(route_ok, len(route_base)),
            "latency_p50_s": _percentile(latencies, 0.50),
            "latency_p95_s": _percentile(latencies, 0.95),
        },
        "governance": {
            "decisions_logged": decisions_logged,
            "guardrail_activations_by_type": dict(guardrail_blocks),
            "private_data_detections": guardrail_blocks.get("pii", 0),
            "auto_responses_on_never_automate_tickets": unsafe_auto,
        },
        "segments": segments,
    }


def to_markdown(metrics: dict[str, Any], meta: dict[str, Any]) -> str:
    def show(v: Any) -> str:
        return "not available yet" if v is None else str(v)

    lines = ["# Metrics report", ""]
    lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in meta.items()]
    for group in ("volume", "business", "technical", "governance"):
        lines += ["", f"## {group.title()}", "", "| Measure | Value |", "|---|---|"]
        for k, v in metrics[group].items():
            if k == "classification_per_class" and v:
                v = f"{len(v)} classes – see metrics.json"
            lines.append(f"| {k.replace('_', ' ')} | {show(v)} |")
    lines += ["", "Note: response time here is how long the system took per ticket. It is not the same as",
              "the time a customer waits for a first reply. The project report explains the difference."]
    return "\n".join(lines) + "\n"
