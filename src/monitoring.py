"""Live monitoring metrics for Prometheus (backlog item B-12; Setup Guide section 06).

While the system runs, it publishes counters at http://localhost:<port>/metrics. Prometheus
reads them every few seconds and Grafana draws the dashboard (see monitoring/README.md).

Metrics (the first three are named as in the Setup Guide):
  tickets_processed_total{channel, outcome}   tickets finished, by channel and by what happened
                                              (auto_respond, escalate, block)
  response_seconds                            time per ticket (histogram: median and 95th percentile)
  guardrail_blocks_total{guardrail}           answers stopped, by safety check
  guardrail_errors_total{guardrail}           safety checks that could not run
  classification_confidence                   the confidence given to each ticket (histogram);
                                              a shift here is the first sign of drift
  routing_decisions_total{rule}               which routing rule decided (src/route.py)
  llm_calls_total / llm_cache_hits_total      real model calls and answers read from the cache (cost)

The metrics live in their own registry, so tests can create fresh ones. record() is cheap and
is called for every ticket, whether or not the endpoint is switched on.
"""
from __future__ import annotations

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, start_http_server

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60)
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.tickets = Counter("tickets_processed", "Tickets processed", ["channel", "outcome"], registry=r)
        self.latency = Histogram("response_seconds", "End to end response time per ticket",
                                 buckets=LATENCY_BUCKETS, registry=r)
        self.guardrail_blocks = Counter("guardrail_blocks", "Answers stopped by a safety check", ["guardrail"],
                                        registry=r)
        self.guardrail_errors = Counter("guardrail_errors", "Safety checks that could not run", ["guardrail"],
                                        registry=r)
        self.confidence = Histogram("classification_confidence", "Confidence given to each classified ticket",
                                    buckets=CONFIDENCE_BUCKETS, registry=r)
        self.routing = Counter("routing_decisions", "Routing decisions by the rule that decided", ["rule"],
                               registry=r)
        self.llm_calls = Counter("llm_calls", "Real calls made to the language model", registry=r)
        self.llm_cache_hits = Counter("llm_cache_hits", "Model answers read from the cache", registry=r)
        self._seen_calls = self._seen_hits = 0
        self._server = None

    def record(self, ticket: Any, result: Any, latency_s: float) -> None:
        """Count one finished ticket. Never raises: monitoring must not stop a run."""
        try:
            r = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            self.tickets.labels(channel=getattr(ticket, "channel", "unknown") or "unknown",
                                outcome=r.get("action") or "unknown").inc()
            self.latency.observe(max(0.0, float(latency_s)))
            for name, outcome in (r.get("guardrails") or {}).items():
                if outcome == "block":
                    self.guardrail_blocks.labels(guardrail=name).inc()
                elif outcome == "error":
                    self.guardrail_errors.labels(guardrail=name).inc()
            if r.get("confidence") is not None and r.get("intent"):
                self.confidence.observe(float(r["confidence"]))
            if r.get("routing_rule"):
                self.routing.labels(rule=r["routing_rule"]).inc()
            from src.llm import get_client
            client = get_client()
            if client.calls_made > self._seen_calls:
                self.llm_calls.inc(client.calls_made - self._seen_calls)
                self._seen_calls = client.calls_made
            if client.cache_hits > self._seen_hits:
                self.llm_cache_hits.inc(client.cache_hits - self._seen_hits)
                self._seen_hits = client.cache_hits
        except Exception:
            pass

    def serve(self, port: int) -> int:
        """Start the /metrics endpoint in the background. Returns the port it listens on."""
        server, _thread = start_http_server(port, registry=self.registry)
        self._server = server
        return server.server_address[1]

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    def text(self) -> str:
        """The metrics as Prometheus reads them (used by tests and for a quick look)."""
        return generate_latest(self.registry).decode()


_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics
