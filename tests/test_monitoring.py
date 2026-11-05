"""Tests for live monitoring (B-12): the metrics Prometheus reads, and the /metrics endpoint."""
import json
import urllib.request

from evaluation import harness
from src import config
from src.ingest import normalise
from src.monitoring import Metrics
from src.pipeline import Result


def _ticket(channel="chat"):
    return normalise({"ticket_id": "M1", "channel": channel, "body": "hi"})


def _value(text, line_start):
    return float(next(l for l in text.splitlines() if l.startswith(line_start)).split()[-1])


def test_tickets_latency_guardrails_confidence_and_rules_are_counted():
    m = Metrics()
    m.record(_ticket("chat"), Result("M1", "auto_respond", "r", intent="rate_limit", confidence=0.97,
                                     routing_rule="all_conditions_met", guardrails={"grounding": "pass"}), 0.4)
    m.record(_ticket("email"), Result("M2", "block", "r", intent="rate_limit", confidence=0.97,
                                      routing_rule="all_conditions_met",
                                      guardrails={"grounding": "block", "injection": "error"}), 2.5)
    m.record(_ticket("email"), Result("M3", "escalate", "r"), 0.01)
    t = m.text()
    assert _value(t, 'tickets_processed_total{channel="chat",outcome="auto_respond"}') == 1
    assert _value(t, 'tickets_processed_total{channel="email",outcome="block"}') == 1
    assert _value(t, 'tickets_processed_total{channel="email",outcome="escalate"}') == 1
    assert _value(t, "response_seconds_count") == 3
    assert _value(t, 'guardrail_blocks_total{guardrail="grounding"}') == 1
    assert _value(t, 'guardrail_errors_total{guardrail="injection"}') == 1
    assert _value(t, "classification_confidence_count") == 2          # only classified tickets
    assert _value(t, 'routing_decisions_total{rule="all_conditions_met"}') == 2


def test_recording_never_raises():
    Metrics().record(None, {"no": "fields"}, "not a number")


def test_harness_publishes_metrics_on_a_port(tmp_path, monkeypatch):
    m = Metrics()
    monkeypatch.setattr(harness, "get_metrics", lambda: m)
    src = tmp_path / "t.json"
    src.write_text(json.dumps(json.loads((config.DATA_DIR / "validation_tickets.json").read_text())[:3]))
    harness.run(str(src), str(tmp_path / "out"), metrics_port=0)
    port = m._server.server_address[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # never via a proxy
    body = opener.open(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    m.stop()
    assert "tickets_processed_total" in body and _value(body, "response_seconds_count") == 3


def test_dashboard_and_prometheus_files_fit_together():
    import yaml
    root = config.ROOT / "monitoring"
    dash = json.loads((root / "grafana" / "dashboards" / "cloudserve_triage.json").read_text())
    known = ("tickets_processed_total", "response_seconds_bucket", "guardrail_blocks_total",
             "guardrail_errors_total", "classification_confidence_bucket", "routing_decisions_total",
             "llm_calls_total", "llm_cache_hits_total")
    assert len(dash["panels"]) >= 8
    for p in dash["panels"]:
        for t in p["targets"]:
            assert any(name in t["expr"] for name in known), t["expr"]
    for name in ("prometheus.yml", "prometheus-local.yml"):
        prom = yaml.safe_load((root / name).read_text())
        assert prom["scrape_configs"][0]["static_configs"][0]["targets"][0].endswith(":8001")
    compose = yaml.safe_load((root / "docker-compose.yml").read_text())
    assert set(compose["services"]) == {"prometheus", "grafana"}
