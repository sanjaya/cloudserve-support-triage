# Monitoring (B-12)

While the system runs it publishes live metrics (`src/monitoring.py`). Prometheus collects them
and Grafana shows them on the dashboard **CloudServe triage**
(`grafana/dashboards/cloudserve_triage.json`).

## 1. Start Prometheus and Grafana

**With Docker Desktop** (simplest):

```bash
docker compose -f monitoring/docker-compose.yml up -d
```

Prometheus is at http://localhost:9090 and Grafana at http://localhost:3000. The data source and
the dashboard are set up automatically (Dashboards → CloudServe → CloudServe triage). Stop
both with `docker compose -f monitoring/docker-compose.yml down`.

**Without Docker** (Homebrew on a Mac):

```bash
brew install prometheus grafana
prometheus --config.file=monitoring/prometheus-local.yml      # leave running in its own terminal
brew services start grafana                                     # then open http://localhost:3000 (admin / admin)
```

In Grafana: Connections → Data sources → Add → Prometheus, URL `http://localhost:9090`, Save.
Then Dashboards → New → Import → upload `monitoring/grafana/dashboards/cloudserve_triage.json`.

## 2. Run the system with the metrics switched on

```bash
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/ \
    --metrics-port 8001 --pace 2 --keep-serving
```

- `--metrics-port 8001` publishes the metrics at http://localhost:8001/metrics (Prometheus reads them every 5 seconds).
- `--pace 2` waits 2 seconds between tickets, so the 80 tickets take about 3 minutes and the
  graphs show traffic over time instead of one spike.
- `--keep-serving` keeps the endpoint open after the run so the final numbers stay visible;
  press Ctrl+C when you are done.

## 3. What the dashboard shows

| Panel | What it answers |
|---|---|
| Tickets processed, answered automatically, stopped by safety checks | Totals since the run started |
| Tickets per hour, by outcome and by channel | How much work arrives and where it goes |
| Answered automatically vs sent to a person | The share of automatic answers (first contact resolution) |
| Response time (median and 95th percentile) | Against the 3-second target (NFR-01) |
| Safety-check blocks per minute, by check | Which check stops answers, and checks that could not run |
| Confidence given to tickets | Where drift shows up first |
| Routing decisions, by rule | Why tickets go to a person (src/route.py) |
| Model calls per minute | Real calls (cost) against answers from the cache |

These are the views the Setup Guide asks for (section 06). Save a screenshot of the dashboard
for the report (B-12 definition of done).
