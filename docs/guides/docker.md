# Docker Infrastructure

qortex ships a Docker Compose stack for the graph database and observability services. The compose file lives at `docker/docker-compose.yml`.

## Services overview

| Service | Image | Purpose |
|---------|-------|---------|
| **Memgraph** | `memgraph/memgraph-mage:latest` | Graph database (Bolt protocol). Production backend for the knowledge graph. |
| **Memgraph Lab** | `memgraph/lab:latest` | Web UI for Memgraph. Visual graph exploration and Cypher queries. |
| **OTel Collector** | `otel/opentelemetry-collector-contrib:latest` | Receives OTLP telemetry from qortex and routes to backends. |
| **Jaeger** | `jaegertracing/all-in-one:latest` | Distributed trace viewer. |
| **Prometheus** | `prom/prometheus:latest` | Metrics storage and PromQL queries. |
| **Grafana** | `grafana/grafana:latest` | Dashboard visualization. Ships with the pre-built `qortex-main` dashboard. |
| **VictoriaLogs** | `victoriametrics/victoria-logs:latest` | Log aggregation (structured JSONL logs). 30-day retention. |

## Compose profiles

The stack uses Docker Compose profiles to separate concerns:

### Default profile (observability only)

```bash
cd docker && docker compose up -d
```

Starts: OTel Collector, Jaeger, Prometheus, Grafana, VictoriaLogs.

Use this when Memgraph is already running elsewhere (e.g. inside a Lima sandbox VM or on a remote host).

### `local-graph` profile (includes Memgraph)

```bash
cd docker && docker compose --profile local-graph up -d
```

Starts: everything above **plus** Memgraph and Memgraph Lab.

Use this for standalone local development when you need the graph database on the host.

### When using the sandbox (Lima VM)

If qortex runs inside a Lima sandbox, the VM already runs Memgraph and Lab. Lima auto-forwards the Bolt port (7687) and Lab port (3000) to the host. **Do not start the `local-graph` profile** -- you would get port conflicts. Start only the default profile for observability.

## Port map

| Port | Service | Protocol/UI |
|------|---------|-------------|
| 7687 | Memgraph | Bolt protocol (graph queries) |
| 7444 | Memgraph | Monitoring endpoint |
| 3000 | Memgraph Lab | Web UI |
| 4317 | OTel Collector | OTLP gRPC receiver |
| 4318 | OTel Collector | OTLP HTTP receiver |
| 8889 | OTel Collector | Prometheus metrics (collector self-metrics) |
| 9091 | Prometheus | Prometheus UI (mapped from container 9090 to avoid conflicts) |
| 16686 | Jaeger | Jaeger UI |
| 3010 | Grafana | Grafana UI (mapped from container 3000 to avoid Memgraph Lab conflict) |
| 9428 | VictoriaLogs | HTTP API |

## Quick start

```bash
# Start observability stack (no Memgraph)
cd docker && docker compose up -d

# Verify services are healthy
curl -s http://localhost:9091/api/v1/query?query=up | python3 -m json.tool

# Open dashboards
open http://localhost:3010/d/qortex-main/qortex-observability  # Grafana
open http://localhost:16686                                      # Jaeger
```

With Memgraph:

```bash
# Start everything including graph database
cd docker && docker compose --profile local-graph up -d

# Verify Memgraph
python3 -c "import socket; s=socket.socket(); s.connect(('localhost',7687)); s.close(); print('ok')"

# Open Memgraph Lab
open http://localhost:3000
```

## Environment variables

Set these in your shell or a `.env` file in the `docker/` directory:

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMGRAPH_USER` | `memgraph` | Memgraph Bolt auth username. |
| `MEMGRAPH_PASSWORD` | `memgraph` | Memgraph Bolt auth password. |
| `GF_SECURITY_ADMIN_PASSWORD` | `qortex` | Grafana admin password. |

### qortex process environment

These variables configure the qortex Python process to connect to the Docker services:

| Variable | Value | Description |
|----------|-------|-------------|
| `QORTEX_GRAPH` | `memgraph` | Use Memgraph backend (instead of in-memory). |
| `MEMGRAPH_HOST` | `localhost` | Memgraph host (default works for local Docker). |
| `MEMGRAPH_PORT` | `7687` | Memgraph Bolt port. |
| `MEMGRAPH_USER` | `memgraph` | Must match the Docker env. |
| `MEMGRAPH_PASSWORD` | `memgraph` | Must match the Docker env. |
| `QORTEX_OTEL_ENABLED` | `true` | Enable OTel metrics and traces export. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4318` | OTel Collector HTTP endpoint. |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | OTLP export protocol. |
| `QORTEX_PROMETHEUS_ENABLED` | `true` | Enable local `/metrics` endpoint for Prometheus scraping. |
| `QORTEX_PROMETHEUS_PORT` | `9464` | Port for the local Prometheus scrape target. |

### Full example

```bash
QORTEX_GRAPH=memgraph \
MEMGRAPH_USER=memgraph MEMGRAPH_PASSWORD=memgraph \
QORTEX_OTEL_ENABLED=true \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
QORTEX_PROMETHEUS_ENABLED=true \
qortex mcp-serve
```

## Volumes

The compose file defines named volumes for persistent data:

| Volume | Service | Purpose |
|--------|---------|---------|
| `memgraph_data` | Memgraph | Graph data (`/var/lib/memgraph`). |
| `memgraph_log` | Memgraph | Server logs (`/var/log/memgraph`). |
| `prometheus_data` | Prometheus | Metric time-series data. |
| `victorialogs_data` | VictoriaLogs | Aggregated logs. |
| `grafana_data` | Grafana | Dashboard state and user preferences. |

To reset all data:

```bash
cd docker && docker compose --profile local-graph down -v
```

## Grafana provisioning

Grafana is pre-configured with:
- **Datasources**: Prometheus (`http://prometheus:9090`), VictoriaLogs (`http://victorialogs:9428`).
- **Dashboard**: `qortex-main` auto-loaded from `docker/grafana/dashboards/qortex.json`.
- **Plugins**: `victoriametrics-logs-datasource`, `jdbranham-diagram-panel`.
- **Anonymous access**: enabled as Viewer (no login needed for read-only).

## Stopping services

```bash
# Stop everything (preserve data)
cd docker && docker compose --profile local-graph down

# Stop and delete all data
cd docker && docker compose --profile local-graph down -v
```

## Next steps

- [Observability](observability.md) -- metrics, traces, and dashboard panels reference
- [Using Memgraph](memgraph.md) -- Memgraph backend setup and Cypher queries
- [Online Indexing](online-indexing.md) -- how conversation turns flow into the graph
