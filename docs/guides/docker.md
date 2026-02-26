# Docker Infrastructure

qortex ships a Docker Compose stack that includes the qortex server itself, the graph database, and full observability services. The compose file lives at `docker/docker-compose.yml`.

## Services overview

| Service | Image | Purpose |
|---------|-------|---------|
| **qortex** | `ghcr.io/peleke/qortex:latest` | qortex REST API + MCP server. The embedding model (sentence-transformers) is baked into the image. |
| **PostgreSQL** | `pgvector/pgvector:pg17` | Shared database for vectors, interoception, and learning state. pgvector extension pre-installed. |
| **Memgraph** | `memgraph/memgraph-mage:latest` | Graph database (Bolt protocol). Production backend for the knowledge graph. |
| **Memgraph Lab** | `memgraph/lab:latest` | Web UI for Memgraph. Visual graph exploration and Cypher queries. |
| **OTel Collector** | `otel/opentelemetry-collector-contrib:latest` | Receives OTLP telemetry from qortex and routes to backends. |
| **Tempo** | `grafana/tempo:2.7.2` | Distributed trace storage. Queried via Grafana with TraceQL. |
| **Prometheus** | `prom/prometheus:latest` | Metrics storage and PromQL queries. |
| **Grafana** | `grafana/grafana:latest` | Dashboard visualization. Ships with the pre-built `qortex-main` dashboard. |
| **VictoriaLogs** | `victoriametrics/victoria-logs:latest` | Log aggregation (structured JSONL logs). 30-day retention. |

> **Note**: Tempo v2.7.2 is pinned deliberately. Later versions (v2.10+) have a known partition ring issue that causes startup failures.

## Running qortex

qortex runs as a Docker container. The embedding model is baked into the image, so there is no separate download step and no `vec.unavailable` fallback.

```bash
# Start qortex + PostgreSQL + observability
cd docker && docker compose up -d qortex

# qortex waits for PostgreSQL to be healthy before starting.
# Verify:
curl -s http://localhost:8400/v1/health
```

The `qortex` service depends on `postgres` and `otel-collector`, which are started automatically. The REST API is available on port 8400 and the MCP Streamable HTTP endpoint on port 8401.

### With Memgraph (full stack)

```bash
cd docker && docker compose --profile local-graph up -d qortex
```

This starts qortex, PostgreSQL, Memgraph, Memgraph Lab, and the full observability stack.

## Compose profiles

The stack uses Docker Compose profiles to separate concerns:

### Default (qortex + postgres + observability)

```bash
cd docker && docker compose up -d qortex
```

Starts: qortex, PostgreSQL, OTel Collector, Tempo, Prometheus, Grafana, VictoriaLogs.

### `local-graph` profile (includes Memgraph)

```bash
cd docker && docker compose --profile local-graph up -d qortex
```

Starts: everything above **plus** Memgraph and Memgraph Lab.

Use this for standalone local development when you need the graph database on the host.

### When using the sandbox (Lima VM)

If qortex runs inside a Lima sandbox, the VM already runs Memgraph and Lab. Lima auto-forwards the Bolt port (7687) and Lab port (3000) to the host. **Do not start the `local-graph` profile** -- you would get port conflicts. Start only the default services.

### Observability only (no qortex container)

If you want to run qortex as a bare process on the host but still use the observability stack:

```bash
cd docker && docker compose up -d otel-collector tempo prometheus grafana victorialogs
```

Then point the host qortex process at the collector:

```bash
QORTEX_OTEL_ENABLED=true \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
qortex serve
```

## Port map

| Port | Service | Protocol/UI |
|------|---------|-------------|
| 8400 | qortex | REST API |
| 8401 | qortex | MCP Streamable HTTP |
| 5432 | PostgreSQL | PostgreSQL wire protocol |
| 7687 | Memgraph | Bolt protocol (graph queries) |
| 7444 | Memgraph | Monitoring endpoint |
| 3000 | Memgraph Lab | Web UI |
| 4317 | OTel Collector | OTLP gRPC receiver |
| 4318 | OTel Collector | OTLP HTTP receiver |
| 8889 | OTel Collector | Prometheus metrics (collector self-metrics) |
| 9091 | Prometheus | Prometheus UI (mapped from container 9090 to avoid conflicts) |
| 3200 | Tempo | Tempo HTTP API (TraceQL query frontend) |
| 3010 | Grafana | Grafana UI (mapped from container 3000 to avoid Memgraph Lab conflict) |
| 9428 | VictoriaLogs | HTTP API |

## Quick start

```bash
# Start qortex with observability
cd docker && docker compose up -d qortex

# Verify qortex is healthy
curl -s http://localhost:8400/v1/health

# Verify observability services
curl -s http://localhost:9091/api/v1/query?query=up | python3 -m json.tool

# Open dashboards
open http://localhost:3010/d/qortex-main/qortex-observability  # Grafana
open http://localhost:3010/explore                               # Tempo traces (via Grafana Explore)
```

With Memgraph:

```bash
# Start everything including graph database
cd docker && docker compose --profile local-graph up -d qortex

# Verify Memgraph
python3 -c "import socket; s=socket.socket(); s.connect(('localhost',7687)); s.close(); print('ok')"

# Open Memgraph Lab
open http://localhost:3000
```

## Environment variables

Set these in your shell or a `.env` file in the `docker/` directory:

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_STORE` | `postgres` | Storage backend (`postgres` or `sqlite`). |
| `QORTEX_GRAPH` | `memgraph` | Graph backend (`memgraph` or `memory`). |
| `QORTEX_EXTRACTION` | `spacy` | Extraction strategy (`spacy`, `llm`, or `none`). |
| `POSTGRES_USER` | `qortex` | PostgreSQL username. |
| `POSTGRES_PASSWORD` | `qortex` | PostgreSQL password. |
| `POSTGRES_DB` | `qortex` | PostgreSQL database name. |
| `MEMGRAPH_USER` | `memgraph` | Memgraph Bolt auth username. |
| `MEMGRAPH_PASSWORD` | `memgraph` | Memgraph Bolt auth password. |
| `GF_SECURITY_ADMIN_PASSWORD` | `qortex` | Grafana admin password. |

### qortex container environment

These variables are pre-configured inside the `qortex` Docker service and typically do not need overriding:

| Variable | Value | Description |
|----------|-------|-------------|
| `QORTEX_STORE` | `postgres` | Use PostgreSQL for vec, learning, and interoception stores. |
| `DATABASE_URL` | `postgresql://qortex:qortex@postgres:5432/qortex` | PostgreSQL connection string (container-internal hostname). |
| `QORTEX_GRAPH` | `memgraph` | Use Memgraph backend (instead of in-memory). |
| `MEMGRAPH_HOST` | `memgraph` | Memgraph container hostname. |
| `MEMGRAPH_PORT` | `7687` | Memgraph Bolt port. |
| `QORTEX_EXTRACTION` | `spacy` | Concept extraction via spaCy (model baked into image). |
| `QORTEX_OTEL_ENABLED` | `true` | Enable OTel metrics and traces export. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4318` | OTel Collector HTTP endpoint (container-internal). |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` | OTLP export protocol. |
| `QORTEX_PROMETHEUS_ENABLED` | `true` | Enable local `/metrics` endpoint for Prometheus scraping. |
| `QORTEX_PROMETHEUS_PORT` | `9464` | Port for the local Prometheus scrape target. |

## Volumes

The compose file defines named volumes for persistent data:

| Volume | Service | Purpose |
|--------|---------|---------|
| `postgres_data` | PostgreSQL | Database files (`/var/lib/postgresql/data`). |
| `memgraph_data` | Memgraph | Graph data (`/var/lib/memgraph`). |
| `memgraph_log` | Memgraph | Server logs (`/var/log/memgraph`). |
| `prometheus_data` | Prometheus | Metric time-series data. |
| `victorialogs_data` | VictoriaLogs | Aggregated logs. |
| `grafana_data` | Grafana | Dashboard state and user preferences. |
| `tempo_data` | Tempo | Trace data (`/var/tempo`). |

To reset all data:

```bash
cd docker && docker compose --profile local-graph down -v
```

## Grafana provisioning

Grafana is pre-configured with:
- **Datasources**: Prometheus (`http://prometheus:9090`), Tempo (`http://tempo:3200`), VictoriaLogs (`http://victorialogs:9428`).
- **Dashboard**: `qortex-main` auto-loaded from `docker/grafana/dashboards/qortex.json`.
- **Plugins**: `victoriametrics-logs-datasource`, `jdbranham-diagram-panel`, `grafana-tempo-datasource`.
- **Anonymous access**: enabled as Viewer (no login needed for read-only).

Traces are viewable in Grafana via the Tempo datasource. Use **Explore > Tempo** to search by service name (`qortex`) or use TraceQL queries.

## Stopping services

```bash
# Stop everything (preserve data)
cd docker && docker compose --profile local-graph down

# Stop and delete all data
cd docker && docker compose --profile local-graph down -v
```

## Next steps

- [PostgreSQL Setup](postgres-setup.md) -- pgvector and postgres store configuration
- [Observability](observability.md) -- metrics, traces, and dashboard panels reference
- [Using Memgraph](memgraph.md) -- Memgraph backend setup and Cypher queries
- [Online Indexing](online-indexing.md) -- how conversation turns flow into the graph
