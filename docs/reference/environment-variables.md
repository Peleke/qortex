# Environment Variables

Centralized reference for all qortex environment variables.

## Backend Selection

| Variable | Default | Values | Description |
|----------|---------|--------|-------------|
| `QORTEX_GRAPH` | `memory` | `memory`, `memgraph` | Graph database backend |
| `QORTEX_VEC` | `sqlite` | `memory`, `sqlite`, `pgvector` | Vector index backend |
| `QORTEX_STORE` | `sqlite` | `sqlite`, `postgres` | Interoception + learning persistence |
| `QORTEX_STATE_DIR` | `~/.qortex` | path | Override for state directory |

## PostgreSQL / pgvector

All three postgres-backed stores (vec, interoception, learning) share the same DSN.

| Variable | Default | Description |
|----------|---------|-------------|
| `PGVECTOR_DSN` | *(constructed)* | Full connection string. Overrides component vars. |
| `PGVECTOR_HOST` | `localhost` | PostgreSQL host |
| `PGVECTOR_PORT` | `5432` | PostgreSQL port |
| `PGVECTOR_USER` | `qortex` | PostgreSQL user |
| `PGVECTOR_PASSWORD` | `qortex` | PostgreSQL password |
| `PGVECTOR_DB` | `qortex` | PostgreSQL database name |

If `PGVECTOR_DSN` is not set: `postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}`

## Memgraph

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMGRAPH_HOST` | `localhost` | Memgraph hostname |
| `MEMGRAPH_PORT` | `7687` | Memgraph Bolt port |
| `MEMGRAPH_USER` | *(none)* | Bolt auth username |
| `MEMGRAPH_PASSWORD` | *(none)* | Bolt auth password |

## REST API

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_API_KEYS` | *(none)* | Comma-separated API keys for Bearer auth |
| `QORTEX_HMAC_SECRET` | *(none)* | Shared secret for HMAC-SHA256 request signing |
| `QORTEX_HMAC_MAX_AGE` | `300` | Max age (seconds) for HMAC timestamp replay protection |
| `QORTEX_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |

## OpenTelemetry

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_OTEL_ENABLED` | `false` | Enable OTel traces and metrics export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | `grpc` or `http/protobuf` |
| `OTEL_SERVICE_NAME` | `qortex` | Service name in traces |
| `QORTEX_OTEL_TRACE_SAMPLE_RATE` | `0.1` | Trace sampling rate (0.0–1.0) |
| `QORTEX_OTEL_TRACE_LATENCY_THRESHOLD_MS` | `100.0` | Always export spans slower than this |

## Prometheus

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_PROMETHEUS_ENABLED` | `false` | Enable `/metrics` endpoint for Prometheus scraping |
| `QORTEX_PROMETHEUS_PORT` | `9464` | Port for the Prometheus scrape target |

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_LOG_FORMATTER` | `structlog` | `structlog` or `stdlib` |
| `QORTEX_LOG_DESTINATION` | `stderr` | `stderr`, `victorialogs`, or `jsonl` |
| `QORTEX_LOG_LEVEL` | `INFO` | Python log level |
| `QORTEX_LOG_FORMAT` | `json` | `json` or `console` |
| `QORTEX_LOG_PATH` | *(none)* | File path for JSONL log output |

## VictoriaLogs

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_VICTORIALOGS_ENDPOINT` | `http://localhost:9428/insert/jsonline` | VictoriaLogs ingest endpoint |
| `QORTEX_VICTORIALOGS_BATCH_SIZE` | `100` | Batch size for log shipping |
| `QORTEX_VICTORIALOGS_FLUSH_INTERVAL` | `5.0` | Flush interval in seconds |

## LLM / Extraction

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(none)* | Anthropic API key for LLM extraction + enrichment |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL for local LLM |

## Miscellaneous

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_CONFIG` | `~/.claude/qortex-consumers.yaml` | Consumer config file path |
| `QORTEX_COMPOSE_FILE` | `docker/docker-compose.yml` | Docker compose file for `qortex infra` commands |
