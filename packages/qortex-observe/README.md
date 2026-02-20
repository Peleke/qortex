# qortex-observe

Event-driven observability for [qortex](https://github.com/Peleke/qortex): metrics, traces, logs, and alerts.

## Install

```bash
pip install qortex-observe
```

With OpenTelemetry exporters:

```bash
pip install "qortex-observe[otel]"
```

## Quick Start

```python
from qortex.observe import configure, emit
from qortex.observe.events import QueryCompleted

# Zero-config: structured logging to stderr
configure()

# Emit typed events — subscribers handle metrics, traces, logs
emit(QueryCompleted(
    query_id="q-1",
    latency_ms=142.5,
    seed_count=12,
    result_count=8,
    mode="hybrid",
))
```

## What It Does

**qortex-observe** decouples event emission from observation. Application code emits typed events; pluggable subscribers route them to metrics, traces, logs, and alerts — without modules knowing about any of those concerns.

### Metrics (48 instruments)

Single source of truth via the `METRICS` schema tuple. Counters, histograms, and gauges covering:

| Domain | Metrics | Examples |
|--------|---------|---------|
| Query lifecycle | 4 | `qortex_queries`, `qortex_query_duration_seconds` |
| PPR convergence | 2 | `qortex_ppr_started`, `qortex_ppr_iterations` |
| Teleportation factors | 4 | `qortex_factor_updates`, `qortex_factor_entropy` |
| Edge promotion | 5 | `qortex_edges_promoted`, `qortex_kg_coverage` |
| Vector search | 8 | `qortex_vec_search_duration_seconds`, `qortex_vec_index_size` |
| Online indexing | 4 | `qortex_messages_ingested`, `qortex_message_ingest_duration_seconds` |
| Learning (bandit) | 7 | `qortex_learning_selections`, `qortex_learning_posterior_mean` |
| Enrichment | 3 | `qortex_enrichment`, `qortex_enrichment_duration_seconds` |
| Credit propagation | 4 | `qortex_credit_propagations`, `qortex_credit_alpha_delta` |
| Carbon | 4 | `qortex_carbon_co2_grams`, `qortex_carbon_tokens` |

### Traces

The `@traced` decorator creates OpenTelemetry spans with overhead timing (wall time minus external I/O). `SelectiveSpanProcessor` exports only error spans, slow spans, or sampled spans — keeping telemetry volume manageable.

```python
from qortex.observe import traced

@traced("retrieval.query")
def retrieve(query: str) -> list[dict]:
    results = vec_search(query)   # external call tracked separately
    return rerank(results)        # compute overhead measured
```

### MCP Trace Propagation

Distributed tracing across the Python server ↔ TypeScript client boundary via W3C `traceparent` in MCP `_meta`:

```python
from qortex.observe.mcp import mcp_trace_middleware

result = mcp_trace_middleware("retrieve", params, handler)
# Creates span "mcp.tool.retrieve" linked to client's trace context
```

### Logging (Swappable Formatter x Destination)

Structured logging with a strategy pattern: pick a **formatter** (structlog or stdlib) and a **destination** (stderr, VictoriaLogs, JSONL file). All combinations work.

### Carbon Accounting

Per-inference CO2 and water tracking with regulatory export formats:

```python
from qortex.observe.carbon import calculate_carbon, calculate_equivalents

calc = calculate_carbon(input_tokens=1000, output_tokens=500, provider="anthropic", model="claude-sonnet")
equiv = calculate_equivalents(calc.total_co2_grams)
# GHG Protocol, CDP, TCFD, ISO 14064-1 exports available
```

## Configuration

All settings are environment-variable driven with safe defaults:

| Env Var | Default | Purpose |
|---------|---------|---------|
| `QORTEX_LOG_FORMATTER` | `structlog` | `structlog` or `stdlib` |
| `QORTEX_LOG_DESTINATION` | `stderr` | `stderr`, `victorialogs`, `jsonl` |
| `QORTEX_LOG_LEVEL` | `INFO` | Python logging level |
| `QORTEX_LOG_FORMAT` | `json` | `json` or `console` |
| `QORTEX_OTEL_ENABLED` | `false` | Enable OTLP metric/trace push |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` | `grpc` or `http/protobuf` |
| `QORTEX_PROMETHEUS_ENABLED` | `false` | Enable Prometheus `/metrics` |
| `QORTEX_PROMETHEUS_PORT` | `9464` | Prometheus scrape port |
| `QORTEX_OTEL_TRACE_SAMPLE_RATE` | `0.1` | Non-error/slow span sample rate |
| `QORTEX_OTEL_TRACE_LATENCY_THRESHOLD_MS` | `100` | Always export slower spans |
| `QORTEX_ALERTS_ENABLED` | `false` | Enable alert rule evaluation |

## 40+ Typed Events

All events are frozen dataclasses grouped by domain:

- **Query**: `QueryStarted`, `QueryCompleted`, `QueryFailed`
- **PPR**: `PPRStarted`, `PPRConverged`, `PPRDiverged`
- **Factors**: `FactorUpdated`, `FactorsPersisted`, `FactorsLoaded`, `FactorDriftSnapshot`
- **Edges**: `OnlineEdgeRecorded`, `EdgePromoted`, `BufferFlushed`
- **Retrieval**: `VecSearchCompleted`, `OnlineEdgesGenerated`, `FeedbackReceived`
- **Online Indexing**: `MessageIngested`, `ToolResultIngested`
- **Learning**: `LearningSelectionMade`, `LearningObservationRecorded`, `LearningPosteriorUpdated`
- **Carbon**: `CarbonTracked`

## Requirements

- Python 3.11+
- [pyventus](https://pypi.org/project/pyventus/) (event bus)
- [structlog](https://pypi.org/project/structlog/) (structured logging)
- OpenTelemetry SDK (optional, for metrics/traces export)

## License

MIT
