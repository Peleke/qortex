# qortex-observe

Event-driven observability for [qortex](https://github.com/Peleke/qortex): metrics, traces, logs, and alerts.

<div align="center">

<!-- Architecture: Event-Driven Observability Pipeline -->
<svg viewBox="0 0 620 520" xmlns="http://www.w3.org/2000/svg" aria-label="qortex-observe architecture: events flow through subscribers to metrics, traces, logs, and alerts">
  <style>
    .obs-bg { fill: #0d1117; }
    .obs-box { fill: #161b22; stroke: #30363d; stroke-width: 1; rx: 6; }
    .obs-box-accent { fill: #161b22; stroke: #6366f1; stroke-width: 1.5; rx: 6; filter: url(#obs-glow); }
    .obs-label { font-family: 'JetBrains Mono', monospace; font-size: 8px; fill: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .obs-title { font-family: system-ui, sans-serif; font-size: 13px; fill: #e6edf3; }
    .obs-subtitle { font-family: system-ui, sans-serif; font-size: 10px; fill: #8b949e; }
    .obs-flow { stroke: #6366f1; stroke-width: 1.2; stroke-dasharray: 4 3; fill: none; opacity: 0.5; }
    .obs-flow-anim { animation: obs-dash 2s linear infinite; }
    @keyframes obs-dash { to { stroke-dashoffset: -14; } }
    .obs-arrow { fill: #6366f1; opacity: 0.5; }
  </style>
  <defs>
    <filter id="obs-glow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="620" height="520" class="obs-bg"/>

  <!-- Modules (top) -->
  <rect x="180" y="20" width="260" height="50" class="obs-box"/>
  <text x="195" y="38" class="obs-label">application code</text>
  <text x="195" y="55" class="obs-title">emit(QueryCompleted(...))</text>

  <!-- Event Bus -->
  <rect x="180" y="110" width="260" height="50" class="obs-box-accent"/>
  <text x="195" y="128" class="obs-label">event bus</text>
  <text x="195" y="145" class="obs-title">QortexEventLinker (pyventus)</text>

  <!-- Flow: modules → bus -->
  <line x1="310" y1="70" x2="310" y2="110" class="obs-flow obs-flow-anim"/>
  <polygon points="310,108 306,100 314,100" class="obs-arrow"/>

  <!-- Subscriber row -->
  <!-- Metrics -->
  <rect x="20" y="210" width="130" height="70" class="obs-box"/>
  <text x="35" y="228" class="obs-label">metrics</text>
  <text x="35" y="248" class="obs-title">48 MetricDefs</text>
  <text x="35" y="264" class="obs-subtitle">OTel SDK instruments</text>

  <!-- Traces -->
  <rect x="170" y="210" width="130" height="70" class="obs-box"/>
  <text x="185" y="228" class="obs-label">traces</text>
  <text x="185" y="248" class="obs-title">@traced spans</text>
  <text x="185" y="264" class="obs-subtitle">selective export</text>

  <!-- Logs -->
  <rect x="320" y="210" width="130" height="70" class="obs-box"/>
  <text x="335" y="228" class="obs-label">logs</text>
  <text x="335" y="248" class="obs-title">structlog</text>
  <text x="335" y="264" class="obs-subtitle">formatter × destination</text>

  <!-- Alerts -->
  <rect x="470" y="210" width="130" height="70" class="obs-box"/>
  <text x="485" y="228" class="obs-label">alerts</text>
  <text x="485" y="248" class="obs-title">rule engine</text>
  <text x="485" y="264" class="obs-subtitle">cooldown + sinks</text>

  <!-- Flow: bus → subscribers -->
  <line x1="230" y1="160" x2="85" y2="210" class="obs-flow obs-flow-anim"/>
  <line x1="280" y1="160" x2="235" y2="210" class="obs-flow obs-flow-anim"/>
  <line x1="340" y1="160" x2="385" y2="210" class="obs-flow obs-flow-anim"/>
  <line x1="390" y1="160" x2="535" y2="210" class="obs-flow obs-flow-anim"/>

  <!-- Exporter row -->
  <!-- OTLP -->
  <rect x="20" y="330" width="130" height="55" class="obs-box"/>
  <text x="35" y="348" class="obs-label">otlp push</text>
  <text x="35" y="366" class="obs-subtitle">gRPC / HTTP protobuf</text>

  <!-- Prometheus -->
  <rect x="170" y="330" width="130" height="55" class="obs-box"/>
  <text x="185" y="348" class="obs-label">prometheus pull</text>
  <text x="185" y="366" class="obs-subtitle">:9464/metrics</text>

  <!-- VictoriaLogs -->
  <rect x="320" y="330" width="130" height="55" class="obs-box"/>
  <text x="335" y="348" class="obs-label">victorialogs</text>
  <text x="335" y="366" class="obs-subtitle">batched HTTP POST</text>

  <!-- JSONL -->
  <rect x="470" y="330" width="130" height="55" class="obs-box"/>
  <text x="485" y="348" class="obs-label">jsonl file</text>
  <text x="485" y="366" class="obs-subtitle">Loki-ready</text>

  <!-- Flow: subscribers → exporters -->
  <line x1="85" y1="280" x2="85" y2="330" class="obs-flow obs-flow-anim"/>
  <line x1="85" y1="280" x2="235" y2="330" class="obs-flow obs-flow-anim"/>
  <line x1="235" y1="280" x2="85" y2="330" class="obs-flow obs-flow-anim"/>
  <line x1="385" y1="280" x2="385" y2="330" class="obs-flow obs-flow-anim"/>
  <line x1="385" y1="280" x2="535" y2="330" class="obs-flow obs-flow-anim"/>

  <!-- Grafana (bottom center) -->
  <rect x="180" y="435" width="260" height="55" class="obs-box-accent"/>
  <text x="195" y="453" class="obs-label">visualization</text>
  <text x="195" y="472" class="obs-title">Grafana dashboards</text>

  <!-- Flow: exporters → Grafana -->
  <line x1="85" y1="385" x2="240" y2="435" class="obs-flow obs-flow-anim"/>
  <line x1="235" y1="385" x2="290" y2="435" class="obs-flow obs-flow-anim"/>
  <line x1="385" y1="385" x2="340" y2="435" class="obs-flow obs-flow-anim"/>

  <!-- Carbon badge (side) -->
  <rect x="470" y="435" width="130" height="55" class="obs-box"/>
  <text x="485" y="453" class="obs-label">carbon accounting</text>
  <text x="485" y="472" class="obs-subtitle">GHG / CDP / TCFD</text>
</svg>

</div>

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
