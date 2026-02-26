# Green Metrics Emission and Grafana Dashboard Plan

## 1. Current State Assessment

### 1.1 vindler/openclaw Green Module (TypeScript)

**Location:** `/Users/peleke/Documents/Projects/openclaw/src/green/`

The green module is a self-contained carbon accounting layer with the following architecture:

- **`carbon-calculator.ts`** -- Pure calculation engine. Takes token counts and provider/model, returns CO2 grams, water mL, and GHG Protocol compliance metadata (scope, category, calculation method, data quality score, uncertainty bounds).
- **`store.ts`** -- SQLite persistence at `~/.openclaw/green/green.db`. Full CRUD for `carbon_traces` table and `carbon_targets` (SBTi). Aggregation queries for summaries, timeseries, provider breakdown. Period-based queries for regulatory exports.
- **`trace-capture.ts`** -- Post-run hook called from `pi-embedded-runner/run.ts` after every LLM inference. `captureAndStoreGreenTrace()` calculates carbon and writes to SQLite. Swallows errors to avoid disrupting the main flow.
- **`exports.ts`** -- Regulatory export functions for GHG Protocol, CDP, TCFD, and ISO 14064-1:2018 formats.
- **`api.ts`** -- HTTP API handler at `/__openclaw__/api/green/` serving summary, traces, timeseries, intensity, targets, config, factors, and all four export formats.
- **`dashboard-html.ts`** -- Self-contained HTML dashboard with Chart.js (dark theme). Serves from the API handler. Auto-refreshes every 30s.
- **`regional-grid.ts`** -- Static IEA country averages and cloud region mapping. API integration stubs for Electricity Maps and EPA eGRID.
- **`config.ts`** -- Default carbon factors for Anthropic and OpenAI models.

**Key finding: Zero OTel emission.** The green module writes to SQLite only. There is no OpenTelemetry integration whatsoever -- no spans, no metrics, no exporters. The module has no dependency on `@opentelemetry/*`. It operates as a completely isolated persistence layer.

**Call site:** The trace is captured at `/Users/peleke/Documents/Projects/openclaw/src/agents/pi-embedded-runner/run.ts` line 714, after every LLM inference run.

### 1.2 qortex-observe Carbon Module (Python)

**Location:** `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/carbon/`

This is a Python port of the openclaw calculator logic, adapted for the qortex event-driven observability pipeline:

- **`calculator.py`** -- Identical calculation logic to the TypeScript version. `calculate_carbon()`, `calculate_equivalents()`, `find_carbon_factor()`.
- **`config.py`** -- Same emission factors as openclaw (Anthropic + OpenAI models, fallback factor).
- **`types.py`** -- Python dataclasses mirroring the TypeScript types: `CarbonFactor`, `CarbonCalculation`, `CarbonSummary`, `CarbonEquivalents`, plus regulatory export types.
- **`ghg.py`** -- Regulatory export functions (`export_ghg_protocol`, `export_cdp`, `export_tcfd`, `export_iso14064`).

### 1.3 qortex-observe OTel Pipeline (Python)

**Location:** `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/`

This is a fully wired, event-driven observability system:

- **Event bus:** `events.py` defines `CarbonTracked` (frozen dataclass) with fields: `provider`, `model`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `total_co2_grams`, `water_ml`, `confidence`, `timestamp`.
- **Emission point:** `/Users/peleke/Documents/Projects/qortex/src/qortex/enrichment/anthropic.py` line 207 emits `CarbonTracked` after every Anthropic API call via `emit(CarbonTracked(...))`.
- **Metric schema:** `metrics_schema.py` defines 4 carbon metrics already:
  - `qortex_carbon_co2_grams` -- Counter (provider, model) -- cumulative CO2
  - `qortex_carbon_water_ml` -- Counter (provider, model) -- cumulative water
  - `qortex_carbon_tokens` -- Counter (provider, model) -- cumulative tokens
  - `qortex_carbon_confidence` -- Gauge -- latest confidence score
- **Metric handler:** `metrics_handlers.py` lines 264-271 handle `CarbonTracked` events, incrementing the above instruments.
- **Metric factory:** `metrics_factory.py` creates OTel instruments from the schema and applies histogram bucket Views.
- **OTel subscriber:** `subscribers/otel.py` sets up TracerProvider + OTLP exporters (gRPC with HTTP/protobuf fallback).
- **Emitter:** `emitter.py` `configure()` wires everything: creates MeterProvider with OTLP push reader and/or Prometheus pull reader, creates instruments, registers handlers, sets up trace export.
- **Tracing:** `tracing.py` provides `@traced` decorator for OTel spans and `SelectiveSpanProcessor` for filtering.
- **MCP propagation:** `mcp/_propagation.py` handles W3C traceparent extraction for distributed tracing across the TypeScript-Python boundary.

### 1.4 Infrastructure (Docker Stack)

**Location:** `/Users/peleke/Documents/Projects/qortex/docker/`

The observability stack is fully provisioned:

- **OTel Collector** (`docker/otel-collector/otel-collector-config.yaml`): Receives OTLP gRPC/HTTP on 4317/4318, routes traces to Jaeger, exports metrics via Prometheus endpoint on 8889.
- **Jaeger** (all-in-one): OTLP-enabled, UI at 16686.
- **Prometheus** (`docker/prometheus/prometheus.yml`): Scrapes qortex direct (host:9464), sandbox (host:9090), and OTel Collector (otel-collector:8889).
- **VictoriaLogs**: Structured log aggregation.
- **Grafana** (port 3010): Provisioned with Prometheus + VictoriaLogs datasources. Dashboard at `docker/grafana/dashboards/qortex.json` (1923 lines). **No Jaeger datasource** is configured yet.
- **Existing dashboard sections:** Embedding Index, Vec Search, Bandit Arms, Credit Propagation, KG Coverage. **No carbon/green section exists.**

### 1.5 Summary of Gaps

| Gap | Location | Impact |
|-----|----------|--------|
| **No OTel emission from openclaw green** | `openclaw/src/green/trace-capture.ts` | Carbon data from the TypeScript runtime never reaches the OTel collector |
| **No Grafana Jaeger datasource** | `docker/grafana/provisioning/datasources/datasources.yml` | Traces cannot be queried or linked from dashboard panels |
| **No carbon panels in Grafana dashboard** | `docker/grafana/dashboards/qortex.json` | The 4 carbon OTel metrics exist but have no visualization |
| **Single CarbonTracked emission point** | Only `qortex/src/qortex/enrichment/anthropic.py` emits | Carbon is only tracked for enrichment LLM calls, not all inference |
| **CarbonTracked event lacks fields** | `events.py` CarbonTracked | Missing: `scope`, `category`, `calculation_method`, `data_quality_score`, `region`, `grid_carbon_used`, `water_ml_breakdown` (input/output/cache CO2 separately) |
| **No carbon OTel spans** | Neither system creates spans for carbon calculations | Jaeger has no carbon trace data to display |
| **Missing metrics for dashboard** | `metrics_schema.py` | Missing: per-inference CO2 histogram, CO2 rate, water rate, intensity metrics, uncertainty gauge, data quality gauge |
| **No cross-boundary carbon propagation** | Between openclaw TS and qortex Python | openclaw's carbon traces don't bridge into qortex's OTel pipeline |

---

## 2. Architecture: How the Two Systems Connect

The connection point between vindler/openclaw (TypeScript) and qortex-observe (Python) is the **MCP protocol boundary**. The existing `mcp/_propagation.py` already handles W3C traceparent propagation across this boundary. The architecture for carbon metrics should follow the same pattern:

```
openclaw (TypeScript runtime)
  |
  |-- pi-embedded-runner/run.ts calls captureAndStoreGreenTrace()
  |     |-- writes to SQLite (existing, keep for local dashboard)
  |     |-- NEW: emit carbon metrics via OTLP to collector
  |
  |-- MCP tool calls to qortex Python server
  |     |-- params._meta contains traceparent (existing)
  |     |-- NEW: params._meta carries carbon trace context
  |
qortex Python server
  |
  |-- enrichment/anthropic.py emits CarbonTracked (existing)
  |-- NEW: MCP handler emits CarbonTracked from openclaw's data
  |
  |-- observe event bus routes CarbonTracked to:
  |     |-- metrics_handlers.py -> OTel instruments (existing)
  |     |-- structlog subscriber -> logs (existing)
  |     |-- NEW: span creation for carbon traces
  |
  |-- OTel SDK pushes to:
        |-- OTLP gRPC -> OTel Collector
              |-- traces -> Jaeger (existing pipeline)
              |-- metrics -> Prometheus (existing pipeline)
                    |-- Grafana queries Prometheus (existing)
                    |-- NEW: Grafana carbon dashboard section
```

**Two viable strategies for getting openclaw carbon data into Grafana:**

**Strategy A (Recommended): TypeScript-side OTLP push.** Add `@opentelemetry/sdk-metrics` to openclaw's green module. After `captureAndStoreGreenTrace()`, emit the same metrics (CO2, water, tokens, confidence) via OTLP to the collector at `localhost:4317`. This means both TypeScript and Python runtimes push metrics independently to the same Prometheus via the collector.

**Strategy B: MCP relay.** When openclaw calls qortex via MCP, include carbon trace data in `_meta`. The Python server extracts it and emits `CarbonTracked`. This piggybacks on existing MCP propagation but means carbon data only flows when MCP calls happen, missing standalone openclaw sessions.

**Recommendation:** Strategy A for metrics (both runtimes push independently), Strategy B for traces (carbon spans linked to the inference span via traceparent).

---

## 3. Implementation Plan

### Phase 1: Expand the qortex-observe Carbon Metrics (Python side)

**Goal:** Enrich the metric schema and CarbonTracked event to support all dashboard needs.

**Step 1.1: Expand CarbonTracked event fields**

File: `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/events.py`

Add optional fields to `CarbonTracked`:
- `input_co2_grams: float = 0.0` (per-token-type breakdown)
- `output_co2_grams: float = 0.0`
- `cache_co2_grams: float = 0.0`
- `scope: int = 3`
- `data_quality_score: int = 3`
- `calculation_method: str = "average-data"`
- `region: str | None = None`
- `grid_carbon_used: float = 400.0`

This is backwards-compatible since all new fields have defaults.

**Step 1.2: Add carbon metrics to schema**

File: `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/metrics_schema.py`

Add these MetricDefs after the existing 4 carbon metrics:
- `qortex_carbon_co2_per_inference` -- Histogram (provider, model) -- CO2 per individual call. Buckets: (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
- `qortex_carbon_water_per_inference` -- Histogram (provider, model) -- water per call. Buckets: (0.01, 0.1, 0.5, 1.0, 5.0, 10.0)
- `qortex_carbon_data_quality` -- Gauge -- latest data quality score (1-5)
- `qortex_carbon_intensity_per_million_tokens` -- Gauge (provider, model) -- running intensity metric
- `qortex_carbon_inferences` -- Counter (provider, model) -- count of carbon-tracked inferences

**Step 1.3: Wire new metrics in handlers**

File: `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/metrics_handlers.py`

Extend `_on_carbon_tracked` handler to record the new instruments from each `CarbonTracked` event.

**Step 1.4: Add carbon span creation**

File: `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/metrics_handlers.py` (or a new `subscribers/carbon_trace.py`)

When a `CarbonTracked` event fires, optionally create an OTel span `carbon.inference` with attributes:
- `carbon.co2_grams`
- `carbon.water_ml`
- `carbon.provider`
- `carbon.model`
- `carbon.confidence`
- `carbon.scope`
- `carbon.data_quality_score`

This span would appear in Jaeger, linked to the parent inference span via context propagation.

### Phase 2: Add OTLP Emission to openclaw Green Module (TypeScript side)

**Goal:** Get openclaw's carbon traces flowing to the OTel Collector alongside its existing SQLite persistence.

**Step 2.1: Add OTel dependencies to openclaw**

Add `@opentelemetry/api`, `@opentelemetry/sdk-metrics`, `@opentelemetry/exporter-metrics-otlp-grpc` (or `http`) to openclaw's dependencies.

**Step 2.2: Create OTel emitter in green module**

New file: `openclaw/src/green/otel-emitter.ts`

- Initializes a MeterProvider with an OTLP push metric reader
- Creates instruments matching the qortex metric names:
  - `qortex_carbon_co2_grams` (Counter)
  - `qortex_carbon_water_ml` (Counter)
  - `qortex_carbon_tokens` (Counter)
  - `qortex_carbon_confidence` (Gauge)
  - `qortex_carbon_co2_per_inference` (Histogram)
  - `qortex_carbon_inferences` (Counter)
- Exposes `emitCarbonMetrics(trace: CarbonTrace): void`
- Gracefully degrades if OTel collector is unreachable (same pattern as the Python side)
- Use `service.name = "openclaw"` to distinguish from qortex Python metrics in Grafana

**Step 2.3: Wire emitter into trace-capture**

File: `/Users/peleke/Documents/Projects/openclaw/src/green/trace-capture.ts`

In `captureAndStoreGreenTrace()`, after the SQLite insert, call `emitCarbonMetrics(trace)`. The emitter should be lazy-initialized and swallow errors (matching the existing error-swallowing pattern).

**Step 2.4: Add OTel span for carbon capture**

In the same function, create an OTel span `green.carbon_capture` with carbon trace attributes. Link it to the parent inference span if W3C traceparent context is available from the MCP call chain.

### Phase 3: Infrastructure Updates

**Step 3.1: Tempo datasource in Grafana** *(DONE — Tempo replaced Jaeger)*

The Tempo datasource is already provisioned at `docker/grafana/provisioning/datasources/datasources.yml`. Traces are queried via Grafana Explore with TraceQL. No additional datasource setup needed.

**Step 3.2: Verify Prometheus scraping**

The existing Prometheus config already scrapes the OTel Collector at `otel-collector:8889`. TypeScript-side OTLP metrics pushed to the collector will be exposed at that endpoint. No Prometheus config change needed.

**Step 3.3: Verify OTel Collector pipelines**

The existing collector config handles traces (to Tempo via OTLP gRPC), metrics (to Prometheus exporter), and service graph generation (servicegraph connector). No collector config change needed.

### Phase 4: Grafana Dashboard -- Green Section

**Goal:** Add a "Green: Environmental Impact" row section to the existing `qortex.json` dashboard.

File: `/Users/peleke/Documents/Projects/qortex/docker/grafana/dashboards/qortex.json`

#### 4.1 Row Header Panel

A collapsible row titled "Green: Environmental Impact" with a descriptive subtitle.

#### 4.2 Architecture Diagram Panel (jdbranham-diagram-panel)

A Mermaid flowchart showing the carbon tracking pipeline:
```
LLM Call -> Token Count -> Carbon Calculator -> CO2 + Water -> OTel Metrics -> Grafana
```

#### 4.3 Explanation Panel (text/markdown)

A markdown panel explaining what each signal means, matching the pattern used by existing sections (Embedding Index, Bandit Loop). Include the table:

| Signal | What to watch | Healthy | Investigate |
|--------|--------------|---------|-------------|
| CO2/Inference | Grams per API call | Consistent with model size | Sudden jumps = model change or degraded caching |
| Cumulative CO2 | Total over time | Linear growth | Exponential = runaway inference loops |
| Water Usage | mL per inference | Proportional to tokens | Spikes = large context windows |
| Confidence | Factor reliability | > 0.3 | < 0.2 = mostly guessing |
| Data Quality | GHG Protocol score | 3 or better | 5 = need better emission factors |

#### 4.4 Stat Panels (top-line KPIs)

Four stat panels in a row:

1. **Total CO2 (grams):** `sum(qortex_carbon_co2_grams_total)`
2. **Total Water (mL):** `sum(qortex_carbon_water_ml_total)`
3. **Total Carbon-Tracked Inferences:** `sum(qortex_carbon_inferences_total)`
4. **Avg Confidence:** `avg(qortex_carbon_confidence)`

#### 4.5 Time Series Panels

1. **CO2 Emissions Over Time** (timeseries, full width):
   - `sum(increase(qortex_carbon_co2_grams_total[$__rate_interval]))` by provider
   - Shows CO2 rate, stacked by provider (Anthropic, OpenAI, etc.)

2. **CO2 Per Inference Distribution** (timeseries, half width):
   - `histogram_quantile(0.50, sum(increase(qortex_carbon_co2_per_inference_bucket[15m])) by (le))`
   - p50 and p95 lines showing per-inference CO2 distribution

3. **Water Usage Rate** (timeseries, half width):
   - `sum(increase(qortex_carbon_water_ml_total[$__rate_interval]))` by provider

#### 4.6 Breakdown Panels

1. **CO2 by Provider** (piechart, half width):
   - `sum by (provider) (qortex_carbon_co2_grams_total)`

2. **CO2 by Model** (bar chart, half width):
   - `topk(10, sum by (model) (qortex_carbon_co2_grams_total))`

#### 4.7 Compliance & Quality Panel

1. **Confidence Score** (gauge):
   - `avg(qortex_carbon_confidence)` with thresholds: red < 0.2, yellow 0.2-0.5, green > 0.5

2. **Data Quality Score** (gauge, if implemented):
   - `avg(qortex_carbon_data_quality)` with inverted thresholds (1=best, 5=worst)

3. **Carbon Intensity** (stat):
   - Compute intensity per million tokens from counter ratio

#### 4.8 Trace Exploration Panel (optional, requires Jaeger datasource)

A Jaeger trace panel filtered to `carbon.inference` spans, showing individual carbon calculations with their attributes.

### Phase 5: Expand CarbonTracked Emission Points

**Goal:** Ensure all LLM calls emit CarbonTracked, not just enrichment.

**Step 5.1:** Audit all LLM call sites in qortex beyond `enrichment/anthropic.py`. Any place that calls an LLM API should calculate carbon and emit `CarbonTracked`.

**Step 5.2:** On the openclaw side, the runner already calls `captureAndStoreGreenTrace` for every inference. After Phase 2, these will automatically flow to OTel.

---

## 4. Metric Naming Convention

All carbon metrics use the `qortex_carbon_` prefix, consistent with existing qortex metrics. Both the TypeScript and Python runtimes emit the same metric names so they aggregate correctly in Prometheus/Grafana. The `service.name` resource attribute distinguishes the origin (`openclaw` vs `qortex`).

Existing metrics (keep):
- `qortex_carbon_co2_grams` (counter)
- `qortex_carbon_water_ml` (counter)
- `qortex_carbon_tokens` (counter)
- `qortex_carbon_confidence` (gauge)

New metrics (add):
- `qortex_carbon_co2_per_inference` (histogram)
- `qortex_carbon_water_per_inference` (histogram)
- `qortex_carbon_inferences` (counter)
- `qortex_carbon_data_quality` (gauge)
- `qortex_carbon_intensity_per_million_tokens` (gauge)

---

## 5. Implementation Order

| Order | Phase | Effort | Dependencies |
|-------|-------|--------|--------------|
| 1 | Phase 1: Expand Python metrics schema + handlers | Small | None |
| 2 | Phase 3.1: Add Jaeger datasource to Grafana | Trivial | None |
| 3 | Phase 4: Build Grafana dashboard section | Medium | Phase 1 (metrics must exist) |
| 4 | Phase 2: Add OTLP to openclaw green module | Medium | Phase 1 (same metric names) |
| 5 | Phase 1.4: Carbon span creation for Jaeger | Small | Phase 3.1 (Jaeger datasource) |
| 6 | Phase 5: Expand emission points | Small | Phase 2 (infrastructure ready) |

Phases 1, 3.1, and 4 can be done in a single PR against the qortex repo. Phase 2 is a separate PR against vindler/openclaw. Phase 5 is follow-up work.

---

## 6. Risks and Mitigations

1. **Metric cardinality explosion:** The `model` label could have high cardinality if users use many models. Mitigation: the existing factors table has 6 entries; this is bounded.

2. **OTel collector unreachable from openclaw:** The TypeScript runtime may not always have the Docker stack running. Mitigation: use the same error-swallowing pattern already in `captureAndStoreGreenTrace()`. SQLite remains the reliable local store; OTel emission is best-effort.

3. **Duplicate counting:** Both openclaw (TypeScript) and qortex (Python) might count the same inference if the MCP relay strategy (B) is also used. Mitigation: use Strategy A only (direct push from each runtime) and distinguish by `service.name` in Grafana queries.

4. **Dashboard JSON complexity:** The existing `qortex.json` is 1923 lines. Adding ~300 lines for the green section is manageable but requires careful grid positioning. Mitigation: use a collapsible row so it does not push existing panels.

---

### Critical Files for Implementation

- `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/metrics_schema.py` - Add new carbon MetricDefs (CO2 histogram, water histogram, inferences counter, data quality gauge, intensity gauge)
- `/Users/peleke/Documents/Projects/qortex/packages/qortex-observe/src/qortex/observe/metrics_handlers.py` - Wire new CarbonTracked fields to new instruments; optionally add carbon span creation
- `/Users/peleke/Documents/Projects/qortex/docker/grafana/dashboards/qortex.json` - Add the Green section (row header, stat panels, timeseries, pie charts, gauges)
- `/Users/peleke/Documents/Projects/openclaw/src/green/trace-capture.ts` - Integration point to add OTLP emission after SQLite insert
- `/Users/peleke/Documents/Projects/qortex/docker/grafana/provisioning/datasources/datasources.yml` - Add Jaeger datasource for trace exploration panels