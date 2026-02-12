# Observability and Grafana Dashboard

qortex ships a full observability stack: structured events, OpenTelemetry traces and metrics, Prometheus scraping, and a pre-built Grafana dashboard that visualizes the entire pipeline.

## Architecture

```
qortex process
  ├─ emit(Event)          ← typed frozen dataclass
  │   ├─ OTel subscriber  → OTel Collector → Prometheus (remote write)
  │   └─ Prom subscriber  → HTTP /metrics (local scrape target)
  └─ structlog logger     → stdout / JSONL sink / VictoriaLogs
```

Events are emitted at every stage of the pipeline: ingestion, vector index operations (add/search), retrieval (vec search, online edge generation, PPR), feedback (factor updates), enrichment, and buffer promotion. Each subscriber translates events into counters, histograms, and gauges.

## Quick Start

```bash
# Start the observability stack
cd docker && docker compose up -d

# Verify services
curl -s http://localhost:9091/api/v1/query?query=up | python3 -m json.tool

# Open the dashboard
open http://localhost:3010/d/qortex-main/qortex-observability
```

| Service | Port | Purpose |
|---------|------|---------|
| Memgraph | 7687 | Graph database |
| Memgraph Lab | 3000 | Memgraph web UI |
| OTel Collector | 4317, 4318 | Receives OTLP (gRPC + HTTP) |
| Prometheus | 9091 | Metrics storage + PromQL |
| Grafana | 3010 | Dashboard visualization |
| Jaeger | 16686 | Trace viewer |
| VictoriaLogs | 9428 | Log aggregation |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_OTEL_ENABLED` | `false` | Enable OTel metrics export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | OTel Collector endpoint (e.g. `http://localhost:4318`) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | — | Protocol (`http/protobuf` or `grpc`) |
| `QORTEX_PROMETHEUS_ENABLED` | `false` | Enable local Prometheus HTTP server |
| `QORTEX_PROMETHEUS_PORT` | `9464` | Port for the local `/metrics` endpoint |
| `MEMGRAPH_USER` | — | Memgraph Bolt auth username |
| `MEMGRAPH_PASSWORD` | — | Memgraph Bolt auth password |

## Dashboard Panels

The Grafana dashboard (`qortex-main`) is organized into seven sections. Each section corresponds to a stage of the pipeline.

### Retrieval Health

These panels show the query lifecycle: from the moment `adapter.retrieve()` is called to when it returns results.

#### Query Rate (queries/sec)

- **Metric:** `rate(qortex_queries_total[5m])`
- **Labels:** `mode` (`graph` or `vec`)
- **Source event:** `QueryCompleted` (emitted at the end of `GraphRAGAdapter.retrieve()`)
- **What it tells you:** How many retrieval queries are completing per second. A sudden drop means queries are failing or the system is idle. A spike indicates burst load.

#### Query Latency (p50/p95/p99)

- **Metric:** `histogram_quantile(0.50|0.95|0.99, rate(qortex_query_duration_seconds_bucket[5m]))`
- **Source event:** `QueryCompleted` (carries `latency_ms`)
- **What it tells you:** End-to-end retrieve latency from query embedding through vec search, online edge generation, PPR, and scoring. p95 above 1s suggests a bottleneck. Check vec search and PPR panels to isolate which stage is slow.

#### Vec Search Latency (p50/p95)

- **Metric:** `histogram_quantile(0.50|0.95, rate(qortex_vec_search_duration_seconds_bucket[5m]))`
- **Source event:** `VecSearchCompleted` (emitted after the embedding + vector similarity step)
- **What it tells you:** Time spent embedding the query and searching the vector index for seed candidates. If this dominates query latency, the embedding model or index size is the bottleneck, not the graph.

#### Query Errors

- **Metric:** `rate(qortex_query_errors_total[5m])`
- **Labels:** `stage` (which pipeline stage failed)
- **Source event:** `QueryFailed`
- **Note:** This event is defined but currently not emitted by any code path. The panel exists as a placeholder for future error tracking. If you see data here, something new is emitting `QueryFailed`.

### Learning Dynamics

These panels track how the system learns from feedback. Teleportation factors bias PPR toward nodes the user finds helpful and away from unhelpful ones.

#### Factor Mean Over Time

- **Metric:** `qortex_factor_mean`
- **Source event:** `FactorDriftSnapshot` (emitted after each batch of factor updates)
- **What it tells you:** The average teleportation factor across all nodes. Starts at 1.0 (uniform). Moves above 1.0 when accepted nodes accumulate boosts. A rising mean signals the system is developing preferences. A flat line at 1.0 means no feedback is flowing.

#### Factor Entropy

- **Metric:** `qortex_factor_entropy`
- **Unit:** bits
- **Source event:** `FactorDriftSnapshot`
- **What it tells you:** Shannon entropy of the factor distribution. High entropy = factors are spread evenly (system is uncertain). Low entropy = factors are concentrated on a few nodes (system has strong preferences). Entropy should decrease as the system receives consistent feedback.

#### Factor Update Rate

- **Metric:** `rate(qortex_factor_updates_total[5m])`
- **Labels:** `outcome` (`accepted`, `rejected`, `partial`)
- **Source event:** `FactorUpdated` (one per node per feedback call)
- **What it tells you:** How fast teleportation factors are changing, broken down by outcome type. If rejected vastly outpaces accepted, the system is serving poor results. A balanced ratio suggests healthy learning.

#### Feedback Accept/Reject Ratio

- **Metric:** `rate(qortex_feedback_total{outcome="accepted"|"rejected"}[5m])`
- **Source event:** `FeedbackReceived` (emitted by `GraphRAGAdapter.feedback()`)
- **What it tells you:** The raw accept vs reject rate from user feedback. This is the primary signal for retrieval quality. If rejects trend upward over time, something is degrading.

### KG Crystallization

These panels track how the knowledge graph evolves: online edges solidifying into persistent structure.

#### KG Coverage Ratio

- **Metric:** `qortex_kg_coverage`
- **Display:** Gauge, 0-100%
- **Source events:** `KGCoverageComputed` (during retrieve) and `BufferFlushed`
- **What it tells you:** The ratio of persistent KG edges to total edges (persistent + online) for a query's candidate set. 100% means the KG fully covers the retrieval neighborhood (no online edges needed). Low coverage means the system is filling gaps with cosine-similarity edges. Coverage should trend upward as online edges get promoted.
- **Correlation with learning:** As coverage rises, PPR operates over more stable, vetted structure. Quality should improve.

#### Buffer Size and Promotions

- **Metrics:** `qortex_buffer_edges` (gauge: current buffer size), `rate(qortex_edges_promoted_total[1h])` (promotions per hour)
- **Source events:** `OnlineEdgeRecorded` (buffer size), `EdgePromoted` (promotion count)
- **What it tells you:** How many candidate edges are waiting in the promotion buffer and how fast they graduate to the persistent KG. A growing buffer with zero promotions means edges aren't being observed often enough (or the promotion threshold is too high). Steady promotion rate = healthy crystallization.

#### Total Promoted (lifetime)

- **Metric:** `qortex_edges_promoted_total`
- **Display:** Stat panel (single number)
- **What it tells you:** Lifetime count of online edges that met the promotion threshold and were written to the persistent KG.

### PPR Performance

These panels show the graph algorithm that powers retrieval scoring.

#### PPR Executions / sec

- **Metric:** `rate(qortex_ppr_started_total[5m])`
- **Source event:** `PPRStarted`
- **What it tells you:** How often Personalized PageRank runs. Should track query rate 1:1 since every `retrieve()` call triggers one PPR run.

#### PPR Iterations to Convergence

- **Metric:** `rate(qortex_ppr_iterations_bucket[5m])`
- **Display:** Histogram
- **Source events:** `PPRConverged`, `PPRDiverged`
- **What it tells you:** How many power iteration steps PPR needs to converge. Fewer iterations = faster convergence. If iterations cluster near `max_iterations` (100), PPR is not converging; the graph may be too dense or disconnected. Typical healthy range: 20-50 iterations.

#### Active Factors and Node Count

- **Metric:** `qortex_factors_active`
- **Source event:** `FactorDriftSnapshot`
- **What it tells you:** How many nodes have non-default teleportation factors. If this equals your total node count, every node has received feedback at some point.

### Online Edge Generation

#### Online Edge Count and Generation Rate

- **Metrics:** `qortex_online_edge_count` (gauge: edges per last query), `rate(qortex_online_edges_generated_total[5m])` (events/sec)
- **Source event:** `OnlineEdgesGenerated`
- **What it tells you:** How many cosine-similarity edges are generated per query to fill KG gaps. High counts mean the KG is sparse for those queries. As KG coverage improves (via edge promotion), this should trend downward.
- **Correlation with KG coverage:** online edge count should inversely correlate with the KG coverage gauge. If both are rising, online edges are being generated but not promoted. Check the promotion threshold or buffer flush frequency.

#### KG Coverage Over Time

- **Metric:** `qortex_kg_coverage`
- **Display:** Time series, 0-100%
- **What it tells you:** Same metric as the gauge above, but as a time series to show the trend. An upward slope means the KG is maturing.

### Enrichment and Ingestion

#### Enrichment Rate

- **Metric:** `rate(qortex_enrichment_total[5m])`
- **Labels:** `backend_type` (e.g. `template`, `AnthropicEnrichmentBackend`)
- **Source event:** `EnrichmentCompleted`
- **What it tells you:** How often rules are enriched (context, antipatterns, rationale added). The enrichment pipeline is separate from retrieve: it runs during ingestion or on-demand.

#### Enrichment Latency (p50/p95)

- **Metric:** `histogram_quantile(0.50|0.95, rate(qortex_enrichment_duration_seconds_bucket[5m]))`
- **Source event:** `EnrichmentCompleted`
- **What it tells you:** Time to enrich a batch of rules. Template enrichment is sub-millisecond. LLM-backed enrichment can be seconds. Watch for p95 spikes indicating API timeouts.

#### Enrichment Fallbacks

- **Metric:** `rate(qortex_enrichment_fallbacks_total[5m])`
- **Source event:** `EnrichmentFallback`
- **What it tells you:** How often the enrichment backend fails and falls back to template-based enrichment. Spikes here indicate LLM API issues.

#### Ingestion Rate

- **Metric:** `rate(qortex_manifests_ingested_total[5m])`
- **Labels:** `domain`
- **Source event:** `ManifestIngested`
- **What it tells you:** How often knowledge manifests are ingested, broken down by domain. Each manifest contains nodes, edges, and rules from a single source.

#### Ingest Latency (p50/p95)

- **Metric:** `histogram_quantile(0.50|0.95, rate(qortex_ingest_duration_seconds_bucket[5m]))`
- **Source event:** `ManifestIngested`
- **What it tells you:** Time to ingest a manifest into the graph backend. For Memgraph, this includes node/edge/rule creation via Cypher. Latency scales with manifest size.

### Vector Index

These panels provide visibility into the vec layer: the index that stores embeddings and serves as the seed source for graph retrieval. Previously the vec layer was a black box; now you can see how it behaves.

#### Vec Index Size

- **Metric:** `qortex_vec_index_size`
- **Display:** Stat panel (single number)
- **Source event:** `VecIndexUpdated` (emitted from `NumpyVectorIndex.add()` and `SqliteVecIndex.add()`)
- **What it tells you:** Total number of vectors in the index. Should match the number of ingested nodes that have embeddings. A stale or low number means embeddings aren't being stored properly.

#### Vec Add Rate

- **Metric:** `rate(qortex_vec_add_total[5m])`
- **Labels:** `index_type` (`numpy` or `sqlite`)
- **Source event:** `VecIndexUpdated`
- **What it tells you:** How often vectors are being added to the index. Spikes during ingestion, flat during query-only workloads. If you ingest but this stays at zero, embeddings aren't reaching the vec index.

#### Vec Add Latency (p50/p95)

- **Metric:** `histogram_quantile(0.50|0.95, rate(qortex_vec_add_duration_seconds_bucket[5m]))`
- **Source event:** `VecIndexUpdated`
- **What it tells you:** Time to add a batch of vectors. NumpyVectorIndex is sub-millisecond for small batches. SqliteVecIndex involves disk I/O. If add latency spikes during ingestion, the index is becoming a bottleneck.

#### Vec Search Top Score

- **Metric:** `qortex_vec_search_top_score`
- **Display:** Time series, range 0-1
- **Source event:** `VecSearchResults` (emitted from the index `.search()` method)
- **What it tells you:** The highest cosine similarity score from the last vector search. High scores (> 0.8) mean the index contains strong matches for the query. Consistently low scores (< 0.3) mean the embedding space doesn't well-represent the queries. Consider a different embedding model, or the index may be under-populated.
- **Correlation with graph learning:** When top scores are high, the seeds fed into PPR are strong, leading to better graph traversal. Low top scores produce weak seeds and noisy PPR results.

#### Vec Search Score Spread

- **Metric:** `qortex_vec_search_score_spread`
- **Source event:** `VecSearchResults`
- **What it tells you:** The difference between the top and bottom cosine sim scores in a single search. A wide spread (> 0.3) means the index is clearly distinguishing relevant from irrelevant vectors, indicating good signal quality. A narrow spread (< 0.05) means results are clustered close together, making it hard for PPR to differentiate.
- **Correlation with graph learning:** High spread = clear ranking signal for PPR seeds. Low spread = PPR is working with near-uniform weights, reducing its ability to focus activation on the most relevant subgraph.

#### Vec Seed Yield

- **Metric:** `qortex_vec_seed_yield`
- **Display:** Gauge, 0-100%
- **Source event:** `VecSeedYield` (emitted from `GraphRAGAdapter.retrieve()` after domain filtering)
- **What it tells you:** The ratio of vec search results that survive domain filtering to become PPR seeds. A yield of 100% means every vec match was in the requested domain. Low yield (< 50%) means the vec index returns many cross-domain results that get discarded. The domain structure may need attention, or the embedding model doesn't capture domain boundaries well.
- **Correlation with graph learning:** Low yield wastes compute (vec search finds candidates that are immediately discarded). If yield drops over time, domain-specific re-indexing may help.

#### Vec Search Candidates Distribution

- **Metric:** `rate(qortex_vec_search_candidates_sum[5m]) / rate(qortex_vec_search_candidates_count[5m])`
- **Source event:** `VecSearchResults`
- **What it tells you:** Average number of candidates returned per vec search. Should be close to `fetch_k` (typically `top_k * 3`). If consistently lower, the index is small or the threshold is filtering aggressively.

## Complete Metric Reference

| Metric | Type | Event | Labels |
|--------|------|-------|--------|
| `qortex_queries_total` | Counter | `QueryCompleted` | `mode` |
| `qortex_query_duration_seconds` | Histogram | `QueryCompleted` | — |
| `qortex_vec_search_duration_seconds` | Histogram | `VecSearchCompleted` | — |
| `qortex_query_errors_total` | Counter | `QueryFailed` | `stage` |
| `qortex_factor_mean` | Gauge | `FactorDriftSnapshot` | — |
| `qortex_factor_entropy` | Gauge | `FactorDriftSnapshot` | — |
| `qortex_factors_active` | Gauge | `FactorDriftSnapshot` | — |
| `qortex_factor_updates_total` | Counter | `FactorUpdated` | `outcome` |
| `qortex_feedback_total` | Counter | `FeedbackReceived` | `outcome` |
| `qortex_kg_coverage` | Gauge | `KGCoverageComputed`, `BufferFlushed` | — |
| `qortex_buffer_edges` | Gauge | `OnlineEdgeRecorded` | — |
| `qortex_edges_promoted_total` | Counter | `EdgePromoted` | — |
| `qortex_ppr_started_total` | Counter | `PPRStarted` | — |
| `qortex_ppr_iterations` | Histogram | `PPRConverged`, `PPRDiverged` | — |
| `qortex_online_edges_generated_total` | Counter | `OnlineEdgesGenerated` | — |
| `qortex_online_edge_count` | Gauge | `OnlineEdgesGenerated` | — |
| `qortex_enrichment_total` | Counter | `EnrichmentCompleted` | `backend_type` |
| `qortex_enrichment_duration_seconds` | Histogram | `EnrichmentCompleted` | — |
| `qortex_enrichment_fallbacks_total` | Counter | `EnrichmentFallback` | — |
| `qortex_manifests_ingested_total` | Counter | `ManifestIngested` | `domain` |
| `qortex_ingest_duration_seconds` | Histogram | `ManifestIngested` | — |
| `qortex_vec_index_size` | Gauge | `VecIndexUpdated` | — |
| `qortex_vec_add_total` | Counter | `VecIndexUpdated` | `index_type` |
| `qortex_vec_add_duration_seconds` | Histogram | `VecIndexUpdated` | — |
| `qortex_vec_search_candidates` | Histogram | `VecSearchResults` | — |
| `qortex_vec_search_top_score` | Gauge | `VecSearchResults` | — |
| `qortex_vec_search_score_spread` | Gauge | `VecSearchResults` | — |
| `qortex_vec_seed_yield` | Gauge | `VecSeedYield` | — |

## Testing the Dashboard

The full-pipeline E2E test exercises every code path and verifies every metric:

```bash
QORTEX_GRAPH=memgraph MEMGRAPH_USER=memgraph MEMGRAPH_PASSWORD=memgraph \
QORTEX_OTEL_ENABLED=true OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
QORTEX_PROMETHEUS_ENABLED=true QORTEX_PROMETHEUS_PORT=9464 \
uv run pytest tests/test_full_pipeline_e2e.py -v -s
```

This test ingests a knowledge graph, runs 25 retrieval queries, submits feedback, triggers edge promotion, and runs enrichment, then asserts every metric is present in Prometheus and queryable through Grafana.
