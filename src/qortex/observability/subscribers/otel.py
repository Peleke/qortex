"""OpenTelemetry subscriber: routes events to OTel spans and metrics.

Requires qortex[observability] (opentelemetry-api, opentelemetry-sdk,
opentelemetry-exporter-otlp).

Push-based: metrics and traces are pushed to an OTLP endpoint (collector,
Jaeger, etc.). The OTEL Collector's Prometheus exporter converts these to
Prometheus-format metrics, so the same Grafana dashboard works whether
qortex runs locally or in a remote sandbox.

Metric naming convention:
    Counter  "qortex_foo"         -> Prometheus "qortex_foo_total"
    Histogram "qortex_bar_seconds" -> Prometheus "qortex_bar_seconds_bucket"
    Gauge    "qortex_baz"         -> Prometheus "qortex_baz"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qortex.observability.config import ObservabilityConfig


def register_otel_subscriber(config: ObservabilityConfig) -> None:
    """Register OTel trace + metric subscribers."""
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from qortex.observability.events import (
        BufferFlushed,
        EdgePromoted,
        EnrichmentCompleted,
        EnrichmentFallback,
        FactorDriftSnapshot,
        FactorUpdated,
        FeedbackReceived,
        ManifestIngested,
        OnlineEdgeRecorded,
        PPRConverged,
        PPRDiverged,
        QueryCompleted,
        QueryFailed,
        QueryStarted,
        VecSearchCompleted,
    )
    from qortex.observability.linker import QortexEventLinker

    # ── Resource ──────────────────────────────────────────────────────
    resource = Resource.create({"service.name": config.otel_service_name})

    # ── Tracer ────────────────────────────────────────────────────────
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otel_endpoint))
    )
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("qortex")

    # ── Meter ─────────────────────────────────────────────────────────
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=config.otel_endpoint)
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter("qortex")

    # ── Instruments (names align with prometheus.py for dashboard compat) ──
    #
    # Counters: OTEL Prometheus exporter adds "_total" suffix automatically.
    # Histograms: include "_seconds" in name, no unit param (avoids double suffix).
    # Gauges: name used as-is.

    # Counters
    queries_total = meter.create_counter(
        "qortex_queries", description="Total queries"
    )
    factor_updates_total = meter.create_counter(
        "qortex_factor_updates", description="Factor update events"
    )
    edges_promoted = meter.create_counter(
        "qortex_edges_promoted", description="Lifetime edge promotions"
    )
    feedback_total = meter.create_counter(
        "qortex_feedback", description="Feedback events"
    )
    query_errors = meter.create_counter(
        "qortex_query_errors", description="Query errors"
    )
    enrichment_total = meter.create_counter(
        "qortex_enrichment", description="Enrichment runs"
    )
    enrichment_fallbacks = meter.create_counter(
        "qortex_enrichment_fallbacks", description="Enrichment fallbacks"
    )
    manifests_ingested = meter.create_counter(
        "qortex_manifests_ingested", description="Manifests ingested"
    )

    # Histograms
    query_latency = meter.create_histogram(
        "qortex_query_duration_seconds", description="Query latency"
    )
    ppr_iterations_hist = meter.create_histogram(
        "qortex_ppr_iterations", description="PPR iterations to converge"
    )
    vec_search_latency = meter.create_histogram(
        "qortex_vec_search_duration_seconds", description="Vec search latency"
    )
    enrichment_latency = meter.create_histogram(
        "qortex_enrichment_duration_seconds", description="Enrichment latency"
    )
    ingest_latency = meter.create_histogram(
        "qortex_ingest_duration_seconds", description="Ingest latency"
    )

    # Gauges (synchronous — set() on event)
    factor_mean = meter.create_gauge(
        "qortex_factor_mean", description="Mean teleportation factor"
    )
    factor_entropy = meter.create_gauge(
        "qortex_factor_entropy", description="Factor distribution entropy"
    )
    active_factors = meter.create_gauge(
        "qortex_factors_active", description="Active teleportation factors"
    )
    buffer_size = meter.create_gauge(
        "qortex_buffer_edges", description="Buffered online edges"
    )
    kg_coverage = meter.create_gauge(
        "qortex_kg_coverage", description="KG coverage ratio"
    )

    # ── Trace state ───────────────────────────────────────────────────
    _MAX_ACTIVE_SPANS = 1000
    _active_spans: dict[str, trace.Span] = {}

    # ── Query lifecycle (traces + metrics) ────────────────────────────

    @QortexEventLinker.on(QueryStarted)
    def _on_query_start(event: QueryStarted) -> None:
        # Evict oldest spans if we hit the cap (prevents unbounded growth
        # when QueryCompleted/QueryFailed events are lost).
        if len(_active_spans) >= _MAX_ACTIVE_SPANS:
            oldest_key = next(iter(_active_spans))
            stale = _active_spans.pop(oldest_key)
            stale.set_attribute("evicted", True)
            stale.end()

        span = tracer.start_span(
            "qortex.query",
            attributes={
                "query.id": event.query_id,
                "query.mode": event.mode,
                "query.top_k": event.top_k,
            },
        )
        _active_spans[event.query_id] = span

    @QortexEventLinker.on(QueryCompleted)
    def _on_query_complete(event: QueryCompleted) -> None:
        span = _active_spans.pop(event.query_id, None)
        if span:
            span.set_attribute("result_count", event.result_count)
            span.set_attribute("seed_count", event.seed_count)
            span.set_attribute("activated_nodes", event.activated_nodes)
            span.set_attribute("latency_ms", event.latency_ms)
            span.end()
        queries_total.add(1, {"mode": event.mode})
        query_latency.record(event.latency_ms / 1000)

    @QortexEventLinker.on(QueryFailed)
    def _on_query_failed(event: QueryFailed) -> None:
        span = _active_spans.pop(event.query_id, None)
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.stage", event.stage)
            span.set_attribute("error.message", event.error)
            span.end()
        query_errors.add(1, {"stage": event.stage})

    # ── PPR convergence ───────────────────────────────────────────────

    @QortexEventLinker.on(PPRConverged)
    def _on_ppr_converged(event: PPRConverged) -> None:
        ppr_iterations_hist.record(event.iterations)

    @QortexEventLinker.on(PPRDiverged)
    def _on_ppr_diverged(event: PPRDiverged) -> None:
        ppr_iterations_hist.record(event.iterations)

    # ── Teleportation factors ─────────────────────────────────────────

    @QortexEventLinker.on(FactorUpdated)
    def _on_factor_updated(event: FactorUpdated) -> None:
        factor_updates_total.add(1, {"outcome": event.outcome})

    @QortexEventLinker.on(FactorDriftSnapshot)
    def _on_factor_drift(event: FactorDriftSnapshot) -> None:
        active_factors.set(event.count)
        factor_mean.set(event.mean)
        factor_entropy.set(event.entropy)

    # ── Edge promotion ────────────────────────────────────────────────

    @QortexEventLinker.on(OnlineEdgeRecorded)
    def _on_edge_recorded(event: OnlineEdgeRecorded) -> None:
        buffer_size.set(event.buffer_size)

    @QortexEventLinker.on(EdgePromoted)
    def _on_edge_promoted(event: EdgePromoted) -> None:
        edges_promoted.add(1)

    @QortexEventLinker.on(BufferFlushed)
    def _on_buffer_flushed(event: BufferFlushed) -> None:
        if event.kg_coverage is not None:
            kg_coverage.set(event.kg_coverage)

    # ── Retrieval ─────────────────────────────────────────────────────

    @QortexEventLinker.on(VecSearchCompleted)
    def _on_vec_search(event: VecSearchCompleted) -> None:
        vec_search_latency.record(event.latency_ms / 1000)

    @QortexEventLinker.on(FeedbackReceived)
    def _on_feedback(event: FeedbackReceived) -> None:
        if event.accepted > 0:
            feedback_total.add(event.accepted, {"outcome": "accepted"})
        if event.rejected > 0:
            feedback_total.add(event.rejected, {"outcome": "rejected"})
        if event.partial > 0:
            feedback_total.add(event.partial, {"outcome": "partial"})

    # ── Enrichment ────────────────────────────────────────────────────

    @QortexEventLinker.on(EnrichmentCompleted)
    def _on_enrichment(event: EnrichmentCompleted) -> None:
        enrichment_total.add(1, {"backend_type": event.backend_type})
        enrichment_latency.record(event.latency_ms / 1000)

    @QortexEventLinker.on(EnrichmentFallback)
    def _on_enrichment_fallback(event: EnrichmentFallback) -> None:
        enrichment_fallbacks.add(1)

    # ── Ingestion ─────────────────────────────────────────────────────

    @QortexEventLinker.on(ManifestIngested)
    def _on_manifest(event: ManifestIngested) -> None:
        manifests_ingested.add(1, {"domain": event.domain})
        ingest_latency.record(event.latency_ms / 1000)
