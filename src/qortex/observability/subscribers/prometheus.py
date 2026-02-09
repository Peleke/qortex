"""Prometheus subscriber: counters, gauges, histograms from events.

Requires qortex[observability] (prometheus-client).
Starts an HTTP server on the configured port for /metrics scraping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qortex.observability.config import ObservabilityConfig


def register_prometheus_subscriber(config: ObservabilityConfig) -> None:
    """Register Prometheus metric subscribers and start /metrics server."""
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

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
        VecSearchCompleted,
    )
    from qortex.observability.linker import QortexEventLinker

    # Start metrics endpoint
    start_http_server(config.prometheus_port)

    # Instruments
    queries_total = Counter(
        "qortex_queries_total", "Total queries", ["mode"]
    )
    query_latency = Histogram(
        "qortex_query_duration_seconds",
        "Query latency",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    ppr_iterations_hist = Histogram(
        "qortex_ppr_iterations",
        "PPR iterations to converge",
        buckets=[1, 5, 10, 20, 50, 100],
    )
    factor_mean = Gauge("qortex_factor_mean", "Mean teleportation factor")
    factor_entropy = Gauge("qortex_factor_entropy", "Factor distribution entropy")
    active_factors = Gauge("qortex_factors_active", "Active teleportation factors")
    buffer_size = Gauge("qortex_buffer_edges", "Buffered online edges")
    edges_promoted = Counter(
        "qortex_edges_promoted_total", "Lifetime edge promotions"
    )
    kg_coverage = Gauge("qortex_kg_coverage", "KG coverage ratio")
    feedback_total = Counter(
        "qortex_feedback_total", "Feedback events", ["outcome"]
    )
    factor_updates_total = Counter(
        "qortex_factor_updates_total", "Factor update events", ["outcome"]
    )
    vec_search_latency = Histogram(
        "qortex_vec_search_duration_seconds",
        "Vec search latency",
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    query_errors = Counter(
        "qortex_query_errors_total", "Query errors", ["stage"]
    )
    enrichment_total = Counter(
        "qortex_enrichment_total", "Enrichment runs", ["backend_type"]
    )
    enrichment_latency = Histogram(
        "qortex_enrichment_duration_seconds",
        "Enrichment latency",
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    enrichment_fallbacks = Counter(
        "qortex_enrichment_fallbacks_total", "Enrichment fallbacks"
    )
    manifests_ingested = Counter(
        "qortex_manifests_ingested_total", "Manifests ingested", ["domain"]
    )
    ingest_latency = Histogram(
        "qortex_ingest_duration_seconds",
        "Ingest latency",
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    )

    @QortexEventLinker.on(QueryCompleted)
    def _prom_query(event: QueryCompleted) -> None:
        queries_total.labels(mode=event.mode).inc()
        query_latency.observe(event.latency_ms / 1000)

    @QortexEventLinker.on(PPRConverged)
    def _prom_ppr_converged(event: PPRConverged) -> None:
        ppr_iterations_hist.observe(event.iterations)

    @QortexEventLinker.on(PPRDiverged)
    def _prom_ppr_diverged(event: PPRDiverged) -> None:
        ppr_iterations_hist.observe(event.iterations)

    @QortexEventLinker.on(FactorUpdated)
    def _prom_factor_updated(event: FactorUpdated) -> None:
        factor_updates_total.labels(outcome=event.outcome).inc()

    @QortexEventLinker.on(FactorDriftSnapshot)
    def _prom_factor_drift(event: FactorDriftSnapshot) -> None:
        active_factors.set(event.count)
        factor_mean.set(event.mean)
        factor_entropy.set(event.entropy)

    @QortexEventLinker.on(OnlineEdgeRecorded)
    def _prom_edge_recorded(event: OnlineEdgeRecorded) -> None:
        buffer_size.set(event.buffer_size)

    @QortexEventLinker.on(EdgePromoted)
    def _prom_edge_promoted(event: EdgePromoted) -> None:
        edges_promoted.inc()

    @QortexEventLinker.on(BufferFlushed)
    def _prom_buffer_flushed(event: BufferFlushed) -> None:
        if event.kg_coverage is not None:
            kg_coverage.set(event.kg_coverage)

    @QortexEventLinker.on(FeedbackReceived)
    def _prom_feedback(event: FeedbackReceived) -> None:
        if event.accepted > 0:
            feedback_total.labels(outcome="accepted").inc(event.accepted)
        if event.rejected > 0:
            feedback_total.labels(outcome="rejected").inc(event.rejected)
        if event.partial > 0:
            feedback_total.labels(outcome="partial").inc(event.partial)

    @QortexEventLinker.on(VecSearchCompleted)
    def _prom_vec_search(event: VecSearchCompleted) -> None:
        vec_search_latency.observe(event.latency_ms / 1000)

    @QortexEventLinker.on(QueryFailed)
    def _prom_query_failed(event: QueryFailed) -> None:
        query_errors.labels(stage=event.stage).inc()

    @QortexEventLinker.on(EnrichmentCompleted)
    def _prom_enrichment(event: EnrichmentCompleted) -> None:
        enrichment_total.labels(backend_type=event.backend_type).inc()
        enrichment_latency.observe(event.latency_ms / 1000)

    @QortexEventLinker.on(EnrichmentFallback)
    def _prom_enrichment_fallback(event: EnrichmentFallback) -> None:
        enrichment_fallbacks.inc()

    @QortexEventLinker.on(ManifestIngested)
    def _prom_manifest(event: ManifestIngested) -> None:
        manifests_ingested.labels(domain=event.domain).inc()
        ingest_latency.observe(event.latency_ms / 1000)
