"""Routes all events to structured log lines via the configured LogFormatter.

Always-on subscriber. Called by emitter.configure() on every startup.
Uses get_logger() from the logging module -- works with structlog, stdlib,
or any registered LogFormatter. No direct structlog dependency.
"""

from __future__ import annotations

from dataclasses import asdict

from qortex_observe.logging import get_logger
from qortex_observe.events import (
    BufferFlushed,
    CreditPropagated,
    EdgePromoted,
    EnrichmentCompleted,
    EnrichmentFallback,
    FactorDriftSnapshot,
    FactorsLoaded,
    FactorsPersisted,
    FactorUpdated,
    FeedbackReceived,
    InteroceptionShutdown,
    InteroceptionStarted,
    KGCoverageComputed,
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
    ManifestIngested,
    OnlineEdgeRecorded,
    OnlineEdgesGenerated,
    PPRConverged,
    PPRDiverged,
    PPRStarted,
    QueryCompleted,
    QueryFailed,
    QueryStarted,
    VecSearchCompleted,
)
from qortex_observe.linker import QortexEventLinker


def _get_logger():
    """Lazy logger -- always reflects the active formatter, not stale import-time state."""
    return get_logger("qortex.events")


def _to_dict(event: object) -> dict:
    """Convert frozen dataclass to dict for structlog."""
    return asdict(event)  # type: ignore[arg-type]


def register_structlog_subscriber() -> None:
    """Register structlog handlers for all events on QortexEventLinker.

    Safe to call multiple times -- QortexEventLinker deduplicates handlers.
    Called by emitter.configure() on every startup.
    """

    # Query lifecycle
    @QortexEventLinker.on(QueryStarted)
    def _log_query_started(event: QueryStarted) -> None:
        _get_logger().info("query.started", **_to_dict(event))

    @QortexEventLinker.on(QueryCompleted)
    def _log_query_completed(event: QueryCompleted) -> None:
        _get_logger().info("query.completed", **_to_dict(event))

    @QortexEventLinker.on(QueryFailed)
    def _log_query_failed(event: QueryFailed) -> None:
        _get_logger().error("query.failed", **_to_dict(event))

    # PPR convergence
    @QortexEventLinker.on(PPRStarted)
    def _log_ppr_started(event: PPRStarted) -> None:
        _get_logger().debug("ppr.started", **_to_dict(event))

    @QortexEventLinker.on(PPRConverged)
    def _log_ppr_converged(event: PPRConverged) -> None:
        _get_logger().info("ppr.converged", **_to_dict(event))

    @QortexEventLinker.on(PPRDiverged)
    def _log_ppr_diverged(event: PPRDiverged) -> None:
        _get_logger().warning("ppr.diverged", **_to_dict(event))

    # Teleportation factors
    @QortexEventLinker.on(FactorUpdated)
    def _log_factor_updated(event: FactorUpdated) -> None:
        _get_logger().debug("factor.updated", **_to_dict(event))

    @QortexEventLinker.on(FactorsPersisted)
    def _log_factors_persisted(event: FactorsPersisted) -> None:
        _get_logger().info("factors.persisted", **_to_dict(event))

    @QortexEventLinker.on(FactorsLoaded)
    def _log_factors_loaded(event: FactorsLoaded) -> None:
        _get_logger().info("factors.loaded", **_to_dict(event))

    @QortexEventLinker.on(FactorDriftSnapshot)
    def _log_factor_drift(event: FactorDriftSnapshot) -> None:
        _get_logger().info("factor.drift", **_to_dict(event))

    # Edge promotion
    @QortexEventLinker.on(OnlineEdgeRecorded)
    def _log_online_edge(event: OnlineEdgeRecorded) -> None:
        _get_logger().debug("edge.recorded", **_to_dict(event))

    @QortexEventLinker.on(EdgePromoted)
    def _log_edge_promoted(event: EdgePromoted) -> None:
        _get_logger().info("edge.promoted", **_to_dict(event))

    @QortexEventLinker.on(BufferFlushed)
    def _log_buffer_flushed(event: BufferFlushed) -> None:
        _get_logger().info("buffer.flushed", **_to_dict(event))

    # Retrieval
    @QortexEventLinker.on(VecSearchCompleted)
    def _log_vec_search(event: VecSearchCompleted) -> None:
        _get_logger().debug("vec.search.completed", **_to_dict(event))

    @QortexEventLinker.on(OnlineEdgesGenerated)
    def _log_online_edges(event: OnlineEdgesGenerated) -> None:
        _get_logger().debug("online.edges.generated", **_to_dict(event))

    @QortexEventLinker.on(KGCoverageComputed)
    def _log_kg_coverage(event: KGCoverageComputed) -> None:
        _get_logger().info("kg.coverage", **_to_dict(event))

    @QortexEventLinker.on(FeedbackReceived)
    def _log_feedback(event: FeedbackReceived) -> None:
        _get_logger().info("feedback.received", **_to_dict(event))

    # Lifecycle
    @QortexEventLinker.on(InteroceptionStarted)
    def _log_interoception_started(event: InteroceptionStarted) -> None:
        _get_logger().info("interoception.started", **_to_dict(event))

    @QortexEventLinker.on(InteroceptionShutdown)
    def _log_interoception_shutdown(event: InteroceptionShutdown) -> None:
        _get_logger().info("interoception.shutdown", **_to_dict(event))

    # Enrichment
    @QortexEventLinker.on(EnrichmentCompleted)
    def _log_enrichment_completed(event: EnrichmentCompleted) -> None:
        _get_logger().info("enrichment.completed", **_to_dict(event))

    @QortexEventLinker.on(EnrichmentFallback)
    def _log_enrichment_fallback(event: EnrichmentFallback) -> None:
        _get_logger().warning("enrichment.fallback", **_to_dict(event))

    # Ingestion
    @QortexEventLinker.on(ManifestIngested)
    def _log_manifest_ingested(event: ManifestIngested) -> None:
        _get_logger().info("manifest.ingested", **_to_dict(event))

    # Learning
    @QortexEventLinker.on(LearningSelectionMade)
    def _log_learning_selection(event: LearningSelectionMade) -> None:
        _get_logger().info("learning.selection", **_to_dict(event))

    @QortexEventLinker.on(LearningObservationRecorded)
    def _log_learning_observation(event: LearningObservationRecorded) -> None:
        _get_logger().info("learning.observation", **_to_dict(event))

    @QortexEventLinker.on(LearningPosteriorUpdated)
    def _log_learning_posterior(event: LearningPosteriorUpdated) -> None:
        _get_logger().debug("learning.posterior", **_to_dict(event))

    # Credit Propagation
    @QortexEventLinker.on(CreditPropagated)
    def _log_credit_propagated(event: CreditPropagated) -> None:
        _get_logger().info("credit.propagated", **_to_dict(event))
