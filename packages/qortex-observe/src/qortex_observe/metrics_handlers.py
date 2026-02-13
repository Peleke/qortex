"""Unified event-to-metric handlers: single set replacing prometheus.py + otel.py metrics.

Each handler looks up its instruments from the dict returned by create_instruments().
Events are routed via QortexEventLinker decorators.

This module is imported ONCE during configure(). The handlers close over the
instruments dict, so no global state beyond the linker registration.
"""

from __future__ import annotations

from typing import Any

from qortex_observe.events import (
    BufferFlushed,
    CarbonTracked,
    CreditPropagated,
    EdgePromoted,
    EnrichmentCompleted,
    EnrichmentFallback,
    FactorDriftSnapshot,
    FactorUpdated,
    FeedbackReceived,
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
    VecIndexUpdated,
    VecSearchCompleted,
    VecSearchResults,
    VecSeedYield,
)
from qortex_observe.linker import QortexEventLinker


def register_metric_handlers(instruments: dict[str, Any]) -> None:
    """Register all event-to-metric handlers using QortexEventLinker.

    Called once from configure() after instruments are created.
    Handlers close over the instruments dict.
    """

    # ── Query lifecycle ──────────────────────────────────────────

    @QortexEventLinker.on(QueryCompleted)
    def _on_query_complete(event: QueryCompleted) -> None:
        instruments["qortex_queries"].add(1, {"mode": event.mode})
        instruments["qortex_query_duration_seconds"].record(event.latency_ms / 1000)
        if event.overhead_seconds is not None:
            instruments["qortex_query_overhead_seconds"].record(event.overhead_seconds)

    @QortexEventLinker.on(QueryFailed)
    def _on_query_failed(event: QueryFailed) -> None:
        instruments["qortex_query_errors"].add(1, {"stage": event.stage})

    # ── PPR ──────────────────────────────────────────────────────

    @QortexEventLinker.on(PPRStarted)
    def _on_ppr_started(event: PPRStarted) -> None:
        instruments["qortex_ppr_started"].add(1)

    @QortexEventLinker.on(PPRConverged)
    def _on_ppr_converged(event: PPRConverged) -> None:
        instruments["qortex_ppr_iterations"].record(event.iterations)

    @QortexEventLinker.on(PPRDiverged)
    def _on_ppr_diverged(event: PPRDiverged) -> None:
        instruments["qortex_ppr_iterations"].record(event.iterations)

    # ── Factors ──────────────────────────────────────────────────

    @QortexEventLinker.on(FactorUpdated)
    def _on_factor_updated(event: FactorUpdated) -> None:
        instruments["qortex_factor_updates"].add(1, {"outcome": event.outcome})

    @QortexEventLinker.on(FactorDriftSnapshot)
    def _on_factor_drift(event: FactorDriftSnapshot) -> None:
        instruments["qortex_factors_active"].set(event.count)
        instruments["qortex_factor_mean"].set(event.mean)
        instruments["qortex_factor_entropy"].set(event.entropy)

    # ── Edge promotion ───────────────────────────────────────────

    @QortexEventLinker.on(OnlineEdgeRecorded)
    def _on_edge_recorded(event: OnlineEdgeRecorded) -> None:
        instruments["qortex_buffer_edges"].set(event.buffer_size)

    @QortexEventLinker.on(EdgePromoted)
    def _on_edge_promoted(event: EdgePromoted) -> None:
        instruments["qortex_edges_promoted"].add(1)

    @QortexEventLinker.on(BufferFlushed)
    def _on_buffer_flushed(event: BufferFlushed) -> None:
        if event.kg_coverage is not None:
            instruments["qortex_kg_coverage"].set(event.kg_coverage)

    # ── Retrieval ────────────────────────────────────────────────

    @QortexEventLinker.on(VecSearchCompleted)
    def _on_vec_search(event: VecSearchCompleted) -> None:
        instruments["qortex_vec_search_duration_seconds"].record(event.latency_ms / 1000)

    @QortexEventLinker.on(OnlineEdgesGenerated)
    def _on_online_edges(event: OnlineEdgesGenerated) -> None:
        instruments["qortex_online_edges_generated"].add(1)
        instruments["qortex_online_edge_count"].set(event.edge_count)

    @QortexEventLinker.on(KGCoverageComputed)
    def _on_kg_coverage(event: KGCoverageComputed) -> None:
        instruments["qortex_kg_coverage"].set(event.coverage)

    @QortexEventLinker.on(FeedbackReceived)
    def _on_feedback(event: FeedbackReceived) -> None:
        if event.accepted > 0:
            instruments["qortex_feedback"].add(event.accepted, {"outcome": "accepted"})
        if event.rejected > 0:
            instruments["qortex_feedback"].add(event.rejected, {"outcome": "rejected"})
        if event.partial > 0:
            instruments["qortex_feedback"].add(event.partial, {"outcome": "partial"})

    # ── Vec index ────────────────────────────────────────────────

    @QortexEventLinker.on(VecIndexUpdated)
    def _on_vec_index_updated(event: VecIndexUpdated) -> None:
        instruments["qortex_vec_add"].add(1, {"index_type": event.index_type})
        instruments["qortex_vec_add_duration_seconds"].record(event.latency_ms / 1000)
        instruments["qortex_vec_index_size"].set(event.total_size)

    @QortexEventLinker.on(VecSearchResults)
    def _on_vec_search_results(event: VecSearchResults) -> None:
        instruments["qortex_vec_search_candidates"].record(event.candidates)
        instruments["qortex_vec_search_top_score"].set(event.top_score)
        instruments["qortex_vec_search_score_spread"].set(event.score_spread)

    @QortexEventLinker.on(VecSeedYield)
    def _on_vec_seed_yield(event: VecSeedYield) -> None:
        instruments["qortex_vec_seed_yield"].set(event.yield_ratio)

    # ── Enrichment ───────────────────────────────────────────────

    @QortexEventLinker.on(EnrichmentCompleted)
    def _on_enrichment(event: EnrichmentCompleted) -> None:
        instruments["qortex_enrichment"].add(1, {"backend_type": event.backend_type})
        instruments["qortex_enrichment_duration_seconds"].record(event.latency_ms / 1000)

    @QortexEventLinker.on(EnrichmentFallback)
    def _on_enrichment_fallback(event: EnrichmentFallback) -> None:
        instruments["qortex_enrichment_fallbacks"].add(1)

    # ── Ingestion ────────────────────────────────────────────────

    @QortexEventLinker.on(ManifestIngested)
    def _on_manifest(event: ManifestIngested) -> None:
        instruments["qortex_manifests_ingested"].add(1, {"domain": event.domain})
        instruments["qortex_ingest_duration_seconds"].record(event.latency_ms / 1000)

    # ── Learning ─────────────────────────────────────────────────

    @QortexEventLinker.on(LearningSelectionMade)
    def _on_learning_selection(event: LearningSelectionMade) -> None:
        instruments["qortex_learning_selections"].add(1, {
            "learner": event.learner,
            "baseline": str(event.is_baseline),
        })
        if event.token_budget > 0:
            instruments["qortex_learning_token_budget_used"].record(event.used_tokens)

    @QortexEventLinker.on(LearningObservationRecorded)
    def _on_learning_observation(event: LearningObservationRecorded) -> None:
        instruments["qortex_learning_observations"].add(1, {
            "learner": event.learner,
            "outcome": event.outcome,
        })

    @QortexEventLinker.on(LearningPosteriorUpdated)
    def _on_learning_posterior(event: LearningPosteriorUpdated) -> None:
        instruments["qortex_learning_posterior_mean"].set(event.mean, {
            "learner": event.learner,
            "arm_id": event.arm_id,
        })
        instruments["qortex_learning_arm_pulls"].add(1, {
            "learner": event.learner,
            "arm_id": event.arm_id,
        })

    # ── Credit propagation ───────────────────────────────────────

    @QortexEventLinker.on(CreditPropagated)
    def _on_credit_propagated(event: CreditPropagated) -> None:
        instruments["qortex_credit_propagations"].add(1, {"learner": event.learner})
        instruments["qortex_credit_concepts_per_propagation"].record(event.concept_count)
        if event.total_alpha_delta > 0:
            instruments["qortex_credit_alpha_delta"].add(event.total_alpha_delta)
        if event.total_beta_delta > 0:
            instruments["qortex_credit_beta_delta"].add(event.total_beta_delta)

    # ── Carbon accounting ─────────────────────────────────────────

    @QortexEventLinker.on(CarbonTracked)
    def _on_carbon_tracked(event: CarbonTracked) -> None:
        labels = {"provider": event.provider, "model": event.model}
        instruments["qortex_carbon_co2_grams"].add(event.total_co2_grams, labels)
        instruments["qortex_carbon_water_ml"].add(event.water_ml, labels)
        total_tokens = event.input_tokens + event.output_tokens + event.cache_read_tokens
        instruments["qortex_carbon_tokens"].add(total_tokens, labels)
        instruments["qortex_carbon_confidence"].set(event.confidence)
