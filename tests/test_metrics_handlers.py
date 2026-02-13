"""Tests for unified event-to-metric handler wiring.

Covers: HANDLER-1 through HANDLER-13

Strategy: Create real OTel instruments with InMemoryMetricReader,
register handlers, invoke handlers directly, read back metric data.
"""

from __future__ import annotations

import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from qortex_observe.events import (  # noqa: E402
    BufferFlushed,
    CreditPropagated,
    EdgePromoted,
    EnrichmentCompleted,
    EnrichmentFallback,
    FactorDriftSnapshot,
    FactorUpdated,
    FeedbackReceived,
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
    ManifestIngested,
    OnlineEdgeRecorded,
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
from qortex_observe.metrics_factory import create_instruments, create_views  # noqa: E402
from qortex_observe.metrics_handlers import register_metric_handlers  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_linker():
    """Reset QortexEventLinker between tests to avoid duplicate handler registration."""
    from qortex_observe.linker import QortexEventLinker

    QortexEventLinker.remove_all()
    yield
    QortexEventLinker.remove_all()


@pytest.fixture()
def instruments_and_reader():
    """Create real OTel instruments from schema with InMemoryMetricReader."""
    reader = InMemoryMetricReader()
    views = create_views()
    provider = MeterProvider(metric_readers=[reader], views=views)
    meter = provider.get_meter("test")
    instruments = create_instruments(meter)
    register_metric_handlers(instruments)
    return instruments, reader


def _collect(reader: InMemoryMetricReader) -> dict:
    """Collect all metrics from reader into {name: metric} dict.

    InMemoryMetricReader consumes data on each call, so collect once.
    """
    data = reader.get_metrics_data()
    if data is None:
        return {}
    result = {}
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                result[metric.name] = metric
    return result


def _get_value(metrics: dict, name: str) -> float | None:
    """Get the first data point value (counter sum or gauge)."""
    metric = metrics.get(name)
    if metric is None:
        return None
    for dp in metric.data.data_points:
        return dp.value
    return None


def _get_value_with_attrs(metrics: dict, name: str, attrs: dict) -> float | None:
    """Get value filtered by attributes."""
    metric = metrics.get(name)
    if metric is None:
        return None
    for dp in metric.data.data_points:
        if all(dp.attributes.get(k) == v for k, v in attrs.items()):
            return dp.value
    return None


def _get_hist_sum(metrics: dict, name: str) -> float | None:
    """Get histogram sum."""
    metric = metrics.get(name)
    if metric is None:
        return None
    for dp in metric.data.data_points:
        return dp.sum
    return None


def _get_hist_count(metrics: dict, name: str) -> int | None:
    """Get histogram count."""
    metric = metrics.get(name)
    if metric is None:
        return None
    for dp in metric.data.data_points:
        return dp.count
    return None


def _emit(event) -> None:
    """Invoke all registered handlers for an event synchronously."""
    import asyncio

    from qortex_observe.linker import QortexEventLinker

    event_name = type(event).__name__
    registry = QortexEventLinker.get_registry()
    subscribers = registry.get(event_name, set())
    for sub in subscribers:
        asyncio.run(sub.execute(event))


class TestQueryHandlers:
    """HANDLER-1, HANDLER-2: Query lifecycle metric routing."""

    def test_query_completed_increments_counter(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = QueryCompleted(
            query_id="q1",
            latency_ms=42.0,
            seed_count=3,
            result_count=5,
            activated_nodes=8,
            mode="graph",
            timestamp="ts",
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_queries", {"mode": "graph"}) == 1

    def test_query_completed_records_latency(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = QueryCompleted(
            query_id="q1",
            latency_ms=50.0,
            seed_count=3,
            result_count=5,
            activated_nodes=8,
            mode="vec",
            timestamp="ts",
        )
        _emit(event)
        m = _collect(reader)
        assert _get_hist_sum(m, "qortex_query_duration_seconds") == pytest.approx(0.05)

    def test_query_failed_increments_error_counter(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = QueryFailed(query_id="q1", error="boom", stage="embedding", timestamp="ts")
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_query_errors", {"stage": "embedding"}) == 1


class TestPPRHandlers:
    """HANDLER-3: PPR metric routing."""

    def test_ppr_started_increments(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = PPRStarted(
            query_id=None, node_count=100, seed_count=5, damping_factor=0.85, extra_edge_count=0
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value(m, "qortex_ppr_started") == 1

    def test_ppr_converged_records_iterations(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = PPRConverged(
            query_id=None,
            iterations=15,
            final_diff=1e-7,
            node_count=100,
            nonzero_scores=42,
            latency_ms=3.0,
        )
        _emit(event)
        m = _collect(reader)
        assert _get_hist_count(m, "qortex_ppr_iterations") == 1
        assert _get_hist_sum(m, "qortex_ppr_iterations") == 15

    def test_ppr_diverged_records_iterations(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = PPRDiverged(query_id=None, iterations=100, final_diff=0.5, node_count=50)
        _emit(event)
        m = _collect(reader)
        assert _get_hist_sum(m, "qortex_ppr_iterations") == 100


class TestFactorHandlers:
    """HANDLER-4, HANDLER-5: Factor metric routing."""

    def test_factor_updated_increments_by_outcome(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = FactorUpdated(
            node_id="n1",
            query_id="q1",
            outcome="accepted",
            old_factor=1.0,
            new_factor=1.1,
            delta=0.1,
            clamped=False,
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_factor_updates", {"outcome": "accepted"}) == 1

    def test_factor_drift_sets_gauges(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = FactorDriftSnapshot(
            count=5, mean=1.2, min_val=0.5, max_val=3.0, boosted=3, penalized=2, entropy=0.8
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value(m, "qortex_factors_active") == 5
        assert _get_value(m, "qortex_factor_mean") == pytest.approx(1.2)
        assert _get_value(m, "qortex_factor_entropy") == pytest.approx(0.8)


class TestEdgeHandlers:
    """HANDLER-6: Edge promotion metric routing."""

    def test_online_edge_sets_buffer_size(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = OnlineEdgeRecorded(
            source_id="a", target_id="b", score=0.9, hit_count=1, buffer_size=42
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value(m, "qortex_buffer_edges") == 42

    def test_edge_promoted_increments(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = EdgePromoted(source_id="a", target_id="b", hit_count=3, avg_score=0.85)
        _emit(event)
        m = _collect(reader)
        assert _get_value(m, "qortex_edges_promoted") == 1

    def test_buffer_flushed_sets_coverage(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = BufferFlushed(
            promoted=2, remaining=8, total_promoted_lifetime=10, kg_coverage=0.75, timestamp="ts"
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value(m, "qortex_kg_coverage") == pytest.approx(0.75)

    def test_buffer_flushed_skips_none_coverage(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = BufferFlushed(
            promoted=0, remaining=0, total_promoted_lifetime=0, kg_coverage=None, timestamp="ts"
        )
        _emit(event)
        m = _collect(reader)
        # Gauge never set, so either absent or no data points
        metric = m.get("qortex_kg_coverage")
        if metric is not None:
            assert len(metric.data.data_points) == 0


class TestFeedbackHandlers:
    """HANDLER-7: Feedback metric routing."""

    def test_feedback_routes_by_outcome(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = FeedbackReceived(
            query_id="q1", outcomes=3, accepted=2, rejected=1, partial=0, source="mcp"
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_feedback", {"outcome": "accepted"}) == 2
        assert _get_value_with_attrs(m, "qortex_feedback", {"outcome": "rejected"}) == 1

    def test_feedback_skips_zero_outcomes(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = FeedbackReceived(
            query_id="q1", outcomes=0, accepted=0, rejected=0, partial=0, source="mcp"
        )
        _emit(event)
        m = _collect(reader)
        metric = m.get("qortex_feedback")
        if metric is not None:
            assert len(metric.data.data_points) == 0


class TestVecSearchHandlers:
    """HANDLER-8: Vec search metric routing."""

    def test_vec_search_records_latency(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = VecSearchCompleted(query_id="q1", candidates=30, fetch_k=60, latency_ms=50.0)
        _emit(event)
        m = _collect(reader)
        assert _get_hist_sum(m, "qortex_vec_search_duration_seconds") == pytest.approx(0.05)

    def test_vec_search_results_sets_gauges(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = VecSearchResults(
            candidates=15, top_score=0.95, score_spread=0.3, latency_ms=10.0, index_type="numpy"
        )
        _emit(event)
        m = _collect(reader)
        assert _get_hist_count(m, "qortex_vec_search_candidates") == 1
        assert _get_value(m, "qortex_vec_search_top_score") == pytest.approx(0.95)
        assert _get_value(m, "qortex_vec_search_score_spread") == pytest.approx(0.3)

    def test_vec_seed_yield_sets_gauge(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = VecSeedYield(
            query_id="q1", vec_candidates=20, seeds_after_filter=16, yield_ratio=0.8
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value(m, "qortex_vec_seed_yield") == pytest.approx(0.8)


class TestVecIndexHandlers:
    """HANDLER-9: Vec index metric routing."""

    def test_vec_add_increments_and_records(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = VecIndexUpdated(count_added=10, total_size=100, latency_ms=25.0, index_type="numpy")
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_vec_add", {"index_type": "numpy"}) == 1
        assert _get_hist_sum(m, "qortex_vec_add_duration_seconds") == pytest.approx(0.025)
        assert _get_value(m, "qortex_vec_index_size") == 100


class TestEnrichmentHandlers:
    """HANDLER-10: Enrichment metric routing."""

    def test_enrichment_completed_increments_and_records(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = EnrichmentCompleted(
            rule_count=5, succeeded=4, failed=1, backend_type="template", latency_ms=1500.0
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_enrichment", {"backend_type": "template"}) == 1
        assert _get_hist_sum(m, "qortex_enrichment_duration_seconds") == pytest.approx(1.5)

    def test_enrichment_fallback_increments(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = EnrichmentFallback(reason="backend_exception", rule_count=2)
        _emit(event)
        m = _collect(reader)
        assert _get_value(m, "qortex_enrichment_fallbacks") == 1


class TestIngestionHandlers:
    """HANDLER-11: Ingestion metric routing."""

    def test_manifest_ingested_increments_and_records(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = ManifestIngested(
            domain="test",
            node_count=10,
            edge_count=5,
            rule_count=3,
            source_id="src1",
            latency_ms=250.0,
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_manifests_ingested", {"domain": "test"}) == 1
        assert _get_hist_sum(m, "qortex_ingest_duration_seconds") == pytest.approx(0.25)


class TestLearningHandlers:
    """HANDLER-12: Learning metric routing."""

    def test_learning_selection_increments(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = LearningSelectionMade(
            learner="test",
            selected_count=3,
            excluded_count=2,
            is_baseline=False,
            token_budget=1000,
            used_tokens=750,
        )
        _emit(event)
        m = _collect(reader)
        assert (
            _get_value_with_attrs(
                m, "qortex_learning_selections", {"learner": "test", "baseline": "False"}
            )
            == 1
        )
        assert _get_hist_count(m, "qortex_learning_token_budget_used") == 1

    def test_learning_selection_skips_zero_budget(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = LearningSelectionMade(
            learner="test",
            selected_count=1,
            excluded_count=0,
            is_baseline=True,
            token_budget=0,
            used_tokens=0,
        )
        _emit(event)
        m = _collect(reader)
        count = _get_hist_count(m, "qortex_learning_token_budget_used")
        assert count is None or count == 0

    def test_learning_observation_increments(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = LearningObservationRecorded(
            learner="test", arm_id="arm:a", reward=1.0, outcome="accepted", context_hash="abc"
        )
        _emit(event)
        m = _collect(reader)
        assert (
            _get_value_with_attrs(
                m, "qortex_learning_observations", {"learner": "test", "outcome": "accepted"}
            )
            == 1
        )

    def test_learning_posterior_sets_gauge_and_increments(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = LearningPosteriorUpdated(
            learner="test", arm_id="arm:a", alpha=2.0, beta=1.0, pulls=1, mean=0.667
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(
            m, "qortex_learning_posterior_mean", {"learner": "test", "arm_id": "arm:a"}
        ) == pytest.approx(0.667)
        assert (
            _get_value_with_attrs(
                m, "qortex_learning_arm_pulls", {"learner": "test", "arm_id": "arm:a"}
            )
            == 1
        )


class TestCreditHandlers:
    """HANDLER-13: Credit propagation metric routing."""

    def test_credit_propagated_increments(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = CreditPropagated(
            query_id="q1",
            concept_count=25,
            direct_count=10,
            ancestor_count=15,
            total_alpha_delta=1.5,
            total_beta_delta=0.3,
            learner="credit",
        )
        _emit(event)
        m = _collect(reader)
        assert _get_value_with_attrs(m, "qortex_credit_propagations", {"learner": "credit"}) == 1
        assert _get_hist_count(m, "qortex_credit_concepts_per_propagation") == 1
        assert _get_value(m, "qortex_credit_alpha_delta") == pytest.approx(1.5)
        assert _get_value(m, "qortex_credit_beta_delta") == pytest.approx(0.3)

    def test_credit_skips_zero_deltas(self, instruments_and_reader):
        _, reader = instruments_and_reader
        event = CreditPropagated(
            query_id="q1",
            concept_count=0,
            direct_count=0,
            ancestor_count=0,
            total_alpha_delta=0.0,
            total_beta_delta=0.0,
            learner="credit",
        )
        _emit(event)
        m = _collect(reader)
        alpha_val = _get_value(m, "qortex_credit_alpha_delta")
        beta_val = _get_value(m, "qortex_credit_beta_delta")
        assert alpha_val is None or alpha_val == 0
        assert beta_val is None or beta_val == 0
