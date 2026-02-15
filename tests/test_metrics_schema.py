"""Tests for the declarative metric schema: completeness, validity, invariants.

Covers: SCHEMA-1 through SCHEMA-6
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from qortex.observe.metrics_schema import METRICS, MetricType


class TestMetricsSchemaCompleteness:
    """SCHEMA-1: The METRICS tuple contains all expected metrics."""

    def test_metrics_count(self):
        """Exactly 42 metrics defined (38 core + 4 carbon)."""
        assert len(METRICS) == 48

    def test_all_metric_types_represented(self):
        types = {m.type for m in METRICS}
        assert MetricType.COUNTER in types
        assert MetricType.HISTOGRAM in types
        assert MetricType.GAUGE in types

    def test_expected_query_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_queries" in names
        assert "qortex_query_duration_seconds" in names
        assert "qortex_query_errors" in names

    def test_expected_ppr_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_ppr_started" in names
        assert "qortex_ppr_iterations" in names

    def test_expected_factor_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_factor_updates" in names
        assert "qortex_factor_mean" in names
        assert "qortex_factor_entropy" in names
        assert "qortex_factors_active" in names

    def test_expected_edge_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_buffer_edges" in names
        assert "qortex_edges_promoted" in names
        assert "qortex_kg_coverage" in names
        assert "qortex_online_edges_generated" in names
        assert "qortex_online_edge_count" in names

    def test_expected_vec_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_vec_search_duration_seconds" in names
        assert "qortex_vec_search_candidates" in names
        assert "qortex_vec_search_top_score" in names
        assert "qortex_vec_search_score_spread" in names
        assert "qortex_vec_seed_yield" in names
        assert "qortex_vec_add" in names
        assert "qortex_vec_add_duration_seconds" in names
        assert "qortex_vec_index_size" in names

    def test_expected_enrichment_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_enrichment" in names
        assert "qortex_enrichment_duration_seconds" in names
        assert "qortex_enrichment_fallbacks" in names

    def test_expected_ingestion_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_manifests_ingested" in names
        assert "qortex_ingest_duration_seconds" in names

    def test_expected_learning_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_learning_selections" in names
        assert "qortex_learning_observations" in names
        assert "qortex_learning_posterior_mean" in names
        assert "qortex_learning_arm_pulls" in names
        assert "qortex_learning_token_budget_used" in names

    def test_expected_credit_metrics_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_credit_propagations" in names
        assert "qortex_credit_concepts_per_propagation" in names
        assert "qortex_credit_alpha_delta" in names
        assert "qortex_credit_beta_delta" in names

    def test_expected_feedback_metric_present(self):
        names = {m.name for m in METRICS}
        assert "qortex_feedback" in names


class TestMetricsSchemaUniqueness:
    """SCHEMA-2: All metric names are unique."""

    def test_no_duplicate_names(self):
        names = [m.name for m in METRICS]
        dupes = [n for n in names if names.count(n) > 1]
        assert len(names) == len(set(names)), f"Duplicate names: {dupes}"


class TestMetricsSchemaHistogramBuckets:
    """SCHEMA-3, SCHEMA-4: Bucket rules for histograms and non-histograms."""

    def test_all_histograms_have_buckets(self):
        for m in METRICS:
            if m.type == MetricType.HISTOGRAM:
                assert m.buckets is not None, f"{m.name} histogram missing buckets"
                assert len(m.buckets) > 0, f"{m.name} histogram has empty buckets"

    def test_histogram_buckets_are_sorted(self):
        for m in METRICS:
            if m.type == MetricType.HISTOGRAM and m.buckets:
                for i in range(1, len(m.buckets)):
                    assert m.buckets[i] > m.buckets[i - 1], (
                        f"{m.name} buckets not sorted: {m.buckets}"
                    )

    def test_counters_have_no_buckets(self):
        for m in METRICS:
            if m.type == MetricType.COUNTER:
                assert m.buckets is None, f"{m.name} counter should not have buckets"

    def test_gauges_have_no_buckets(self):
        for m in METRICS:
            if m.type == MetricType.GAUGE:
                assert m.buckets is None, f"{m.name} gauge should not have buckets"


class TestMetricsSchemaLabels:
    """SCHEMA-5: Label format validation."""

    def test_labels_are_tuples(self):
        for m in METRICS:
            assert isinstance(m.labels, tuple), f"{m.name} labels is {type(m.labels)}"

    def test_label_values_are_strings(self):
        for m in METRICS:
            for label in m.labels:
                assert isinstance(label, str), f"{m.name} label {label!r} not a string"

    def test_no_empty_string_labels(self):
        for m in METRICS:
            for label in m.labels:
                assert label != "", f"{m.name} has empty string label"


class TestMetricsSchemaConventions:
    """SCHEMA-6: Naming conventions for Prometheus/OTel compatibility."""

    def test_all_names_start_with_qortex(self):
        for m in METRICS:
            assert m.name.startswith("qortex_"), f"{m.name} missing qortex_ prefix"

    def test_duration_histogram_names_end_with_seconds(self):
        duration_hists = [
            m for m in METRICS if m.type == MetricType.HISTOGRAM and "duration" in m.name
        ]
        for m in duration_hists:
            assert m.name.endswith("_seconds"), f"{m.name} should end with _seconds"

    def test_counter_names_do_not_end_with_total(self):
        """OTel Prometheus exporter adds _total automatically."""
        for m in METRICS:
            if m.type == MetricType.COUNTER:
                assert not m.name.endswith("_total"), (
                    f"{m.name} should not end with _total (OTel adds it)"
                )

    def test_metricdef_is_frozen(self):
        m = METRICS[0]
        with pytest.raises(FrozenInstanceError):
            m.name = "changed"  # type: ignore[misc]

    def test_metrics_tuple_is_tuple(self):
        assert isinstance(METRICS, tuple)
