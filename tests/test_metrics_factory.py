"""Tests for OTel instrument creation from declarative schema.

Covers: FACTORY-1 through FACTORY-6
"""

from __future__ import annotations

import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from qortex_observe.metrics_factory import create_instruments, create_views  # noqa: E402
from qortex_observe.metrics_schema import METRICS, MetricType  # noqa: E402


@pytest.fixture()
def meter_and_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("test")
    return meter, reader, provider


class TestCreateInstruments:
    """FACTORY-1..4: Instrument creation from schema."""

    def test_returns_dict_with_all_metric_names(self, meter_and_reader):
        """FACTORY-1: create_instruments() returns dict with keys matching METRICS."""
        meter, _, _ = meter_and_reader
        instruments = create_instruments(meter)
        assert len(instruments) == len(METRICS)
        for m in METRICS:
            assert m.name in instruments, f"Missing instrument for {m.name}"

    def test_counters_have_add(self, meter_and_reader):
        """FACTORY-2: Counter MetricDefs produce instruments with .add()."""
        meter, _, _ = meter_and_reader
        instruments = create_instruments(meter)
        for m in METRICS:
            if m.type == MetricType.COUNTER:
                assert hasattr(instruments[m.name], "add"), f"{m.name} missing .add()"

    def test_histograms_have_record(self, meter_and_reader):
        """FACTORY-3: Histogram MetricDefs produce instruments with .record()."""
        meter, _, _ = meter_and_reader
        instruments = create_instruments(meter)
        for m in METRICS:
            if m.type == MetricType.HISTOGRAM:
                assert hasattr(instruments[m.name], "record"), f"{m.name} missing .record()"

    def test_gauges_have_set(self, meter_and_reader):
        """FACTORY-4: Gauge MetricDefs produce instruments with .set()."""
        meter, _, _ = meter_and_reader
        instruments = create_instruments(meter)
        for m in METRICS:
            if m.type == MetricType.GAUGE:
                assert hasattr(instruments[m.name], "set"), f"{m.name} missing .set()"

    def test_instruments_are_callable(self, meter_and_reader):
        """Instruments can be invoked without error."""
        meter, reader, _ = meter_and_reader
        instruments = create_instruments(meter)
        # Smoke test: call each instrument once
        for m in METRICS:
            inst = instruments[m.name]
            if m.type == MetricType.COUNTER:
                inst.add(1)
            elif m.type == MetricType.HISTOGRAM:
                inst.record(1.0)
            elif m.type == MetricType.GAUGE:
                inst.set(1.0)


class TestCreateViews:
    """FACTORY-5..6: OTel Views for custom histogram buckets."""

    def test_views_count_matches_histograms_with_buckets(self):
        """FACTORY-5: One View per histogram with custom buckets."""
        views = create_views()
        histograms_with_buckets = [
            m for m in METRICS if m.type == MetricType.HISTOGRAM and m.buckets
        ]
        assert len(views) == len(histograms_with_buckets)

    def test_views_are_view_instances(self):
        """All returned items are OTel View objects."""
        from opentelemetry.sdk.metrics.view import View

        views = create_views()
        for v in views:
            assert isinstance(v, View)

    def test_views_applied_to_meter_provider_no_error(self):
        """Views can be passed to MeterProvider without error."""
        views = create_views()
        reader = InMemoryMetricReader()
        # Should not raise
        provider = MeterProvider(metric_readers=[reader], views=views)
        meter = provider.get_meter("test")
        instruments = create_instruments(meter)
        # Record a value to exercise the View
        instruments["qortex_query_duration_seconds"].record(0.05)
        # Force collection
        data = reader.get_metrics_data()
        assert data is not None
