"""Tests for @traced decorator and OverheadTimer (Phase 2 of #73)."""

from __future__ import annotations

import asyncio
import time

import pytest

from qortex.observability.tracing import OverheadTimer, get_overhead_timer, traced


@pytest.fixture(autouse=True)
def _reset_tracer_provider():
    """Reset OTel TracerProvider between tests so each test can set its own."""
    yield
    try:
        from opentelemetry import trace

        # Reset the singleton so next test can set a new provider
        # Cleanup handled by _reset_tracer_provider fixture
        # Force the internal "set once" flag to allow re-setting
        if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
            trace._TRACER_PROVIDER_SET_ONCE._done = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# OverheadTimer unit tests (DisjointIntervals algorithm)
# ---------------------------------------------------------------------------


class TestOverheadTimer:
    def test_no_externals(self):
        """All time is overhead when no external calls recorded."""
        timer = OverheadTimer()
        time.sleep(0.01)
        overhead = timer.overhead_seconds()
        assert overhead > 0.005

    def test_single_external(self):
        """overhead = total - external."""
        timer = OverheadTimer()
        t0 = time.monotonic()
        time.sleep(0.02)
        t1 = time.monotonic()
        timer.record_external(t0, t1)
        overhead = timer.overhead_seconds()
        # Overhead should be very small (just the timer bookkeeping)
        assert overhead < 0.01

    def test_overlapping_externals(self):
        """Concurrent external calls merge correctly (no double-count)."""
        timer = OverheadTimer()
        # Two overlapping intervals: [0.0, 0.5] and [0.3, 0.8]
        # Merged: [0.0, 0.8] = 0.8s external
        timer.record_external(0.0, 0.5)
        timer.record_external(0.3, 0.8)
        merged = OverheadTimer._merge_intervals(timer._external_intervals)
        assert len(merged) == 1
        assert merged[0] == (0.0, 0.8)

    def test_adjacent_externals(self):
        """Back-to-back externals merge into one interval."""
        timer = OverheadTimer()
        timer.record_external(0.0, 0.5)
        timer.record_external(0.5, 1.0)
        merged = OverheadTimer._merge_intervals(timer._external_intervals)
        assert len(merged) == 1
        assert merged[0] == (0.0, 1.0)

    def test_disjoint_externals(self):
        """Non-overlapping externals sum independently."""
        timer = OverheadTimer()
        timer.record_external(0.0, 0.2)
        timer.record_external(0.5, 0.7)
        merged = OverheadTimer._merge_intervals(timer._external_intervals)
        assert len(merged) == 2
        total_external = sum(end - start for start, end in merged)
        assert abs(total_external - 0.4) < 0.001

    def test_external_exceeds_total(self):
        """Clamped to 0 if external > total (clock skew edge case)."""
        timer = OverheadTimer()
        # Record external interval that is way longer than wall time
        timer.record_external(timer._start - 10.0, timer._start + 10.0)
        overhead = timer.overhead_seconds()
        assert overhead == 0.0

    def test_empty_intervals(self):
        """Empty interval list returns empty."""
        assert OverheadTimer._merge_intervals([]) == []


# ---------------------------------------------------------------------------
# @traced decorator tests
# ---------------------------------------------------------------------------


class TestTracedDecorator:
    def test_traced_creates_span(self):
        """@traced function creates OTel span with correct name."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @traced("test.operation")
        def do_work():
            return 42

        result = do_work()
        assert result == 42

        # Force flush
        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test.operation"

        # Cleanup
        # Cleanup handled by _reset_tracer_provider fixture

    def test_traced_nested_creates_parent_child(self):
        """Nested @traced calls form parent-child span tree."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @traced("child.op")
        def child():
            return "child"

        @traced("parent.op")
        def parent():
            return child()

        result = parent()
        assert result == "child"

        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        child_span = next(s for s in spans if s.name == "child.op")
        parent_span = next(s for s in spans if s.name == "parent.op")

        # Same trace
        assert child_span.context.trace_id == parent_span.context.trace_id
        # Child's parent is the parent span
        assert child_span.parent.span_id == parent_span.context.span_id

        # Cleanup handled by _reset_tracer_provider fixture

    def test_traced_external_records_interval(self):
        """@traced(external=True) records interval on parent OverheadTimer."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @traced("external.db", external=True)
        def db_call():
            time.sleep(0.02)
            return "data"

        @traced("parent.query")
        def query():
            return db_call()

        result = query()
        assert result == "data"

        provider.force_flush()
        spans = exporter.get_finished_spans()
        parent_span = next(s for s in spans if s.name == "parent.query")

        # Parent span should have overhead attributes
        attrs = dict(parent_span.attributes)
        assert "qortex.overhead_seconds" in attrs
        assert "qortex.total_seconds" in attrs
        # Overhead should be less than total (external time subtracted)
        assert attrs["qortex.overhead_seconds"] < attrs["qortex.total_seconds"]

        # Cleanup handled by _reset_tracer_provider fixture

    def test_traced_exception_records_error(self):
        """@traced records exception as span event + error attributes."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @traced("failing.op")
        def fail():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            fail()

        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert attrs["error"] is True
        assert attrs["error.type"] == "ValueError"

        # Cleanup handled by _reset_tracer_provider fixture

    def test_traced_exception_propagates(self):
        """@traced doesn't swallow exceptions."""
        from opentelemetry.sdk.trace import TracerProvider

        from opentelemetry import trace

        trace.set_tracer_provider(TracerProvider())

        @traced("boom")
        def boom():
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            boom()

        # Cleanup handled by _reset_tracer_provider fixture

    def test_traced_async_creates_span(self):
        """@traced works on async functions."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @traced("async.op")
        async def async_work():
            await asyncio.sleep(0.01)
            return "async_result"

        result = asyncio.run(async_work())
        assert result == "async_result"

        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async.op"

        # Cleanup handled by _reset_tracer_provider fixture

    def test_traced_graceful_without_otel(self):
        """@traced is no-op when opentelemetry not installed."""
        import unittest.mock

        # Mock ImportError for opentelemetry
        with unittest.mock.patch.dict("sys.modules", {"opentelemetry": None}):
            # Need a fresh decorator since the import happens at call time
            @traced("no.otel")
            def work():
                return "still works"

            # Should not raise
            result = work()
            assert result == "still works"

    def test_get_overhead_timer_returns_none_outside_traced(self):
        """get_overhead_timer() returns None when not inside @traced."""
        assert get_overhead_timer() is None


# ---------------------------------------------------------------------------
# Overhead metric integration
# ---------------------------------------------------------------------------


class TestOverheadMetric:
    def test_query_completed_overhead_field(self):
        """QueryCompleted accepts overhead_seconds."""
        from qortex.observability.events import QueryCompleted

        event = QueryCompleted(
            query_id="q1",
            latency_ms=100.0,
            seed_count=5,
            result_count=3,
            activated_nodes=10,
            mode="graph",
            timestamp="2024-01-01T00:00:00",
            overhead_seconds=0.05,
        )
        assert event.overhead_seconds == 0.05

    def test_query_completed_overhead_defaults_none(self):
        """overhead_seconds defaults to None for backward compat."""
        from qortex.observability.events import QueryCompleted

        event = QueryCompleted(
            query_id="q1",
            latency_ms=100.0,
            seed_count=5,
            result_count=3,
            activated_nodes=10,
            mode="graph",
            timestamp="2024-01-01T00:00:00",
        )
        assert event.overhead_seconds is None

    def test_overhead_metric_recorded_when_present(self):
        """Overhead metric is recorded when QueryCompleted has overhead_seconds."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        from qortex.observability.events import QueryCompleted
        from qortex.observability.linker import QortexEventLinker
        from qortex.observability.metrics_factory import create_instruments, create_views
        from qortex.observability.metrics_handlers import register_metric_handlers

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader], views=create_views())
        meter = provider.get_meter("test")
        instruments = create_instruments(meter)
        register_metric_handlers(instruments)

        event = QueryCompleted(
            query_id="q1",
            latency_ms=100.0,
            seed_count=5,
            result_count=3,
            activated_nodes=10,
            mode="graph",
            timestamp="2024-01-01T00:00:00",
            overhead_seconds=0.042,
        )

        # Fire event through linker
        import asyncio as aio

        event_name = type(event).__name__
        registry = QortexEventLinker.get_registry()
        for sub in registry.get(event_name, set()):
            aio.run(sub.execute(event))

        data = reader.get_metrics_data()
        metrics = {}
        if data:
            for rm in data.resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        metrics[metric.name] = metric

        assert "qortex_query_overhead_seconds" in metrics
        # Should have recorded 0.042
        dp = metrics["qortex_query_overhead_seconds"].data.data_points[0]
        assert dp.sum > 0

        QortexEventLinker.remove_all()

    def test_overhead_metric_not_recorded_when_none(self):
        """Overhead metric is NOT recorded when overhead_seconds is None."""
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        from qortex.observability.events import QueryCompleted
        from qortex.observability.linker import QortexEventLinker
        from qortex.observability.metrics_factory import create_instruments, create_views
        from qortex.observability.metrics_handlers import register_metric_handlers

        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader], views=create_views())
        meter = provider.get_meter("test")
        instruments = create_instruments(meter)
        register_metric_handlers(instruments)

        event = QueryCompleted(
            query_id="q1",
            latency_ms=100.0,
            seed_count=5,
            result_count=3,
            activated_nodes=10,
            mode="graph",
            timestamp="2024-01-01T00:00:00",
            # overhead_seconds defaults to None
        )

        import asyncio as aio

        event_name = type(event).__name__
        registry = QortexEventLinker.get_registry()
        for sub in registry.get(event_name, set()):
            aio.run(sub.execute(event))

        data = reader.get_metrics_data()
        metrics = {}
        if data:
            for rm in data.resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        metrics[metric.name] = metric

        # qortex_query_overhead_seconds should either not exist or have no data points
        if "qortex_query_overhead_seconds" in metrics:
            dp = metrics["qortex_query_overhead_seconds"].data.data_points
            assert len(dp) == 0

        QortexEventLinker.remove_all()
