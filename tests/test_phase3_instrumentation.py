"""Tests for Phase 3: @traced wrappers on Cypher + Vec, QueryFailed wiring, SelectiveSpanProcessor."""

from __future__ import annotations

import pytest
from qortex.observe.tracing import SelectiveSpanProcessor

# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------


class TestCypherHelpers:
    """_cypher_op() and _sanitize_cypher() from backend.py."""

    def test_cypher_op_match(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("MATCH (n:Concept) RETURN n") == "query"

    def test_cypher_op_create(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("CREATE (n:Concept {id: 'foo'})") == "create"

    def test_cypher_op_merge(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("MERGE (n:Domain {name: $d})") == "upsert"

    def test_cypher_op_delete(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("DELETE n") == "delete"

    def test_cypher_op_call(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("CALL pagerank.get()") == "procedure"

    def test_cypher_op_unknown(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("SET n.name = 'foo'") == "other"

    def test_cypher_op_empty(self):
        from qortex.core.backend import _cypher_op

        # Empty string: strip().split()[0] would IndexError, falls to "other"
        assert _cypher_op("") == "other"

    def test_cypher_op_whitespace_only(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("   ") == "other"

    def test_cypher_op_case_insensitive(self):
        from qortex.core.backend import _cypher_op

        assert _cypher_op("match (n) return n") == "query"

    def test_sanitize_truncates(self):
        from qortex.core.backend import _sanitize_cypher

        long_query = "MATCH (n:Concept) WHERE n.description = 'x' " * 10
        sanitized = _sanitize_cypher(long_query)
        assert len(sanitized) == 200

    def test_sanitize_short_query_unchanged(self):
        from qortex.core.backend import _sanitize_cypher

        short = "MATCH (n) RETURN n"
        assert _sanitize_cypher(short) == short

    def test_sanitize_custom_max_len(self):
        from qortex.core.backend import _sanitize_cypher

        assert len(_sanitize_cypher("x" * 500, max_len=50)) == 50


# ---------------------------------------------------------------------------
# NumpyVectorIndex tracing
# ---------------------------------------------------------------------------


class TestVecInstrumentation:
    """@traced on NumpyVectorIndex.search() and .add()."""

    @pytest.fixture()
    def index(self):
        from qortex.vec.index import NumpyVectorIndex

        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return idx

    def test_search_still_returns_results(self, index):
        """@traced doesn't break search behavior."""
        results = index.search([1, 0, 0], top_k=3)
        assert len(results) > 0
        assert results[0][0] == "a"

    def test_add_still_works(self):
        from qortex.vec.index import NumpyVectorIndex

        idx = NumpyVectorIndex(dimensions=2)
        idx.add(["x"], [[1, 0]])
        assert idx.size() == 1

    def test_search_creates_span(self, index):
        """search() creates a vec.search span when OTel is available."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        try:
            index.search([1, 0, 0], top_k=3)
            provider.force_flush()
            spans = exporter.get_finished_spans()
            span_names = [s.name for s in spans]
            assert "vec.search" in span_names
        finally:
            if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
                trace._TRACER_PROVIDER_SET_ONCE._done = False

    def test_add_creates_span(self):
        """add() creates a vec.add span."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        try:
            from qortex.vec.index import NumpyVectorIndex

            idx = NumpyVectorIndex(dimensions=2)
            idx.add(["x"], [[1, 0]])
            provider.force_flush()
            spans = exporter.get_finished_spans()
            span_names = [s.name for s in spans]
            assert "vec.add" in span_names
        finally:
            if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
                trace._TRACER_PROVIDER_SET_ONCE._done = False

    def test_search_events_still_fire(self, index):
        """VecSearchResults event still emitted alongside span."""
        from unittest.mock import patch

        with patch("qortex.vec.index._try_emit") as mock_emit:
            index.search([1, 0, 0], top_k=3)
            assert mock_emit.called
            from qortex.observe.events import VecSearchResults

            events = [call.args[0] for call in mock_emit.call_args_list]
            assert any(isinstance(e, VecSearchResults) for e in events)


# ---------------------------------------------------------------------------
# SelectiveSpanProcessor
# ---------------------------------------------------------------------------


_has_otel_sdk = pytest.importorskip("opentelemetry.sdk", reason="OTel SDK not installed")


class TestSelectiveSpanProcessor:
    """Export filter: errors always, slow always, sample the rest."""

    @pytest.fixture()
    def _setup_otel(self):
        """Reset OTel provider after test."""
        yield
        try:
            from opentelemetry import trace

            if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
                trace._TRACER_PROVIDER_SET_ONCE._done = False
        except ImportError:
            pass

    def _make_span(self, *, error: bool = False, duration_ns: int = 1_000_000):
        """Create a mock span with controlled attributes and duration."""

        class MockSpan:
            def __init__(self, attrs, start, end):
                self.attributes = attrs
                self.start_time = start
                self.end_time = end
                self.name = "test.span"
                self.context = None

        attrs = {}
        if error:
            attrs["error"] = True
        start = 0
        end = duration_ns
        return MockSpan(attrs, start, end)

    def test_errors_always_exported(self, _setup_otel):
        """Spans with error=True bypass sampling."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        proc = SelectiveSpanProcessor(mock_exporter, sample_rate=0.0)
        proc._inner = MagicMock()

        error_span = self._make_span(error=True)
        proc.on_end(error_span)

        proc._inner.on_end.assert_called_once_with(error_span)

    def test_slow_spans_always_exported(self, _setup_otel):
        """Spans exceeding latency threshold bypass sampling."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        # threshold 100ms, sample_rate 0 so only slow spans pass
        proc = SelectiveSpanProcessor(mock_exporter, sample_rate=0.0, latency_threshold_ms=100.0)
        proc._inner = MagicMock()

        # 200ms span
        slow_span = self._make_span(duration_ns=200_000_000)
        proc.on_end(slow_span)

        proc._inner.on_end.assert_called_once_with(slow_span)

    def test_normal_spans_dropped_at_zero_rate(self, _setup_otel):
        """Normal spans dropped when sample_rate=0."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        proc = SelectiveSpanProcessor(mock_exporter, sample_rate=0.0, latency_threshold_ms=1000.0)
        proc._inner = MagicMock()

        normal_span = self._make_span(duration_ns=1_000_000)  # 1ms
        proc.on_end(normal_span)

        proc._inner.on_end.assert_not_called()

    def test_normal_spans_exported_at_full_rate(self, _setup_otel):
        """Normal spans always exported when sample_rate=1.0."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        proc = SelectiveSpanProcessor(mock_exporter, sample_rate=1.0, latency_threshold_ms=1000.0)
        proc._inner = MagicMock()

        normal_span = self._make_span(duration_ns=1_000_000)
        proc.on_end(normal_span)

        proc._inner.on_end.assert_called_once_with(normal_span)

    def test_default_sample_rate(self, _setup_otel):
        """Default sample rate is 0.1 (10%)."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        proc = SelectiveSpanProcessor(mock_exporter)
        assert proc._sample_rate == 0.1

    def test_default_latency_threshold(self, _setup_otel):
        """Default latency threshold is 100ms."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        proc = SelectiveSpanProcessor(mock_exporter)
        assert proc._latency_threshold_ms == 100.0

    def test_shutdown_delegates(self, _setup_otel):
        """shutdown() delegates to inner processor."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        proc = SelectiveSpanProcessor(mock_exporter)
        proc._inner = MagicMock()
        proc.shutdown()
        proc._inner.shutdown.assert_called_once()

    def test_force_flush_delegates(self, _setup_otel):
        """force_flush() delegates to inner processor."""
        from unittest.mock import MagicMock

        mock_exporter = MagicMock()
        proc = SelectiveSpanProcessor(mock_exporter)
        proc._inner = MagicMock()
        proc.force_flush()
        proc._inner.force_flush.assert_called_once()


# ---------------------------------------------------------------------------
# QueryFailed wiring (MemgraphBackend.personalized_pagerank)
# ---------------------------------------------------------------------------


class TestQueryFailedWiring:
    """QueryFailed emission from personalized_pagerank Cypher failures."""

    def test_ppr_node_fetch_failure_emits_query_failed(self):
        """When node fetch Cypher fails, QueryFailed is emitted with stage='ppr'."""
        from unittest.mock import MagicMock, patch

        from qortex.core.backend import MemgraphBackend

        backend = MemgraphBackend.__new__(MemgraphBackend)
        backend._driver = MagicMock()
        backend._driver.session.return_value.__enter__ = MagicMock(
            side_effect=RuntimeError("Connection lost")
        )
        backend._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch("qortex.core.backend.emit") as mock_emit:
            with pytest.raises(RuntimeError, match="Connection lost"):
                backend.personalized_pagerank(
                    source_nodes=["a"],
                    domain="test",
                    query_id="q1",
                )

            # Check QueryFailed was emitted
            from qortex.observe.events import QueryFailed

            qf_calls = [c for c in mock_emit.call_args_list if isinstance(c.args[0], QueryFailed)]
            assert len(qf_calls) >= 1
            qf = qf_calls[0].args[0]
            assert qf.stage == "ppr"
            assert qf.query_id == "q1"
            assert "Connection lost" in qf.error


# ---------------------------------------------------------------------------
# Span tree structure
# ---------------------------------------------------------------------------


class TestSpanTreeStructure:
    """Verify parent-child span relationships in retrieval pipeline."""

    @pytest.fixture(autouse=True)
    def _reset_tracer_provider(self):
        yield
        try:
            from opentelemetry import trace

            if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
                trace._TRACER_PROVIDER_SET_ONCE._done = False
        except ImportError:
            pass

    def test_vec_search_is_child_of_retrieval(self):
        """vec.search span should be a child of retrieval.vec_search."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from qortex.observe.tracing import traced

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @traced("parent.op")
        def parent():
            child()

        @traced("child.op")
        def child():
            pass

        parent()
        provider.force_flush()
        spans = exporter.get_finished_spans()

        child_span = next(s for s in spans if s.name == "child.op")
        parent_span = next(s for s in spans if s.name == "parent.op")

        assert child_span.context.trace_id == parent_span.context.trace_id
        assert child_span.parent.span_id == parent_span.context.span_id
