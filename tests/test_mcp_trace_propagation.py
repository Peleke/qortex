"""Tests for qortex.observe.mcp -- MCP trace context propagation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qortex.observe.mcp._propagation import extract_mcp_context, mcp_trace_middleware

# Valid W3C traceparent: version-trace_id-parent_id-flags
VALID_TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


# ── extract_mcp_context Tests ─────────────────────────────────────


class TestExtractMcpContext:
    def test_extract_with_valid_traceparent(self):
        """Valid traceparent in _meta returns a non-None context."""
        params = {"_meta": {"traceparent": VALID_TRACEPARENT}}
        ctx = extract_mcp_context(params)
        assert ctx is not None

    def test_extract_no_meta_key(self):
        """Missing _meta returns None."""
        ctx = extract_mcp_context({"query": "test"})
        assert ctx is None

    def test_extract_empty_meta(self):
        ctx = extract_mcp_context({"_meta": {}})
        assert ctx is None

    def test_extract_meta_without_traceparent(self):
        ctx = extract_mcp_context({"_meta": {"session_id": "abc"}})
        assert ctx is None

    def test_extract_meta_not_dict(self):
        ctx = extract_mcp_context({"_meta": "not-a-dict"})
        assert ctx is None

    def test_extract_params_not_dict(self):
        """Non-dict params returns None instead of crashing."""
        ctx = extract_mcp_context("not-a-dict")  # type: ignore
        assert ctx is None

    def test_extract_empty_params(self):
        ctx = extract_mcp_context({})
        assert ctx is None

    def test_extract_with_tracestate(self):
        """traceparent + tracestate both propagated."""
        params = {
            "_meta": {
                "traceparent": VALID_TRACEPARENT,
                "tracestate": "vendor=value",
            }
        }
        ctx = extract_mcp_context(params)
        assert ctx is not None

    def test_extract_graceful_without_otel(self):
        """Returns None when opentelemetry not installed."""
        with patch.dict("sys.modules", {"opentelemetry.propagate": None}):
            # Force re-import failure
            import importlib

            from qortex.observe.mcp import _propagation

            importlib.reload(_propagation)
            ctx = _propagation.extract_mcp_context({"_meta": {"traceparent": VALID_TRACEPARENT}})
            # Reload to restore
            importlib.reload(_propagation)
            # When OTel import fails, returns None
            assert ctx is None


# ── mcp_trace_middleware Tests ────────────────────────────────────


class TestMcpTraceMiddleware:
    @pytest.fixture(autouse=True)
    def _reset_tracer_provider(self):
        """Reset the global TracerProvider between tests."""
        yield
        try:
            from opentelemetry import trace

            if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
                trace._TRACER_PROVIDER_SET_ONCE._done = False
        except ImportError:
            pass

    def _make_provider(self):
        """Create a fresh TracerProvider with InMemorySpanExporter."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return provider, exporter

    def test_middleware_calls_handler(self):
        """Handler is called and its result returned."""
        handler = MagicMock(return_value={"result": "ok"})
        result = mcp_trace_middleware("test_tool", {}, handler)
        handler.assert_called_once_with({})
        assert result == {"result": "ok"}

    def test_middleware_creates_span(self):
        """Tool call produces a span with correct name."""
        provider, exporter = self._make_provider()
        try:
            handler = MagicMock(return_value="ok")
            mcp_trace_middleware("retrieve", {"query": "test"}, handler)

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].name == "mcp.tool.retrieve"
        finally:
            provider.shutdown()

    def test_middleware_sets_attributes(self):
        """Span has mcp.method.name, mcp.tool.name, gen_ai.operation.name."""
        provider, exporter = self._make_provider()
        try:
            handler = MagicMock(return_value="ok")
            mcp_trace_middleware("ingest", {}, handler)

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes)
            assert attrs["mcp.method.name"] == "tools/call"
            assert attrs["mcp.tool.name"] == "ingest"
            assert attrs["gen_ai.operation.name"] == "execute_tool"
        finally:
            provider.shutdown()

    def test_middleware_propagates_exception(self):
        """Exceptions propagate through middleware."""

        def failing_handler(params):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            mcp_trace_middleware("fail_tool", {}, failing_handler)

    def test_middleware_records_exception_on_span(self):
        """Span records exception when handler fails."""
        from opentelemetry import trace

        provider, exporter = self._make_provider()
        try:

            def failing_handler(params):
                raise RuntimeError("test error")

            with pytest.raises(RuntimeError):
                mcp_trace_middleware("err_tool", {}, failing_handler)

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].status.status_code == trace.StatusCode.ERROR
            # Exception should be recorded as event
            events = spans[0].events
            assert any(e.name == "exception" for e in events)
        finally:
            provider.shutdown()

    def test_middleware_links_to_parent_trace(self):
        """When traceparent is present, span links to parent trace."""
        provider, exporter = self._make_provider()
        try:
            handler = MagicMock(return_value="ok")
            params = {"_meta": {"traceparent": VALID_TRACEPARENT}}
            mcp_trace_middleware("linked_tool", params, handler)

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            # The span's trace ID should match the traceparent's trace ID
            expected_trace_id = int("4bf92f3577b34da6a3ce929d0e0e4736", 16)
            assert span.context.trace_id == expected_trace_id
            # The parent should be the span from the traceparent
            expected_parent_id = int("00f067aa0ba902b7", 16)
            assert span.parent.span_id == expected_parent_id
        finally:
            provider.shutdown()

    def test_middleware_graceful_without_otel(self):
        """When OTel not available, handler still executes."""
        handler = MagicMock(return_value=42)
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
            import importlib

            from qortex.observe.mcp import _propagation

            importlib.reload(_propagation)
            result = _propagation.mcp_trace_middleware("tool", {}, handler)
            importlib.reload(_propagation)

        handler.assert_called_once()
        assert result == 42


# ── _mcp_traced decorator wiring Tests ────────────────────────────


class TestMcpTracedDecorator:
    """Verify _mcp_traced decorator on MCP server tool wrappers."""

    async def test_all_tools_have_traced_decorator(self):
        """Every @mcp.tool wrapper should also have @_mcp_traced."""
        from qortex.mcp.server import mcp

        # Use public API to list tools (avoids private _tool_manager._tools)
        tools = await mcp.get_tools()
        assert len(tools) >= 36, f"Expected >=36 tools, got {len(tools)}"

        for tool in tools:
            if hasattr(tool, "fn"):
                fn = tool.fn
                assert hasattr(fn, "__wrapped__"), (
                    f"{tool.name} missing __wrapped__ -- not decorated with @_mcp_traced"
                )

    def test_traced_decorator_creates_span(self):
        """_mcp_traced creates a span when calling a tool."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from qortex.mcp.server import _mcp_traced

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        try:

            @_mcp_traced
            def my_test_tool(x: int = 0) -> dict:
                return {"value": x}

            result = my_test_tool(x=42)
            assert result == {"value": 42}

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].name == "mcp.tool.my_test_tool"
            assert spans[0].attributes["mcp.tool.name"] == "my_test_tool"
        finally:
            provider.shutdown()
            try:
                if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
                    trace._TRACER_PROVIDER_SET_ONCE._done = False
            except Exception:
                pass

    def test_traced_decorator_preserves_signature(self):
        """_mcp_traced preserves function name and docstring."""
        from qortex.mcp.server import _mcp_traced

        @_mcp_traced
        def example_tool(query: str, top_k: int = 10) -> dict:
            """Example docstring."""
            return {}

        assert example_tool.__name__ == "example_tool"
        assert example_tool.__doc__ == "Example docstring."

    def test_traced_decorator_propagates_exception(self):
        """_mcp_traced re-raises exceptions from the wrapped function."""
        from qortex.mcp.server import _mcp_traced

        @_mcp_traced
        def failing_tool() -> dict:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing_tool()
