"""MCP trace context propagation.

Extracts W3C traceparent from MCP params._meta for distributed tracing
across language boundaries. Follows OTel semantic conventions for GenAI
(OTel PR #2083) and MCP trace propagation (MCP PR #414).

Server-side extraction: MCP client sends traceparent in params._meta,
qortex server extracts it and creates spans linked to the client's trace.

Client-side injection: handled by qortex-observe-ts (Phase 8).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def extract_mcp_context(params: dict[str, Any]) -> Any:
    """Extract OTel trace context from MCP params._meta.

    Returns an OTel Context with the parent span from the client,
    or None if no trace context is available or OTel is not installed.
    """
    try:
        from opentelemetry.propagate import extract
    except ImportError:
        return None

    meta = params.get("_meta") if isinstance(params, dict) else None
    if not meta or not isinstance(meta, dict) or "traceparent" not in meta:
        return None

    return extract(meta)


def mcp_trace_middleware(
    tool_name: str,
    params: dict[str, Any],
    handler: Callable[..., Any],
) -> Any:
    """Wrap an MCP tool call in a traced span linked to client context.

    Graceful degradation: if OTel is not installed or context extraction
    fails, the handler is called directly without tracing.

    Args:
        tool_name: The MCP tool name (e.g. "retrieve", "ingest").
        params: The MCP tool call parameters (may contain _meta.traceparent).
        handler: The actual tool implementation to call.
    """
    try:
        from opentelemetry import context as otel_context
        from opentelemetry import trace
    except ImportError:
        return handler(params)

    ctx = extract_mcp_context(params)
    tracer = trace.get_tracer("qortex.mcp")

    span_kwargs: dict[str, Any] = {
        "attributes": {
            "mcp.method.name": "tools/call",
            "mcp.tool.name": tool_name,
            "gen_ai.operation.name": "execute_tool",
        },
    }

    # Attach extracted parent context if available
    token = None
    if ctx is not None:
        token = otel_context.attach(ctx)

    try:
        with tracer.start_as_current_span(f"mcp.tool.{tool_name}", **span_kwargs) as span:
            try:
                result = handler(params)
                return result
            except Exception as exc:
                span.set_status(trace.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise
    finally:
        if token is not None:
            otel_context.detach(token)
