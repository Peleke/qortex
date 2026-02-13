"""OpenTelemetry trace subscriber: routes query events to OTel spans.

Metrics have moved to the unified pipeline (metrics_schema + metrics_factory
+ metrics_handlers). This module only handles trace spans.

Requires qortex[observability] (opentelemetry-api, opentelemetry-sdk,
opentelemetry-exporter-otlp).

Protocol selection (OTEL_EXPORTER_OTLP_PROTOCOL):
    "grpc"          -> gRPC exporters (default, requires grpcio)
    "http/protobuf" -> HTTP/protobuf exporters (works through firewalls/VMs)

If gRPC import fails, automatically falls back to HTTP/protobuf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qortex.observability.config import ObservabilityConfig


def _get_exporters(protocol: str, endpoint: str):
    """Return (SpanExporter, MetricExporter) for the selected protocol.

    Falls back from gRPC to HTTP/protobuf if grpcio is unavailable.
    """
    if protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=endpoint + "/v1/traces"), OTLPMetricExporter(
            endpoint=endpoint + "/v1/metrics"
        )

    # Default: gRPC — fall back to HTTP if grpcio unavailable
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=endpoint), OTLPMetricExporter(
            endpoint=endpoint
        )
    except ImportError:
        from qortex.observability.logging import get_logger

        get_logger().warning(
            "otel.grpc.unavailable",
            hint="grpcio not installed, falling back to http/protobuf",
        )
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(
            endpoint=endpoint + "/v1/traces"
        ), OTLPMetricExporter(endpoint=endpoint + "/v1/metrics")


def register_otel_traces(config: ObservabilityConfig) -> None:
    """Register OTel trace spans for query lifecycle.

    Only handles traces. Metrics are registered separately via the
    unified metrics pipeline in emitter.configure().
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from qortex.observability.events import (
        QueryCompleted,
        QueryFailed,
        QueryStarted,
    )
    from qortex.observability.linker import QortexEventLinker

    # ── Resource ──────────────────────────────────────────────────────
    resource = Resource.create({"service.name": config.otel_service_name})

    # ── Exporters (gRPC with HTTP fallback) ───────────────────────────
    span_exporter, _ = _get_exporters(config.otel_protocol, config.otel_endpoint)

    # ── Tracer ────────────────────────────────────────────────────────
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("qortex")

    # ── Trace state ───────────────────────────────────────────────────
    _MAX_ACTIVE_SPANS = 1000
    _active_spans: dict[str, trace.Span] = {}

    # ── Query lifecycle (traces only) ─────────────────────────────────

    @QortexEventLinker.on(QueryStarted)
    def _on_query_start(event: QueryStarted) -> None:
        # Evict oldest spans if we hit the cap (prevents unbounded growth
        # when QueryCompleted/QueryFailed events are lost).
        if len(_active_spans) >= _MAX_ACTIVE_SPANS:
            oldest_key = next(iter(_active_spans))
            stale = _active_spans.pop(oldest_key)
            stale.set_attribute("evicted", True)
            stale.end()

        span = tracer.start_span(
            "qortex.query",
            attributes={
                "query.id": event.query_id,
                "query.mode": event.mode,
                "query.top_k": event.top_k,
            },
        )
        _active_spans[event.query_id] = span

    @QortexEventLinker.on(QueryCompleted)
    def _on_query_complete(event: QueryCompleted) -> None:
        span = _active_spans.pop(event.query_id, None)
        if span:
            span.set_attribute("result_count", event.result_count)
            span.set_attribute("seed_count", event.seed_count)
            span.set_attribute("activated_nodes", event.activated_nodes)
            span.set_attribute("latency_ms", event.latency_ms)
            span.end()

    @QortexEventLinker.on(QueryFailed)
    def _on_query_failed(event: QueryFailed) -> None:
        span = _active_spans.pop(event.query_id, None)
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.stage", event.stage)
            span.set_attribute("error.message", event.error)
            span.end()


# Backwards compat alias (used by existing tests)
register_otel_subscriber = register_otel_traces
