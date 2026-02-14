"""OpenTelemetry trace setup: TracerProvider + exporter configuration.

Span creation is handled by the @traced decorator (tracing.py), not by
event handlers. This module only sets up the OTel SDK and exporter pipeline.
Metrics are handled by the unified pipeline (metrics_schema + metrics_factory
+ metrics_handlers).

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
    from qortex.observe.config import ObservabilityConfig


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

    # Default: gRPC. Falls back to HTTP if grpcio unavailable.
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
        from qortex.observe.logging import get_logger

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
    """Set up OTel TracerProvider and exporter.

    Spans are created by the @traced decorator, not event handlers.
    This function just configures the SDK so those spans get exported.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    from qortex.observe.tracing import SelectiveSpanProcessor

    # ── Resource ──────────────────────────────────────────────────────
    resource = Resource.create({"service.name": config.otel_service_name})

    # ── Exporters (gRPC with HTTP fallback) ───────────────────────────
    span_exporter, _ = _get_exporters(config.otel_protocol, config.otel_endpoint)

    # ── Tracer with selective export ──────────────────────────────────
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SelectiveSpanProcessor(
        span_exporter,
        sample_rate=config.otel_trace_sample_rate,
        latency_threshold_ms=config.otel_trace_latency_threshold_ms,
    ))
    trace.set_tracer_provider(provider)


# Backwards compat alias (used by existing tests)
register_otel_subscriber = register_otel_traces
