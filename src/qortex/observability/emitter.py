"""Singleton emitter: configure once, emit everywhere.

The global emit() function is the only API modules need.
It's a no-op when not configured (zero overhead in tests).
"""

from __future__ import annotations

from typing import Any

from pyventus.events import EventEmitter

_emitter: EventEmitter | None = None
_configured: bool = False


def emit(event: Any) -> None:
    """Fire-and-forget event emission. No-op if not configured."""
    if _emitter is not None:
        _emitter.emit(event)


def _setup_metrics_pipeline(cfg: "ObservabilityConfig") -> None:
    """Create unified metrics pipeline: schema -> factory -> handlers.

    Sets up OTel MeterProvider with configured readers (OTLP push
    and/or Prometheus pull), creates instruments from the declarative
    schema, applies histogram bucket Views, and registers event handlers.
    """
    from qortex.observability.logging import get_logger

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource

    from qortex.observability.metrics_factory import create_instruments, create_views
    from qortex.observability.metrics_handlers import register_metric_handlers

    resource = Resource.create({"service.name": cfg.otel_service_name})
    readers: list = []

    # OTLP push reader (to collector)
    if cfg.otel_enabled:
        try:
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            from qortex.observability.subscribers.otel import _get_exporters

            _, metric_exporter = _get_exporters(cfg.otel_protocol, cfg.otel_endpoint)
            readers.append(PeriodicExportingMetricReader(metric_exporter))
        except ImportError:
            get_logger().warning(
                "otel_enabled but OTLP exporter not installed",
                hint="pip install qortex[observability]",
            )

    # Prometheus pull reader (/metrics endpoint)
    if cfg.prometheus_enabled:
        try:
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            from prometheus_client import start_http_server

            readers.append(PrometheusMetricReader())
            start_http_server(cfg.prometheus_port)
        except ImportError:
            get_logger().warning(
                "prometheus_enabled but opentelemetry-exporter-prometheus not installed",
                hint="pip install qortex[observability]",
            )

    if not readers:
        get_logger().warning(
            "metrics.pipeline.no_readers",
            hint="No metric readers configured; metrics will be collected but not exported",
        )

    views = create_views()
    meter_provider = MeterProvider(
        resource=resource, metric_readers=readers, views=views,
    )
    meter = meter_provider.get_meter("qortex")
    instruments = create_instruments(meter)
    register_metric_handlers(instruments)

    reader_names = []
    if cfg.otel_enabled:
        reader_names.append("otlp")
    if cfg.prometheus_enabled:
        reader_names.append(f"prometheus:{cfg.prometheus_port}")

    get_logger().info(
        "metrics.pipeline.registered",
        readers=reader_names,
        metric_count=len(instruments),
    )


def configure(config: "ObservabilityConfig | None" = None) -> EventEmitter:
    """Initialize the global emitter and register subscribers.

    Called once at startup (MCP server init, CLI entry, test setup).
    Idempotent -- second call returns existing emitter.
    """
    global _emitter, _configured

    if _configured and _emitter is not None:
        return _emitter

    from qortex.observability.config import ObservabilityConfig

    cfg = config or ObservabilityConfig()

    # Setup structured logging first (formatter x destination from config)
    from qortex.observability.logging import setup_logging

    setup_logging(cfg)

    # Create emitter bound to our isolated linker
    from pyventus.core.processing.asyncio import AsyncIOProcessingService

    from qortex.observability.linker import QortexEventLinker

    _emitter = EventEmitter(
        event_linker=QortexEventLinker,
        event_processor=AsyncIOProcessingService(),
    )

    # Always register structured log subscriber (uses configured LogFormatter)
    from qortex.observability.subscribers import structlog_sub  # noqa: F401

    # Always register JSONL event sink if path configured
    if cfg.jsonl_path:
        from qortex.observability.subscribers.jsonl import register_jsonl_subscriber

        register_jsonl_subscriber(cfg.jsonl_path)

    # Unified metrics pipeline (OTel instruments, one set of handlers)
    if cfg.otel_enabled or cfg.prometheus_enabled:
        try:
            _setup_metrics_pipeline(cfg)
        except ImportError:
            from qortex.observability.logging import get_logger

            get_logger().warning(
                "metrics pipeline requires opentelemetry-sdk",
                hint="pip install qortex[observability]",
            )
        except Exception:
            from qortex.observability.logging import get_logger

            get_logger().error(
                "metrics.pipeline.failed",
                exc_info=True,
                hint="Metrics pipeline crashed during setup",
            )

    # OTel traces (spans only, metrics handled above)
    if cfg.otel_enabled:
        try:
            from qortex.observability.subscribers.otel import register_otel_traces

            register_otel_traces(cfg)
            from qortex.observability.logging import get_logger

            get_logger().info(
                "otel.traces.registered",
                endpoint=cfg.otel_endpoint,
                service=cfg.otel_service_name,
            )
        except ImportError:
            from qortex.observability.logging import get_logger

            get_logger().warning(
                "otel_enabled but opentelemetry not installed",
                hint="pip install qortex[observability]",
            )
        except Exception:
            from qortex.observability.logging import get_logger

            get_logger().error(
                "otel.traces.failed",
                exc_info=True,
                hint="OTel trace subscriber crashed during registration",
            )

    # Always register alert subscriber (no-op sink by default)
    from qortex.observability.subscribers.alert import register_alert_subscriber

    register_alert_subscriber(cfg)

    _configured = True
    return _emitter


def is_configured() -> bool:
    return _configured


def reset() -> None:
    """Reset for testing."""
    global _emitter, _configured

    from qortex.observability.logging import shutdown_logging

    shutdown_logging()

    _emitter = None
    _configured = False
