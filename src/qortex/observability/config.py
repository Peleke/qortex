"""Observability configuration, env-var driven.

All settings have safe defaults. Zero config required for basic
structured logging. Enable optional backends via env vars.

Logging architecture:
    LogFormatter (how records are structured) × LogDestination (where they go)

    Formatter: QORTEX_LOG_FORMATTER=structlog (default) | stdlib
    Destination: QORTEX_LOG_DESTINATION=stderr (default) | victorialogs | jsonl
    Renderer: QORTEX_LOG_FORMAT=json (default) | console
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ObservabilityConfig:
    """Observability configuration, env-var driven."""

    # --- Logging: formatter × destination ---
    log_formatter: str = field(
        default_factory=lambda: os.environ.get("QORTEX_LOG_FORMATTER", "structlog")
    )  # "structlog" | "stdlib"

    log_destination: str = field(
        default_factory=lambda: os.environ.get("QORTEX_LOG_DESTINATION", "stderr")
    )  # "stderr" | "victorialogs" | "jsonl"

    log_level: str = field(
        default_factory=lambda: os.environ.get("QORTEX_LOG_LEVEL", "INFO")
    )

    log_format: str = field(
        default_factory=lambda: os.environ.get("QORTEX_LOG_FORMAT", "json")
    )  # "json" | "console" (dev-friendly renderer)

    # VictoriaLogs destination
    victorialogs_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "QORTEX_VICTORIALOGS_ENDPOINT",
            "http://localhost:9428/insert/jsonline",
        )
    )
    victorialogs_batch_size: int = field(
        default_factory=lambda: int(
            os.environ.get("QORTEX_VICTORIALOGS_BATCH_SIZE", "100")
        )
    )
    victorialogs_flush_interval: float = field(
        default_factory=lambda: float(
            os.environ.get("QORTEX_VICTORIALOGS_FLUSH_INTERVAL", "5.0")
        )
    )

    # JSONL file destination (also used as event sink path)
    jsonl_path: str | None = field(
        default_factory=lambda: os.environ.get("QORTEX_LOG_PATH")
    )

    # --- OpenTelemetry ---
    otel_enabled: bool = field(
        default_factory=lambda: os.environ.get("QORTEX_OTEL_ENABLED", "").lower()
        in ("1", "true", "on")
    )
    otel_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
    )
    otel_service_name: str = field(
        default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "qortex")
    )

    # --- Prometheus ---
    prometheus_enabled: bool = field(
        default_factory=lambda: os.environ.get("QORTEX_PROMETHEUS_ENABLED", "").lower()
        in ("1", "true", "on")
    )
    prometheus_port: int = field(
        default_factory=lambda: int(os.environ.get("QORTEX_PROMETHEUS_PORT", "9090"))
    )

    # --- Alerting ---
    alert_enabled: bool = field(
        default_factory=lambda: os.environ.get("QORTEX_ALERTS_ENABLED", "").lower()
        in ("1", "true", "on")
    )
