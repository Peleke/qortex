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


def _int_env(name: str, default: str, *, min_val: int = 0, max_val: int | None = None) -> int:
    """Parse an integer from an env var with validation."""
    raw = os.environ.get(name, default)
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(
            f"Invalid integer for {name}: {raw!r}. Expected a number, got {type(raw).__name__}."
        ) from None
    if val < min_val:
        raise ValueError(f"{name}={val} is below minimum {min_val}.")
    if max_val is not None and val > max_val:
        raise ValueError(f"{name}={val} exceeds maximum {max_val}.")
    return val


def _float_env(name: str, default: str, *, min_val: float = 0.0) -> float:
    """Parse a float from an env var with validation."""
    raw = os.environ.get(name, default)
    try:
        val = float(raw)
    except ValueError:
        raise ValueError(
            f"Invalid float for {name}: {raw!r}. Expected a number."
        ) from None
    if val < min_val:
        raise ValueError(f"{name}={val} is below minimum {min_val}.")
    return val


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
        default_factory=lambda: _int_env(
            "QORTEX_VICTORIALOGS_BATCH_SIZE", "100", min_val=1, max_val=10_000
        )
    )
    victorialogs_flush_interval: float = field(
        default_factory=lambda: _float_env(
            "QORTEX_VICTORIALOGS_FLUSH_INTERVAL", "5.0", min_val=0.1
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
    otel_protocol: str = field(
        default_factory=lambda: os.environ.get(
            "OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"
        )
    )  # "grpc" | "http/protobuf"

    # Trace sampling: fraction of non-error, non-slow spans to export.
    # 1.0 = export everything (demos/debugging), 0.1 = 10% sample (production).
    otel_trace_sample_rate: float = field(
        default_factory=lambda: _float_env(
            "QORTEX_OTEL_TRACE_SAMPLE_RATE", "0.1", min_val=0.0
        )
    )
    otel_trace_latency_threshold_ms: float = field(
        default_factory=lambda: _float_env(
            "QORTEX_OTEL_TRACE_LATENCY_THRESHOLD_MS", "100.0", min_val=0.0
        )
    )

    # --- Prometheus ---
    prometheus_enabled: bool = field(
        default_factory=lambda: os.environ.get("QORTEX_PROMETHEUS_ENABLED", "").lower()
        in ("1", "true", "on")
    )
    prometheus_port: int = field(
        default_factory=lambda: _int_env(
            "QORTEX_PROMETHEUS_PORT", "9464", min_val=1, max_val=65535
        )
    )

    # --- Alerting ---
    alert_enabled: bool = field(
        default_factory=lambda: os.environ.get("QORTEX_ALERTS_ENABLED", "").lower()
        in ("1", "true", "on")
    )
