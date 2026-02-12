"""Structured logging: swappable formatter × destination via config.

Architecture:
    LogFormatter  — HOW records are structured (structlog, stdlib, future: loguru)
    LogDestination — WHERE output goes (stderr, VictoriaLogs, JSONL file)

    setup_logging(config) composes them: formatter.setup() returns a
    logging.Formatter, destination.create_handler() returns a logging.Handler,
    the handler gets the formatter, and it's attached to the root logger.

    All 44 existing files using logging.getLogger() get structured output
    automatically — zero code changes — because both formatters bridge stdlib.

Swapping:
    QORTEX_LOG_FORMATTER=structlog   (default)
    QORTEX_LOG_DESTINATION=stderr    (default dev)
    QORTEX_LOG_DESTINATION=victorialogs  (default prod)

    Or register your own:
        from qortex.observability.logging import register_formatter, register_destination
        register_destination("datadog", MyDatadogDestination)

VictoriaLogs:
    Accepts JSON lines at POST /insert/jsonline.  Auto-indexes all fields.
    Special fields: _time (timestamp), _msg (message text).
    Batches in memory, flushes periodically or on threshold via background thread.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qortex.observability.config import ObservabilityConfig


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class LogFormatter(Protocol):
    """Strategy: how log records are structured.

    setup() configures the formatting pipeline and returns a
    logging.Formatter that handlers will use.

    get_logger() returns a logger with the right API (structlog's
    key=value kwargs for structlog, stdlib's %-formatting for stdlib).
    """

    def setup(self, config: ObservabilityConfig) -> logging.Formatter: ...

    def get_logger(self, name: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class LogDestination(Protocol):
    """Strategy: where formatted log output is shipped.

    create_handler() returns a logging.Handler — the bridge between
    Python's logging system and the destination.
    """

    def create_handler(self, formatter: logging.Formatter) -> logging.Handler: ...

    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class StructlogFormatter:
    """structlog processor pipeline + stdlib bridge.

    After setup():
    - structlog.get_logger() → structured JSON/console output
    - logging.getLogger()  → ALSO structured (via stdlib bridge)
    - All 44 existing files get structured logs without code changes
    """

    def setup(self, config: ObservabilityConfig) -> logging.Formatter:
        import structlog

        shared_processors: list = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]

        if config.log_format == "console":
            renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
        else:
            renderer = structlog.processors.JSONRenderer()

        # Configure structlog native loggers
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Return formatter for the handler
        return structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

    def get_logger(self, name: str, **kwargs: Any) -> Any:
        import structlog

        return structlog.get_logger(name, **kwargs)


class StdlibFormatter:
    """Pure stdlib logging with JSON formatting.

    No structlog dependency used at runtime. For environments that
    can't or don't want structlog.
    """

    def setup(self, config: ObservabilityConfig) -> logging.Formatter:
        if config.log_format == "console":
            return logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        return _StdlibJsonFormatter()

    def get_logger(self, name: str, **kwargs: Any) -> Any:
        return _StructuredStdlibLogger(logging.getLogger(name))


class _StdlibJsonFormatter(logging.Formatter):
    """JSON formatter for stdlib logging (no structlog dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        d: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        # Include any structured data from the extra dict
        if hasattr(record, "_structured"):
            d.update(record._structured)  # type: ignore[union-attr]
        if record.exc_info and record.exc_info[1]:
            d["exception"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


class _StructuredStdlibLogger:
    """Wrapper that gives stdlib loggers a structlog-like kwargs API.

    Bridges the gap: event subscriber code does logger.info("query.started",
    query_id="abc", latency_ms=42.0).  Stdlib logger doesn't accept arbitrary
    kwargs, so this wrapper stores them in the LogRecord for the formatter.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "(unknown)",
            0,
            event,
            (),
            None,
        )
        record._structured = kwargs  # type: ignore[attr-defined]
        self._logger.handle(record)

    def debug(self, event: str, **kw: Any) -> None:
        self._log(logging.DEBUG, event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._log(logging.INFO, event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._log(logging.WARNING, event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._log(logging.ERROR, event, **kw)

    def critical(self, event: str, **kw: Any) -> None:
        self._log(logging.CRITICAL, event, **kw)

    def exception(self, event: str, **kw: Any) -> None:
        self._log(logging.ERROR, event, **kw)


# ---------------------------------------------------------------------------
# Destinations
# ---------------------------------------------------------------------------


class StderrDestination:
    """Write to stderr. Default for dev."""

    def create_handler(self, formatter: logging.Formatter) -> logging.Handler:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        return handler

    def shutdown(self) -> None:
        pass


class VictoriaLogsDestination:
    """Batch and push JSON logs to VictoriaLogs via HTTP.

    VictoriaLogs accepts JSON lines at POST /insert/jsonline.
    Auto-indexes all fields.  Special fields: _time, _msg.

    Batches in memory, flushes on threshold or periodic timer.
    Uses urllib.request (stdlib) — no extra dependencies.
    """

    def __init__(self, config: ObservabilityConfig) -> None:
        self._endpoint = config.victorialogs_endpoint
        self._batch_size = config.victorialogs_batch_size
        self._flush_interval = config.victorialogs_flush_interval
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._closed = False

    def create_handler(self, formatter: logging.Formatter) -> logging.Handler:
        handler = _VictoriaLogsHandler(self, formatter)
        self._start_flush_timer()
        return handler

    def ship(self, line: str) -> None:
        """Accept a formatted log line into the buffer."""
        with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._batch_size:
                self._flush_locked()

    def _start_flush_timer(self) -> None:
        if self._closed:
            return
        self._timer = threading.Timer(self._flush_interval, self._periodic_flush)
        self._timer.daemon = True
        self._timer.start()

    def _periodic_flush(self) -> None:
        with self._lock:
            self._flush_locked()
        self._start_flush_timer()

    def _flush_locked(self) -> None:
        """Flush buffer to VictoriaLogs. Must be called with _lock held."""
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()
        payload = "\n".join(batch)
        try:
            req = urllib.request.Request(
                self._endpoint,
                data=payload.encode("utf-8"),
                headers={"Content-Type": "application/stream+json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # fire-and-forget — don't let log shipping break the app

    def shutdown(self) -> None:
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
        with self._lock:
            self._flush_locked()


class _VictoriaLogsHandler(logging.Handler):
    """Handler that ships formatted records to VictoriaLogs."""

    def __init__(
        self, destination: VictoriaLogsDestination, fmt: logging.Formatter
    ) -> None:
        super().__init__()
        self.setFormatter(fmt)
        self._destination = destination

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # VictoriaLogs expects _time and _msg as special fields.
            # If the formatter output is JSON, we can inject them.
            try:
                d = json.loads(msg)
                if "timestamp" in d and "_time" not in d:
                    d["_time"] = d["timestamp"]
                if "event" in d and "_msg" not in d:
                    d["_msg"] = d["event"]
                msg = json.dumps(d, default=str)
            except (json.JSONDecodeError, TypeError):
                pass  # non-JSON formatter, ship as-is
            self._destination.ship(msg)
        except Exception:
            self.handleError(record)


class JsonlFileDestination:
    """Append to a JSONL file. Loki/VictoriaLogs-ready."""

    def __init__(self, config: ObservabilityConfig) -> None:
        path = config.jsonl_path or "/tmp/qortex.jsonl"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def create_handler(self, formatter: logging.Formatter) -> logging.Handler:
        handler = logging.FileHandler(str(self._path), mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        return handler

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_FORMATTERS: dict[str, type] = {
    "structlog": StructlogFormatter,
    "stdlib": StdlibFormatter,
}

_DESTINATIONS: dict[str, type] = {
    "stderr": StderrDestination,
    "victorialogs": VictoriaLogsDestination,
    "jsonl": JsonlFileDestination,
}


def register_formatter(name: str, cls: type) -> None:
    """Register a custom log formatter. Call before configure()."""
    _FORMATTERS[name] = cls


def register_destination(name: str, cls: type) -> None:
    """Register a custom log destination. Call before configure()."""
    _DESTINATIONS[name] = cls


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_active_formatter: LogFormatter | None = None
_active_destination: LogDestination | None = None


def setup_logging(config: ObservabilityConfig) -> None:
    """Compose formatter × destination from config and wire to root logger.

    Template method:
        1. Instantiate formatter (how records are structured)
        2. Instantiate destination (where output goes)
        3. formatter.setup() → logging.Formatter
        4. destination.create_handler(formatter) → logging.Handler
        5. Attach handler to root logger
    """
    global _active_formatter, _active_destination

    # Resolve formatter
    formatter_cls = _FORMATTERS.get(config.log_formatter)
    if formatter_cls is None:
        raise ValueError(
            f"Unknown log formatter: {config.log_formatter!r}. "
            f"Available: {list(_FORMATTERS)}. "
            f"Register custom formatters with register_formatter()."
        )

    # Resolve destination
    dest_cls = _DESTINATIONS.get(config.log_destination)
    if dest_cls is None:
        raise ValueError(
            f"Unknown log destination: {config.log_destination!r}. "
            f"Available: {list(_DESTINATIONS)}. "
            f"Register custom destinations with register_destination()."
        )

    # Instantiate
    formatter = formatter_cls()
    # Destinations that need config get it via constructor
    if dest_cls in (VictoriaLogsDestination, JsonlFileDestination):
        destination = dest_cls(config)
    else:
        destination = dest_cls()

    # Compose: formatter.setup() → Formatter, destination.create_handler() → Handler
    log_formatter = formatter.setup(config)
    handler = destination.create_handler(log_formatter)

    # Wire to root logger — only replace our own handler, preserve external ones
    # (monitoring agents, pytest caplog, log aggregation, etc.)
    handler._qortex_managed = True  # type: ignore[attr-defined]
    root_logger = logging.getLogger()
    root_logger.handlers = [
        h for h in root_logger.handlers
        if not getattr(h, "_qortex_managed", False)
    ]
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    _active_formatter = formatter
    _active_destination = destination


def get_logger(name: str = "", **kwargs: Any) -> Any:
    """Get a logger from the active formatter.

    Returns a structlog BoundLogger (default) or a _StructuredStdlibLogger,
    both of which accept logger.info("event", key=value) kwargs.

    Falls back to a _StructuredStdlibLogger wrapper before setup_logging()
    is called, so structured kwargs work even pre-configuration.
    """
    if _active_formatter is not None:
        return _active_formatter.get_logger(name, **kwargs)
    # Fallback before configuration — wrap stdlib so kwargs don't crash
    return _StructuredStdlibLogger(logging.getLogger(name))


def shutdown_logging() -> None:
    """Flush and close the active destination. Call on process exit."""
    if _active_destination is not None:
        _active_destination.shutdown()
