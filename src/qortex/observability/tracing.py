"""@traced decorator: automatic span creation + overhead timing.

Creates OTel spans via start_as_current_span(). Nested @traced calls
form a parent-child tree through contextvars. The OverheadTimer tracks
wall time minus external call time (DisjointIntervals algorithm from
TensorZero).

Usage:
    @traced("retrieval.query")
    def retrieve(self, query: str) -> RetrievalResult: ...

    @traced("cypher.execute", external=True)
    def _run(self, cypher: str) -> list[dict]: ...
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ── Context vars for propagation ───────────────────────────────────
_overhead_timer: contextvars.ContextVar[OverheadTimer | None] = contextvars.ContextVar(
    "_overhead_timer", default=None
)
_config_hash: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_config_hash", default=None
)


@dataclass
class OverheadTimer:
    """Track overhead = wall time - external call time, handling concurrency.

    Port of TensorZero's DisjointIntervals algorithm. External intervals
    are merged (handling overlap from concurrent calls) before subtraction.
    """

    _start: float = field(default_factory=time.monotonic)
    _external_intervals: list[tuple[float, float]] = field(default_factory=list)

    def record_external(self, start: float, end: float) -> None:
        """Record an external call interval."""
        self._external_intervals.append((start, end))

    def overhead_seconds(self) -> float:
        """Compute overhead = total wall time - external time."""
        total = time.monotonic() - self._start
        merged = self._merge_intervals(self._external_intervals)
        external = sum(end - start for start, end in merged)
        return max(total - external, 0.0)

    @staticmethod
    def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Merge overlapping intervals (handles concurrent external calls)."""
        if not intervals:
            return []
        sorted_ivs = sorted(intervals)
        merged = [sorted_ivs[0]]
        for start, end in sorted_ivs[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged


def traced(name: str, *, external: bool = False) -> Callable[[F], F]:
    """Decorator: wrap function in an OTel span, track overhead if external.

    Args:
        name: Span name (e.g. "retrieval.query", "cypher.execute").
        external: If True, the span's duration is recorded as external time
                  on the parent's OverheadTimer. Use for DB calls, LLM calls,
                  and any I/O where we're waiting, not computing.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                from opentelemetry import trace
            except ImportError:
                return fn(*args, **kwargs)

            tracer = trace.get_tracer("qortex")
            parent_timer = _overhead_timer.get()

            with tracer.start_as_current_span(name) as span:
                # Attach config snapshot hash if available
                config_h = _config_hash.get()
                if config_h:
                    span.set_attribute("config.snapshot_hash", config_h)

                # If this is a top-level traced call, create an OverheadTimer
                if parent_timer is None:
                    timer = OverheadTimer()
                    token = _overhead_timer.set(timer)
                else:
                    timer = None
                    token = None

                t0 = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(exc).__name__)
                    span.record_exception(exc)
                    raise
                finally:
                    t1 = time.monotonic()
                    # Record external interval on parent's timer
                    if external and parent_timer is not None:
                        parent_timer.record_external(t0, t1)
                    # If we own the timer, record overhead on the span
                    if timer is not None and token is not None:
                        overhead = timer.overhead_seconds()
                        span.set_attribute("qortex.overhead_seconds", overhead)
                        span.set_attribute("qortex.total_seconds", t1 - timer._start)
                        _overhead_timer.reset(token)

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                from opentelemetry import trace
            except ImportError:
                return await fn(*args, **kwargs)

            tracer = trace.get_tracer("qortex")
            parent_timer = _overhead_timer.get()

            with tracer.start_as_current_span(name) as span:
                config_h = _config_hash.get()
                if config_h:
                    span.set_attribute("config.snapshot_hash", config_h)

                if parent_timer is None:
                    timer = OverheadTimer()
                    token = _overhead_timer.set(timer)
                else:
                    timer = None
                    token = None

                t0 = time.monotonic()
                try:
                    result = await fn(*args, **kwargs)
                    return result
                except Exception as exc:
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(exc).__name__)
                    span.record_exception(exc)
                    raise
                finally:
                    t1 = time.monotonic()
                    if external and parent_timer is not None:
                        parent_timer.record_external(t0, t1)
                    if timer is not None and token is not None:
                        overhead = timer.overhead_seconds()
                        span.set_attribute("qortex.overhead_seconds", overhead)
                        span.set_attribute("qortex.total_seconds", t1 - timer._start)
                        _overhead_timer.reset(token)

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


def get_overhead_timer() -> OverheadTimer | None:
    """Get the current overhead timer (for passing overhead to events)."""
    return _overhead_timer.get()


# ── Selective Span Processor ──────────────────────────────────────────

class SelectiveSpanProcessor:
    """Export only interesting spans: errors, slow queries, or sampled.

    Wraps a BatchSpanProcessor. Spans are exported if ANY of:
      - error=True attribute set
      - duration exceeds latency_threshold_ms
      - random sample at sample_rate

    This reduces span volume from Cypher instrumentation (5-30 spans
    per query) while preserving all diagnostic spans.
    """

    def __init__(
        self,
        exporter: Any,
        sample_rate: float = 0.1,
        latency_threshold_ms: float = 100.0,
    ):
        import random

        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        self._inner = BatchSpanProcessor(exporter)
        self._sample_rate = sample_rate
        self._latency_threshold_ms = latency_threshold_ms
        self._random = random

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        self._inner.on_start(span, parent_context)

    def on_end(self, span: Any) -> None:
        attrs = span.attributes or {}
        # Always export error spans
        if attrs.get("error"):
            self._inner.on_end(span)
            return
        # Always export slow spans
        if span.end_time is not None and span.start_time is not None:
            duration_ms = (span.end_time - span.start_time) / 1_000_000
            if duration_ms > self._latency_threshold_ms:
                self._inner.on_end(span)
                return
        # Sample the rest
        if self._random.random() < self._sample_rate:
            self._inner.on_end(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
