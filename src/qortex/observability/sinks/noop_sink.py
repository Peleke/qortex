"""No-op sink: zero overhead when observability is disabled."""

from __future__ import annotations


class NoOpSink:
    """Discards all events. Zero overhead."""

    def write(self, event_dict: dict) -> None:
        pass
