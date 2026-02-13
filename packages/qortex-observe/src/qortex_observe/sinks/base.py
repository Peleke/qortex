"""LogSink protocol: strategy pattern for event output destinations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LogSink(Protocol):
    """Where structured event dicts get written."""

    def write(self, event_dict: dict) -> None: ...
