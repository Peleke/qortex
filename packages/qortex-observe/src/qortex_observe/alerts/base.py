"""Alert primitives: AlertRule and AlertSink protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class AlertRule:
    """A condition that fires an alert when matched."""

    name: str
    description: str
    severity: str  # "info" | "warning" | "critical"
    condition: Callable[[Any], bool]
    cooldown: timedelta = timedelta(minutes=5)
    _last_fired: datetime | None = field(default=None, repr=False)


@runtime_checkable
class AlertSink(Protocol):
    """Where alert notifications go."""

    def fire(self, rule: AlertRule, event: Any) -> None: ...
