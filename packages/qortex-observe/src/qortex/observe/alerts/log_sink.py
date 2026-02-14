"""Log alert sink: fires alerts via the configured LogFormatter (default)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from qortex.observe.alerts.base import AlertRule
from qortex.observe.logging import get_logger

logger = get_logger("qortex.alerts")


class LogAlertSink:
    """Default: log alerts via the configured logging backend."""

    def fire(self, rule: AlertRule, event: Any) -> None:
        event_data = asdict(event) if hasattr(event, "__dataclass_fields__") else {}
        logger.warning(
            "alert.fired",
            rule=rule.name,
            severity=rule.severity,
            description=rule.description,
            event_type=type(event).__name__,
            **event_data,
        )
