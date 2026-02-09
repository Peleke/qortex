"""No-op alert sink: default when alerting is disabled."""

from __future__ import annotations

from typing import Any

from qortex.observability.alerts.base import AlertRule


class NoOpAlertSink:
    """Discards all alerts. Zero overhead."""

    def fire(self, rule: AlertRule, event: Any) -> None:
        pass
