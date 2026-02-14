"""Alert evaluation subscriber: checks events against alert rules.

Defaults to LogAlertSink (structlog warning). Designed for Alertmanager later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qortex.observe.alerts.base import AlertSink
from qortex.observe.alerts.log_sink import LogAlertSink
from qortex.observe.alerts.noop_sink import NoOpAlertSink
from qortex.observe.alerts.rules import BUILTIN_RULES
from qortex.observe.config import ObservabilityConfig
from qortex.observe.events import (
    BufferFlushed,
    FactorDriftSnapshot,
    PPRConverged,
    PPRDiverged,
    QueryCompleted,
    QueryFailed,
)
from qortex.observe.linker import QortexEventLinker

# Events that alert rules can match against
_ALERTABLE_EVENTS = (
    QueryCompleted,
    QueryFailed,
    PPRConverged,
    PPRDiverged,
    FactorDriftSnapshot,
    BufferFlushed,
)


def register_alert_subscriber(config: ObservabilityConfig) -> None:
    """Register alert evaluation on alertable events."""
    sink: AlertSink
    if config.alert_enabled:
        sink = LogAlertSink()
    else:
        sink = NoOpAlertSink()

    rules = list(BUILTIN_RULES)

    @QortexEventLinker.on(*_ALERTABLE_EVENTS)
    def _evaluate_alerts(event: Any) -> None:
        now = datetime.now(UTC)
        for rule in rules:
            try:
                if not rule.condition(event):
                    continue
            except Exception:
                continue

            # Cooldown check
            if rule._last_fired is not None:
                elapsed = now - rule._last_fired
                if elapsed < rule.cooldown:
                    continue

            rule._last_fired = now
            sink.fire(rule, event)
