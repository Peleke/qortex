"""Built-in alert rules for qortex observability."""

from __future__ import annotations

from qortex.observability.alerts.base import AlertRule
from qortex.observability.events import FactorDriftSnapshot, PPRDiverged

BUILTIN_RULES: list[AlertRule] = [
    AlertRule(
        name="ppr_divergence",
        description="PPR failed to converge within max iterations",
        severity="warning",
        condition=lambda e: isinstance(e, PPRDiverged),
    ),
    AlertRule(
        name="factor_drift_high",
        description="Factor distribution entropy dropping (over-specialization)",
        severity="info",
        condition=lambda e: isinstance(e, FactorDriftSnapshot) and e.entropy < 0.5,
    ),
]
