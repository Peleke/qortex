"""Teleportation factors: feedback-driven PPR personalization.

In standard PPR, all seed nodes get equal teleportation probability.
Teleportation factors bias the random walk toward nodes that produced
good outcomes in the past.

Lifecycle:
    1. Query: GraphRAGAdapter reads factors → biased PPR seed weights
    2. Feedback: outcomes ("accepted"/"rejected") update factors
    3. Persist: factors survive process restarts via JSONL file

This is qortex's unique advantage — no other framework does this.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qortex.observability import emit
from qortex.observability.events import (
    FactorDriftSnapshot,
    FactorsLoaded,
    FactorsPersisted,
    FactorUpdated,
)
from qortex.observability.logging import get_logger

logger = get_logger(__name__)

# Outcome weights: how much each outcome shifts the factor
_OUTCOME_WEIGHTS: dict[str, float] = {
    "accepted": 0.1,  # boost: this node was useful
    "rejected": -0.05,  # penalize: this node was not useful (smaller magnitude — be conservative)
    "partial": 0.03,  # slight boost: partially useful
}

# Factor bounds: prevent runaway amplification or suppression
_MIN_FACTOR = 0.1
_MAX_FACTOR = 5.0
_DEFAULT_FACTOR = 1.0


@dataclass
class FactorUpdate:
    """A single factor update from a feedback event."""

    node_id: str
    outcome: str
    delta: float
    new_factor: float
    query_id: str


@dataclass
class TeleportationFactors:
    """Feedback-driven teleportation weights for PPR.

    Each node_id maps to a factor (default 1.0). During PPR, seed node
    weights are multiplied by their factor before normalization.

    Higher factor = more teleportation probability = more likely to be
    activated in future queries.
    """

    factors: dict[str, float] = field(default_factory=dict)
    _persistence_path: Path | None = field(default=None, repr=False)

    # Lifecycle hooks — callables invoked at key moments.
    # Consumers can register hooks for event systems, logging, metrics.
    _hooks: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "on_update": [],  # Called after each factor update: (FactorUpdate) -> None
            "on_persist": [],  # Called after factors are written to disk: (Path) -> None
            "on_load": [],  # Called after factors are loaded from disk: (int) -> None
        },
        repr=False,
    )

    def get(self, node_id: str) -> float:
        """Get the teleportation factor for a node (default 1.0)."""
        return self.factors.get(node_id, _DEFAULT_FACTOR)

    def weight_seeds(self, seed_ids: list[str]) -> dict[str, float]:
        """Apply factors to seed nodes, returning normalized weights.

        Used by GraphRAGAdapter to bias PPR personalization vector.
        """
        if not seed_ids:
            return {}

        raw = {nid: self.get(nid) for nid in seed_ids}
        total = sum(raw.values())
        if total == 0:
            return {nid: 1.0 / len(seed_ids) for nid in seed_ids}
        return {nid: w / total for nid, w in raw.items()}

    def update(self, query_id: str, outcomes: dict[str, str]) -> list[FactorUpdate]:
        """Update factors from feedback outcomes.

        Args:
            query_id: The query that produced these results.
            outcomes: Mapping of node_id → "accepted" | "rejected" | "partial".

        Returns:
            List of FactorUpdate records (for logging/hooks).
        """
        updates: list[FactorUpdate] = []

        for node_id, outcome in outcomes.items():
            delta = _OUTCOME_WEIGHTS.get(outcome, 0.0)
            if delta == 0.0:
                continue

            old = self.factors.get(node_id, _DEFAULT_FACTOR)
            new = max(_MIN_FACTOR, min(_MAX_FACTOR, old + delta))
            self.factors[node_id] = new

            update = FactorUpdate(
                node_id=node_id,
                outcome=outcome,
                delta=delta,
                new_factor=new,
                query_id=query_id,
            )
            updates.append(update)

            # Emit observability event
            emit(FactorUpdated(
                node_id=node_id,
                query_id=query_id,
                outcome=outcome,
                old_factor=old,
                new_factor=new,
                delta=delta,
                clamped=(new == _MIN_FACTOR or new == _MAX_FACTOR),
            ))

            # Fire legacy hooks (backward compat)
            for hook in self._hooks.get("on_update", []):
                try:
                    hook(update)
                except Exception:
                    logger.debug("factor.hook.failed", hook_event="on_update")

        # Emit drift snapshot after batch update
        if self.factors:
            vals = list(self.factors.values())
            n = len(vals)
            total = sum(vals)
            if total > 0:
                probs = [v / total for v in vals]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            else:
                entropy = 0.0
            emit(FactorDriftSnapshot(
                count=n,
                mean=sum(vals) / n,
                min_val=min(vals),
                max_val=max(vals),
                boosted=sum(1 for v in vals if v > _DEFAULT_FACTOR),
                penalized=sum(1 for v in vals if v < _DEFAULT_FACTOR),
                entropy=entropy,
            ))

        return updates

    def register_hook(self, event: str, callback: Any) -> None:
        """Register a lifecycle hook.

        Events:
            on_update:  Called after each factor update with FactorUpdate arg
            on_persist: Called after factors are written to disk with Path arg
            on_load:    Called after factors are loaded from disk with count arg
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event: {event}. Valid: {list(self._hooks)}")
        self._hooks[event].append(callback)

    def persist(self, path: Path | None = None) -> Path | None:
        """Write factors to a JSONL file. Returns the path written to."""
        target = path or self._persistence_path
        if target is None:
            return None

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.factors, indent=2))

        emit(FactorsPersisted(
            path=str(target),
            count=len(self.factors),
            timestamp=datetime.now(UTC).isoformat(),
        ))

        for hook in self._hooks.get("on_persist", []):
            try:
                hook(target)
            except Exception:
                logger.debug("factor.hook.failed", hook_event="on_persist")

        return target

    @classmethod
    def load(cls, path: Path) -> TeleportationFactors:
        """Load factors from a JSON file."""
        path = Path(path)
        if not path.exists():
            return cls(_persistence_path=path)

        try:
            data = json.loads(path.read_text())
            instance = cls(factors=data, _persistence_path=path)

            emit(FactorsLoaded(
                path=str(path),
                count=len(data),
                timestamp=datetime.now(UTC).isoformat(),
            ))

            for hook in instance._hooks.get("on_load", []):
                try:
                    hook(len(data))
                except Exception:
                    logger.debug("factor.hook.failed", hook_event="on_load")
            return instance
        except (json.JSONDecodeError, TypeError):
            logger.warning("factors.load.failed", path=str(path))
            return cls(_persistence_path=path)

    def summary(self) -> dict[str, Any]:
        """Summary stats for monitoring/logging."""
        if not self.factors:
            return {
                "count": 0,
                "mean": _DEFAULT_FACTOR,
                "min": _DEFAULT_FACTOR,
                "max": _DEFAULT_FACTOR,
            }

        vals = list(self.factors.values())
        return {
            "count": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "boosted": sum(1 for v in vals if v > _DEFAULT_FACTOR),
            "penalized": sum(1 for v in vals if v < _DEFAULT_FACTOR),
        }
