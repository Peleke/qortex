"""Checkpoint manager for snapshot/restore operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from qortex.core.backend import GraphBackend


@dataclass
class Checkpoint:
    """A saved state of the knowledge graph."""

    id: str
    name: str
    timestamp: datetime
    domains: list[str]  # Which domains were checkpointed

    # State hashes for integrity
    projection_hash: str | None = None  # Hash of flat rules at this point

    # Optional metrics from experiment engine
    metrics: dict = field(default_factory=dict)


@dataclass
class CheckpointDiff:
    """Difference between two checkpoints."""

    added_concepts: int
    removed_concepts: int
    added_rules: int
    removed_rules: int
    changed_edges: int


class CheckpointManager:
    """Manage checkpoints for rollback capability.

    Critical for validation loop:
    - Checkpoint before activating new rules
    - If experiment engine shows degradation, rollback
    """

    def __init__(self, backend: GraphBackend):
        self.backend = backend

    def create(
        self,
        name: str,
        domains: list[str] | None = None,
        metrics: dict | None = None,
    ) -> Checkpoint:
        """Create a named checkpoint.

        Args:
            name: Human-readable checkpoint name
            domains: Domains to checkpoint (None = all)
            metrics: Optional experiment engine metrics at this point
        """
        checkpoint_id = self.backend.checkpoint(name, domains)

        return Checkpoint(
            id=checkpoint_id,
            name=name,
            timestamp=datetime.utcnow(),
            domains=domains or [d.name for d in self.backend.list_domains()],
            metrics=metrics or {},
        )

    def restore(self, checkpoint: Checkpoint | str) -> None:
        """Restore to a checkpoint state.

        Args:
            checkpoint: Checkpoint object or ID string
        """
        checkpoint_id = checkpoint.id if isinstance(checkpoint, Checkpoint) else checkpoint
        self.backend.restore(checkpoint_id)

    def list(self) -> list[Checkpoint]:
        """List all available checkpoints."""
        raw = self.backend.list_checkpoints()
        return [
            Checkpoint(
                id=c["id"],
                name=c["name"],
                timestamp=c["timestamp"],
                domains=c.get("domains", []),
                metrics=c.get("metrics", {}),
            )
            for c in raw
        ]

    def diff(self, a: Checkpoint | str, b: Checkpoint | str) -> CheckpointDiff:
        """Compare two checkpoints.

        TODO M4: Implement checkpoint diffing
        """
        raise NotImplementedError("M4: Implement checkpoint diff")

    def auto_rollback_on_degrade(
        self,
        metric_name: str,
        threshold: float,
        checkpoint: Checkpoint | str,
    ) -> bool:
        """Automatically rollback if metric degrades.

        Args:
            metric_name: Metric to monitor (e.g., "rule_effectiveness")
            threshold: Minimum acceptable value
            checkpoint: Checkpoint to restore if threshold breached

        Returns:
            True if rollback occurred
        """
        # TODO M4: Implement with experiment engine integration
        raise NotImplementedError("M4: Implement auto-rollback")
