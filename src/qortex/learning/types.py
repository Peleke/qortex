"""Core types for the qortex learning module.

Arm = a candidate action (e.g. a prompt strategy, a retrieval config).
ArmState = posterior belief about an arm's reward distribution.
LearnerConfig = configuration for a Learner instance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class Arm:
    """A candidate action in the bandit."""

    id: str  # hierarchical: "type:category:id"
    metadata: dict = field(default_factory=dict)
    token_cost: int = 0  # estimated token cost for this arm


@dataclass
class ArmState:
    """Beta-Bernoulli posterior state for an arm."""

    alpha: float = 1.0  # successes + prior
    beta: float = 1.0  # failures + prior
    pulls: int = 0
    total_reward: float = 0.0
    last_updated: str = ""

    @property
    def mean(self) -> float:
        """Posterior mean: alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "pulls": self.pulls,
            "total_reward": self.total_reward,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ArmState:
        return cls(
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 1.0),
            pulls=d.get("pulls", 0),
            total_reward=d.get("total_reward", 0.0),
            last_updated=d.get("last_updated", ""),
        )


@dataclass
class ArmOutcome:
    """Result of using an arm."""

    arm_id: str
    reward: float  # 0.0 to 1.0
    outcome: str  # "accepted" | "rejected" | "partial" | custom
    context: dict = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class SelectionResult:
    """Result of selecting arms from the bandit."""

    selected: list[Arm]
    excluded: list[Arm]
    is_baseline: bool  # True if we used uniform random (exploration)
    scores: dict[str, float] = field(default_factory=dict)  # arm_id → sampled score
    token_budget: int = 0
    used_tokens: int = 0


@dataclass
class RunTrace:
    """Trace of a selection + observation cycle."""

    session_id: str
    learner: str
    selected_arms: list[str]
    outcomes: dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str = ""


@dataclass
class LearnerConfig:
    """Configuration for a Learner instance."""

    name: str
    baseline_rate: float = 0.1  # probability of uniform exploration
    seed_boost: float = 2.0  # prior boost for seed arms
    seed_arms: list[str] = field(default_factory=list)  # arms with boosted priors
    state_dir: str = ""  # override for state persistence path
    max_arms: int = 1000  # cap on tracked arms
    min_pulls: int = 0  # force-include arms with fewer than N observations


def context_hash(context: dict) -> str:
    """Deterministic hash for a context dict (for partitioning arm state)."""
    if not context:
        return "default"
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
