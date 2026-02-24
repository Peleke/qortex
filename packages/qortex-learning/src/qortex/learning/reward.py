"""Reward models: map outcomes to float rewards.

RewardModel is the protocol. BinaryReward and TernaryReward are the
built-in implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardModel(Protocol):
    """Protocol for mapping outcomes to float rewards."""

    def compute(self, outcome: str) -> float:
        """Map an outcome string to a reward in [0, 1]."""
        ...


class BinaryReward:
    """Binary: accepted=1.0, everything else=0.0."""

    def compute(self, outcome: str) -> float:
        return 1.0 if outcome == "accepted" else 0.0


class TernaryReward:
    """Ternary: accepted=1.0, partial=0.5, rejected=0.0."""

    _rewards = {"accepted": 1.0, "partial": 0.5, "rejected": 0.0}

    def compute(self, outcome: str) -> float:
        return self._rewards.get(outcome, 0.0)
