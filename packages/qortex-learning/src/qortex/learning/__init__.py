"""qortex learning module: bandit-based learning for prompt optimization.

Public API:
    Learner          — Main class: select(), observe(), metrics(), posteriors()
    LearnerConfig    — Configuration for a Learner instance
    LearningStore    — Protocol for state persistence backends
    JsonLearningStore  — JSON file backend
    SqliteLearningStore — SQLite backend (default)
    ThompsonSampling — Beta-Bernoulli Thompson Sampling strategy
    BinaryReward     — Binary reward model (accepted=1, else=0)
    TernaryReward    — Ternary reward model (accepted=1, partial=0.5, rejected=0)
"""

from qortex.learning.learner import Learner
from qortex.learning.reward import BinaryReward, RewardModel, TernaryReward
from qortex.learning.store import JsonLearningStore, LearningStore, SqliteLearningStore
from qortex.learning.strategy import LearningStrategy, ThompsonSampling
from qortex.learning.types import (
    Arm,
    ArmOutcome,
    ArmState,
    LearnerConfig,
    RunTrace,
    SelectionResult,
    context_hash,
)

__all__ = [
    "Learner",
    "LearnerConfig",
    "LearningStore",
    "JsonLearningStore",
    "SqliteLearningStore",
    "LearningStrategy",
    "ThompsonSampling",
    "RewardModel",
    "BinaryReward",
    "TernaryReward",
    "Arm",
    "ArmOutcome",
    "ArmState",
    "SelectionResult",
    "RunTrace",
    "context_hash",
]
