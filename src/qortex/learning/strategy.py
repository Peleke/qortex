"""Learning strategies: protocols and implementations.

LearningStrategy is the protocol. ThompsonSampling is the default
Beta-Bernoulli implementation.
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from qortex.learning.types import Arm, ArmState, LearnerConfig, SelectionResult


@runtime_checkable
class LearningStrategy(Protocol):
    """Protocol for bandit selection strategies."""

    def select(
        self,
        candidates: list[Arm],
        states: dict[str, ArmState],
        k: int,
        config: LearnerConfig,
        token_budget: int = 0,
    ) -> SelectionResult:
        """Select k arms from candidates using the strategy."""
        ...

    def update(
        self,
        arm_id: str,
        reward: float,
        state: ArmState,
    ) -> ArmState:
        """Update posterior for an arm given observed reward."""
        ...


class ThompsonSampling:
    """Beta-Bernoulli Thompson Sampling.

    Each arm has a Beta(alpha, beta) posterior. Selection samples from
    each arm's posterior and picks the top-k. Observation updates
    alpha (reward) and beta (1 - reward).
    """

    def select(
        self,
        candidates: list[Arm],
        states: dict[str, ArmState],
        k: int,
        config: LearnerConfig,
        token_budget: int = 0,
    ) -> SelectionResult:
        # Baseline exploration: uniform random with probability baseline_rate
        is_baseline = random.random() < config.baseline_rate

        if is_baseline:
            # Uniform random selection (exploration)
            shuffled = list(candidates)
            random.shuffle(shuffled)

            if token_budget > 0:
                selected: list[Arm] = []
                used = 0
                for arm in shuffled:
                    if len(selected) >= k:
                        break
                    if used + arm.token_cost <= token_budget:
                        selected.append(arm)
                        used += arm.token_cost
                excluded = [a for a in candidates if a not in selected]
            else:
                selected = shuffled[:k]
                excluded = [a for a in candidates if a not in selected]
                used = sum(a.token_cost for a in selected)

            return SelectionResult(
                selected=selected,
                excluded=excluded,
                is_baseline=True,
                scores={a.id: 0.5 for a in selected},
                token_budget=token_budget,
                used_tokens=used,
            )

        # Thompson sampling: sample from each arm's posterior
        scores: dict[str, float] = {}
        for arm in candidates:
            state = states.get(arm.id, ArmState())
            scores[arm.id] = random.betavariate(state.alpha, state.beta)

        # Sort by sampled score (descending)
        ranked = sorted(candidates, key=lambda a: scores[a.id], reverse=True)

        # If token_budget > 0, respect it
        if token_budget > 0:
            selected: list[Arm] = []
            used = 0
            for arm in ranked:
                if len(selected) >= k:
                    break
                if used + arm.token_cost <= token_budget:
                    selected.append(arm)
                    used += arm.token_cost
            excluded = [a for a in candidates if a not in selected]
            return SelectionResult(
                selected=selected,
                excluded=excluded,
                is_baseline=False,
                scores=scores,
                token_budget=token_budget,
                used_tokens=used,
            )

        selected = ranked[:k]
        excluded = ranked[k:]
        used_tokens = sum(a.token_cost for a in selected)

        return SelectionResult(
            selected=selected,
            excluded=excluded,
            is_baseline=False,
            scores=scores,
            token_budget=token_budget,
            used_tokens=used_tokens,
        )

    def update(
        self,
        arm_id: str,
        reward: float,
        state: ArmState,
    ) -> ArmState:
        """Update Beta posterior: alpha += reward, beta += (1 - reward)."""
        return ArmState(
            alpha=state.alpha + reward,
            beta=state.beta + (1.0 - reward),
            pulls=state.pulls + 1,
            total_reward=state.total_reward + reward,
            last_updated=state.last_updated,  # caller sets timestamp
        )
