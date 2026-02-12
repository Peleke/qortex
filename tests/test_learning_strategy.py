"""Tests for learning strategy: ThompsonSampling select/update, seed handling, baseline rate."""

from __future__ import annotations

import random

import pytest

from qortex.learning.strategy import ThompsonSampling
from qortex.learning.types import Arm, ArmState, LearnerConfig


@pytest.fixture
def strategy():
    return ThompsonSampling()


@pytest.fixture
def candidates():
    return [
        Arm(id="arm:a", token_cost=100),
        Arm(id="arm:b", token_cost=200),
        Arm(id="arm:c", token_cost=150),
        Arm(id="arm:d", token_cost=50),
    ]


@pytest.fixture
def config():
    return LearnerConfig(name="test", baseline_rate=0.0)


class TestThompsonSamplingSelect:
    def test_select_k_arms(self, strategy, candidates, config):
        states = {a.id: ArmState() for a in candidates}
        result = strategy.select(candidates, states, k=2, config=config)

        assert len(result.selected) == 2
        assert len(result.excluded) == 2
        assert not result.is_baseline

    def test_select_all(self, strategy, candidates, config):
        states = {a.id: ArmState() for a in candidates}
        result = strategy.select(candidates, states, k=10, config=config)

        assert len(result.selected) == 4
        assert len(result.excluded) == 0

    def test_select_single(self, strategy, candidates, config):
        states = {a.id: ArmState() for a in candidates}
        result = strategy.select(candidates, states, k=1, config=config)

        assert len(result.selected) == 1
        assert len(result.excluded) == 3

    def test_scores_populated(self, strategy, candidates, config):
        states = {a.id: ArmState() for a in candidates}
        result = strategy.select(candidates, states, k=2, config=config)

        for arm in candidates:
            assert arm.id in result.scores
            assert 0.0 <= result.scores[arm.id] <= 1.0

    def test_token_budget_respected(self, strategy, candidates, config):
        states = {a.id: ArmState() for a in candidates}
        # Budget of 250 should fit at most arm:d(50) + arm:a(100) + arm:c(150) = 300
        # or arm:d(50) + arm:a(100) = 150, etc.
        result = strategy.select(
            candidates, states, k=4, config=config, token_budget=250
        )

        assert result.used_tokens <= 250
        assert result.token_budget == 250

    def test_high_alpha_arm_preferred(self, strategy, candidates, config):
        """Arm with high alpha (many successes) should be selected more often."""
        states = {a.id: ArmState() for a in candidates}
        states["arm:a"] = ArmState(alpha=100.0, beta=1.0, pulls=100)

        selected_count = 0
        for _ in range(100):
            result = strategy.select(candidates, states, k=1, config=config)
            if result.selected[0].id == "arm:a":
                selected_count += 1

        # arm:a should be selected most of the time with alpha=100, beta=1
        assert selected_count > 80


class TestThompsonSamplingBaseline:
    def test_baseline_uniform_random(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=1.0)
        states = {a.id: ArmState() for a in candidates}

        result = strategy.select(candidates, states, k=2, config=config)
        assert result.is_baseline

    def test_no_baseline_when_rate_zero(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=0.0)
        states = {a.id: ArmState() for a in candidates}

        result = strategy.select(candidates, states, k=2, config=config)
        assert not result.is_baseline

    def test_baseline_rate_probabilistic(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=0.5)
        states = {a.id: ArmState() for a in candidates}

        baseline_count = 0
        for _ in range(200):
            result = strategy.select(candidates, states, k=1, config=config)
            if result.is_baseline:
                baseline_count += 1

        # Should be roughly 50% baseline ± some variance
        assert 60 < baseline_count < 140


class TestThompsonSamplingMinPulls:
    """min_pulls forces under-explored arms into the selection set."""

    def test_min_pulls_forces_zero_obs_arms(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=0.0, min_pulls=5)
        states = {a.id: ArmState(pulls=0) for a in candidates}
        result = strategy.select(candidates, states, k=1, config=config)

        # All 4 arms have 0 pulls (< 5), so all must be force-included
        assert len(result.selected) == 4
        assert len(result.excluded) == 0

    def test_min_pulls_mixed_experience(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=0.0, min_pulls=3)
        states = {
            "arm:a": ArmState(pulls=0),
            "arm:b": ArmState(pulls=10, alpha=10.0, beta=1.0),
            "arm:c": ArmState(pulls=2),
            "arm:d": ArmState(pulls=5, alpha=5.0, beta=1.0),
        }
        # k=3: 2 forced + 1 TS pick from eligible
        result = strategy.select(candidates, states, k=3, config=config)

        selected_ids = {a.id for a in result.selected}
        # arm:a (0 pulls) and arm:c (2 pulls) must be force-included
        assert "arm:a" in selected_ids
        assert "arm:c" in selected_ids
        assert len(result.selected) == 3  # 2 forced + 1 TS pick
        assert len(result.excluded) == 1

    def test_min_pulls_zero_disables(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=0.0, min_pulls=0)
        states = {a.id: ArmState(pulls=0) for a in candidates}
        result = strategy.select(candidates, states, k=1, config=config)

        # min_pulls=0 means no forcing
        assert len(result.selected) == 1

    def test_min_pulls_with_token_budget(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=0.0, min_pulls=5)
        states = {a.id: ArmState(pulls=0) for a in candidates}
        # Forced arms still consume token budget
        result = strategy.select(
            candidates, states, k=4, config=config, token_budget=8000
        )

        # All arms forced (token_cost 100+200+150+50=500 < 8000)
        assert len(result.selected) == 4
        assert result.used_tokens == 500

    def test_min_pulls_with_baseline(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=1.0, min_pulls=5)
        states = {a.id: ArmState(pulls=0) for a in candidates}
        result = strategy.select(candidates, states, k=1, config=config)

        # Even in baseline mode, all under-explored arms force-included
        assert len(result.selected) == 4
        assert result.is_baseline

    def test_min_pulls_satisfied_arms_not_forced(self, strategy, candidates):
        config = LearnerConfig(name="test", baseline_rate=0.0, min_pulls=5)
        states = {a.id: ArmState(pulls=10, alpha=5.0, beta=5.0) for a in candidates}
        result = strategy.select(candidates, states, k=1, config=config)

        # All arms have 10 pulls (>= 5), normal TS behavior
        assert len(result.selected) == 1
        assert len(result.excluded) == 3


class TestThompsonSamplingUpdate:
    def test_update_success(self, strategy):
        state = ArmState(alpha=1.0, beta=1.0)
        new_state = strategy.update("arm:a", 1.0, state)

        assert new_state.alpha == 2.0
        assert new_state.beta == 1.0
        assert new_state.pulls == 1
        assert new_state.total_reward == 1.0

    def test_update_failure(self, strategy):
        state = ArmState(alpha=1.0, beta=1.0)
        new_state = strategy.update("arm:a", 0.0, state)

        assert new_state.alpha == 1.0
        assert new_state.beta == 2.0
        assert new_state.pulls == 1
        assert new_state.total_reward == 0.0

    def test_update_partial(self, strategy):
        state = ArmState(alpha=1.0, beta=1.0)
        new_state = strategy.update("arm:a", 0.5, state)

        assert new_state.alpha == 1.5
        assert new_state.beta == 1.5
        assert new_state.pulls == 1
        assert new_state.total_reward == 0.5

    def test_update_accumulates(self, strategy):
        state = ArmState(alpha=1.0, beta=1.0)
        state = strategy.update("arm:a", 1.0, state)
        state = strategy.update("arm:a", 1.0, state)
        state = strategy.update("arm:a", 0.0, state)

        assert state.alpha == 3.0
        assert state.beta == 2.0
        assert state.pulls == 3
        assert state.total_reward == 2.0
