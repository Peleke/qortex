"""Tests for Learner lifecycle, state persistence, observe→select loop."""

from __future__ import annotations

import pytest

from qortex.learning.learner import Learner
from qortex.learning.store import JsonLearningStore, SqliteLearningStore
from qortex.learning.types import Arm, ArmOutcome, LearnerConfig
from qortex_observe import reset as obs_reset


@pytest.fixture(autouse=True)
def _reset_observability():
    obs_reset()
    yield
    obs_reset()


@pytest.fixture
def state_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture(params=["sqlite", "json"], ids=["sqlite", "json"])
def store_backend(request, state_dir):
    """Return a store factory so each test gets a fresh store."""
    if request.param == "sqlite":
        return lambda name: SqliteLearningStore(name, state_dir)
    return lambda name: JsonLearningStore(name, state_dir)


@pytest.fixture
def config(state_dir):
    return LearnerConfig(
        name="test-learner",
        baseline_rate=0.0,  # deterministic for testing
        state_dir=state_dir,
    )


@pytest.fixture
def learner(config, store_backend):
    return Learner(config, store=store_backend(config.name))


class TestLearnerSelect:
    def test_select_returns_arms(self, learner):
        candidates = [Arm(id="a"), Arm(id="b"), Arm(id="c")]
        result = learner.select(candidates, k=2)

        assert len(result.selected) == 2
        assert len(result.excluded) == 1
        assert not result.is_baseline

    def test_select_with_context(self, learner):
        candidates = [Arm(id="a"), Arm(id="b")]
        result = learner.select(candidates, context={"task": "testing"}, k=1)

        assert len(result.selected) == 1


class TestLearnerObserve:
    def test_observe_updates_posterior(self, learner):
        state = learner.observe(
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted")
        )

        assert state.alpha == 2.0
        assert state.beta == 1.0
        assert state.pulls == 1

    def test_observe_uses_reward_model(self, learner):
        # TernaryReward: "partial" → 0.5
        state = learner.observe(
            ArmOutcome(arm_id="arm:b", reward=0.0, outcome="partial")
        )

        assert state.alpha == 1.5
        assert state.beta == 1.5
        assert state.pulls == 1

    def test_observe_rejected(self, learner):
        state = learner.observe(
            ArmOutcome(arm_id="arm:c", reward=0.0, outcome="rejected")
        )

        assert state.alpha == 1.0
        assert state.beta == 2.0

    def test_observe_multiple(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        state = learner.observe(
            ArmOutcome(arm_id="arm:a", reward=0.0, outcome="rejected")
        )

        assert state.pulls == 3
        assert state.alpha == 3.0
        assert state.beta == 2.0


class TestLearnerPersistence:
    def test_state_persists_to_file(self, config, state_dir, store_backend):
        learner = Learner(config, store=store_backend(config.name))
        learner.observe(ArmOutcome(arm_id="arm:x", reward=1.0, outcome="accepted"))

        # Verify state is readable via store
        state = learner.store.get("arm:x")
        assert state.alpha == 2.0

    def test_state_reloaded_on_new_learner(self, config, state_dir, store_backend):
        learner1 = Learner(config, store=store_backend(config.name))
        learner1.observe(ArmOutcome(arm_id="arm:y", reward=1.0, outcome="accepted"))

        # Create new learner with same backend — should reload state
        learner2 = Learner(config, store=store_backend(config.name))
        posteriors = learner2.posteriors()

        assert "arm:y" in posteriors
        assert posteriors["arm:y"]["alpha"] == 2.0


class TestLearnerPosteriors:
    def test_posteriors_empty(self, learner):
        p = learner.posteriors()
        assert p == {}

    def test_posteriors_after_observe(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:b", reward=0.0, outcome="rejected"))

        p = learner.posteriors()
        assert "arm:a" in p
        assert "arm:b" in p
        assert p["arm:a"]["mean"] > p["arm:b"]["mean"]

    def test_posteriors_filter_by_arm_ids(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:b", reward=0.0, outcome="rejected"))

        p = learner.posteriors(arm_ids=["arm:a"])
        assert "arm:a" in p
        assert "arm:b" not in p


class TestLearnerMetrics:
    def test_metrics_empty(self, learner):
        m = learner.metrics()
        assert m["total_pulls"] == 0
        assert m["accuracy"] == 0.0

    def test_metrics_after_observations(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:a", reward=0.0, outcome="rejected"))

        m = learner.metrics()
        assert m["total_pulls"] == 3
        assert m["total_reward"] == 2.0


class TestLearnerSeedBoost:
    def test_seed_arms_get_boosted_prior(self, state_dir, store_backend):
        config = LearnerConfig(
            name="seed-test",
            seed_arms=["arm:seed"],
            seed_boost=5.0,
            state_dir=state_dir,
        )
        learner = Learner(config, store=store_backend("seed-test"))

        p = learner.posteriors()
        assert "arm:seed" in p
        assert p["arm:seed"]["alpha"] == 5.0


class TestLearnerSessions:
    def test_session_lifecycle(self, learner):
        sid = learner.session_start("test-session")
        assert sid

        summary = learner.session_end(sid)
        assert summary["session_id"] == sid
        assert summary["started_at"]
        assert summary["ended_at"]

    def test_session_not_found(self, learner):
        result = learner.session_end("nonexistent")
        assert "error" in result


class TestContextPartitioning:
    def test_different_contexts_independent(self, learner):
        ctx_a = {"task": "typing"}
        ctx_b = {"task": "linting"}

        learner.observe(
            ArmOutcome(arm_id="arm:x", reward=1.0, outcome="accepted"),
            context=ctx_a,
        )
        learner.observe(
            ArmOutcome(arm_id="arm:x", reward=0.0, outcome="rejected"),
            context=ctx_b,
        )

        p_a = learner.posteriors(context=ctx_a)
        p_b = learner.posteriors(context=ctx_b)

        assert p_a["arm:x"]["alpha"] == 2.0  # 1 + 1.0 reward
        assert p_b["arm:x"]["alpha"] == 1.0  # 1 + 0.0 reward


class TestApplyCreditDeltas:
    """Tests for Learner.apply_credit_deltas() — causal credit integration."""

    def test_basic_delta_application(self, learner):
        deltas = {
            "concept:a": {"alpha_delta": 0.5, "beta_delta": 0.0},
            "concept:b": {"alpha_delta": 0.0, "beta_delta": 0.3},
        }
        results = learner.apply_credit_deltas(deltas)

        assert results["concept:a"].alpha == pytest.approx(1.5)  # 1.0 + 0.5
        assert results["concept:a"].beta == pytest.approx(1.0)
        assert results["concept:b"].alpha == pytest.approx(1.0)
        assert results["concept:b"].beta == pytest.approx(1.3)  # 1.0 + 0.3

    def test_pulls_incremented(self, learner):
        deltas = {"arm:x": {"alpha_delta": 0.5, "beta_delta": 0.0}}
        results = learner.apply_credit_deltas(deltas)
        assert results["arm:x"].pulls == 1

        # Apply again
        results = learner.apply_credit_deltas(deltas)
        assert results["arm:x"].pulls == 2

    def test_total_reward_tracks_alpha_delta(self, learner):
        deltas = {"arm:x": {"alpha_delta": 0.7, "beta_delta": 0.1}}
        results = learner.apply_credit_deltas(deltas)
        assert results["arm:x"].total_reward == pytest.approx(0.7)

    def test_alpha_beta_floored_at_001(self, learner):
        """Negative deltas can't take alpha/beta below 0.01."""
        deltas = {"arm:x": {"alpha_delta": -5.0, "beta_delta": -5.0}}
        results = learner.apply_credit_deltas(deltas)
        assert results["arm:x"].alpha == pytest.approx(0.01)
        assert results["arm:x"].beta == pytest.approx(0.01)

    def test_empty_deltas(self, learner):
        results = learner.apply_credit_deltas({})
        assert results == {}

    def test_persisted_to_store(self, learner):
        deltas = {"arm:x": {"alpha_delta": 0.5, "beta_delta": 0.0}}
        learner.apply_credit_deltas(deltas)

        # Read back from store
        state = learner.store.get("arm:x")
        assert state.alpha == pytest.approx(1.5)
        assert state.pulls == 1

    def test_with_context(self, learner):
        ctx = {"task": "test"}
        deltas = {"arm:x": {"alpha_delta": 1.0, "beta_delta": 0.0}}
        learner.apply_credit_deltas(deltas, context=ctx)

        # Different context should be unaffected
        state_default = learner.store.get("arm:x")
        state_ctx = learner.store.get("arm:x", ctx)
        assert state_default.alpha == pytest.approx(1.0)  # untouched
        assert state_ctx.alpha == pytest.approx(2.0)

    def test_multiple_concepts(self, learner):
        deltas = {
            f"concept:{i}": {"alpha_delta": 0.1 * i, "beta_delta": 0.0}
            for i in range(1, 6)
        }
        results = learner.apply_credit_deltas(deltas)
        assert len(results) == 5
        assert results["concept:3"].alpha == pytest.approx(1.3)

    def test_missing_delta_keys_default_to_zero(self, learner):
        deltas = {"arm:x": {"alpha_delta": 0.5}}  # no beta_delta
        results = learner.apply_credit_deltas(deltas)
        assert results["arm:x"].alpha == pytest.approx(1.5)
        assert results["arm:x"].beta == pytest.approx(1.0)  # unchanged


class TestBatchObserve:
    """Tests for Learner.batch_observe() — bulk observation wrapper."""

    def test_batch_observe_returns_all_states(self, learner):
        outcomes = [
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"),
            ArmOutcome(arm_id="arm:b", reward=0.0, outcome="rejected"),
        ]
        results = learner.batch_observe(outcomes)

        assert len(results) == 2
        assert results["arm:a"].alpha == 2.0
        assert results["arm:b"].beta == 2.0

    def test_batch_observe_accumulates_same_arm(self, learner):
        outcomes = [
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"),
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"),
            ArmOutcome(arm_id="arm:a", reward=0.0, outcome="rejected"),
        ]
        results = learner.batch_observe(outcomes)

        assert results["arm:a"].pulls == 3
        assert results["arm:a"].alpha == 3.0
        assert results["arm:a"].beta == 2.0

    def test_batch_observe_with_context(self, learner):
        ctx = {"task": "testing"}
        outcomes = [
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"),
        ]
        learner.batch_observe(outcomes, context=ctx)

        # Context-partitioned: default context untouched
        p_default = learner.posteriors()
        p_ctx = learner.posteriors(context=ctx)
        assert "arm:a" not in p_default
        assert p_ctx["arm:a"]["alpha"] == 2.0

    def test_batch_observe_empty(self, learner):
        results = learner.batch_observe([])
        assert results == {}

    def test_batch_observe_persisted(self, learner):
        outcomes = [
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"),
        ]
        learner.batch_observe(outcomes)

        state = learner.store.get("arm:a")
        assert state.alpha == 2.0


class TestTopArms:
    """Tests for Learner.top_arms() — ranked arm retrieval."""

    def test_top_arms_sorted_by_mean(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:good", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:good", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:bad", reward=0.0, outcome="rejected"))
        learner.observe(ArmOutcome(arm_id="arm:bad", reward=0.0, outcome="rejected"))

        top = learner.top_arms(k=2)
        assert len(top) == 2
        assert top[0][0] == "arm:good"
        assert top[1][0] == "arm:bad"
        assert top[0][1].mean > top[1][1].mean

    def test_top_arms_k_larger_than_arms(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        top = learner.top_arms(k=100)
        assert len(top) == 1

    def test_top_arms_empty(self, learner):
        top = learner.top_arms()
        assert top == []

    def test_top_arms_with_context(self, learner):
        ctx = {"task": "typing"}
        learner.observe(
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"),
            context=ctx,
        )
        learner.observe(
            ArmOutcome(arm_id="arm:b", reward=1.0, outcome="accepted"),
        )

        top_ctx = learner.top_arms(context=ctx, k=10)
        top_default = learner.top_arms(k=10)

        assert len(top_ctx) == 1
        assert top_ctx[0][0] == "arm:a"
        assert len(top_default) == 1
        assert top_default[0][0] == "arm:b"

    def test_top_arms_returns_arm_state(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        top = learner.top_arms(k=1)
        arm_id, state = top[0]
        assert arm_id == "arm:a"
        assert hasattr(state, "alpha")
        assert hasattr(state, "mean")


class TestDecayArm:
    """Tests for Learner.decay_arm() — signal decay toward prior."""

    def test_decay_weakens_confidence(self, learner):
        # Build up signal
        for _ in range(5):
            learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))

        state_before = learner.store.get("arm:a")
        assert state_before.alpha == 6.0  # 1.0 prior + 5 rewards

        state_after = learner.decay_arm("arm:a", decay_factor=0.5)
        assert state_after.alpha == pytest.approx(3.0)
        assert state_after.beta == pytest.approx(0.5)

    def test_decay_preserves_mean_ratio(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))

        state_before = learner.store.get("arm:a")
        mean_before = state_before.mean

        state_after = learner.decay_arm("arm:a", decay_factor=0.5)
        # alpha/beta both scaled by same factor, so mean = alpha/(alpha+beta) is preserved
        assert state_after.mean == pytest.approx(mean_before, abs=0.01)

    def test_decay_floors_at_001(self, learner):
        # Default arm: alpha=1.0, beta=1.0
        state = learner.decay_arm("arm:new", decay_factor=0.001)
        assert state.alpha == pytest.approx(0.01)
        assert state.beta == pytest.approx(0.01)

    def test_decay_persisted(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.decay_arm("arm:a", decay_factor=0.5)

        state = learner.store.get("arm:a")
        assert state.alpha == pytest.approx(1.0)  # (1+1)*0.5

    def test_decay_with_context(self, learner):
        ctx = {"task": "testing"}
        learner.observe(
            ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"),
            context=ctx,
        )
        learner.decay_arm("arm:a", decay_factor=0.5, context=ctx)

        state_ctx = learner.store.get("arm:a", ctx)
        state_default = learner.store.get("arm:a")
        assert state_ctx.alpha == pytest.approx(1.0)  # decayed
        assert state_default.alpha == pytest.approx(1.0)  # untouched (default prior)

    def test_decay_updates_total_reward(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        state = learner.decay_arm("arm:a", decay_factor=0.5)
        assert state.total_reward == pytest.approx(0.5)

    def test_decay_preserves_pulls(self, learner):
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        learner.observe(ArmOutcome(arm_id="arm:a", reward=1.0, outcome="accepted"))
        state = learner.decay_arm("arm:a", decay_factor=0.5)
        assert state.pulls == 2  # not changed
