"""Tests for learning MCP tool wrappers via create_server()."""

from __future__ import annotations

import pytest
from qortex.observe import reset as obs_reset

from qortex.core.memory import InMemoryBackend
from qortex.mcp import server as mcp_server


@pytest.fixture(autouse=True)
def _reset_state(tmp_path):
    obs_reset()
    mcp_server._backend = None
    mcp_server._vector_index = None
    mcp_server._adapter = None
    mcp_server._graph_adapter = None
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None
    mcp_server._learners = {}
    mcp_server._learning_state_dir = str(tmp_path)
    yield
    mcp_server._backend = None
    mcp_server._vector_index = None
    mcp_server._adapter = None
    mcp_server._graph_adapter = None
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None
    mcp_server._learners = {}
    mcp_server._learning_state_dir = ""
    obs_reset()


@pytest.fixture
def server():
    backend = InMemoryBackend()
    backend.connect()
    mcp_server.create_server(backend=backend)
    return mcp_server


class TestLearningSelectMCP:
    async def test_select_basic(self, server):
        result = await server._learning_select_impl(
            learner="test",
            candidates=[
                {"id": "arm:a"},
                {"id": "arm:b"},
                {"id": "arm:c"},
            ],
            k=2,
        )

        assert len(result["selected_arms"]) == 2
        assert len(result["excluded_arms"]) == 1
        assert "is_baseline" in result

    async def test_select_with_context(self, server):
        result = await server._learning_select_impl(
            learner="test",
            candidates=[{"id": "arm:a"}, {"id": "arm:b"}],
            context={"task": "typing"},
            k=1,
        )

        assert len(result["selected_arms"]) == 1

    async def test_select_with_token_budget(self, server):
        result = await server._learning_select_impl(
            learner="test",
            candidates=[
                {"id": "arm:a", "token_cost": 100},
                {"id": "arm:b", "token_cost": 200},
                {"id": "arm:c", "token_cost": 300},
            ],
            k=3,
            token_budget=250,
        )

        assert result["used_tokens"] <= 250
        assert result["token_budget"] == 250


class TestLearningObserveMCP:
    async def test_observe_basic(self, server):
        result = await server._learning_observe_impl(
            learner="test",
            arm_id="arm:a",
            outcome="accepted",
        )

        assert result["arm_id"] == "arm:a"
        assert result["alpha"] == 2.0
        assert result["pulls"] == 1

    async def test_observe_with_reward(self, server):
        result = await server._learning_observe_impl(
            learner="test",
            arm_id="arm:b",
            reward=0.7,
        )

        assert result["alpha"] == 1.7
        assert result["pulls"] == 1

    async def test_observe_multiple(self, server):
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        result = await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="rejected")

        assert result["pulls"] == 3
        assert result["alpha"] == 3.0


class TestLearningPosteriorsMCP:
    async def test_posteriors_empty(self, server):
        result = await server._learning_posteriors_impl(learner="test")
        assert result["posteriors"] == {}

    async def test_posteriors_after_observe(self, server):
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        result = await server._learning_posteriors_impl(learner="test")

        assert "arm:a" in result["posteriors"]
        assert result["posteriors"]["arm:a"]["mean"] > 0.5

    async def test_posteriors_filter(self, server):
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        await server._learning_observe_impl(learner="test", arm_id="arm:b", outcome="rejected")

        result = await server._learning_posteriors_impl(learner="test", arm_ids=["arm:a"])
        assert "arm:a" in result["posteriors"]
        assert "arm:b" not in result["posteriors"]


class TestLearningMetricsMCP:
    async def test_metrics_basic(self, server):
        result = await server._learning_metrics_impl(learner="test")

        assert "total_pulls" in result
        assert "accuracy" in result
        assert result["learner"] == "test"


class TestLearningSessionMCP:
    async def test_session_lifecycle(self, server):
        start = await server._learning_session_start_impl(learner="test", session_name="s1")
        assert "session_id" in start

        end = await server._learning_session_end_impl(start["session_id"])
        assert end["session_id"] == start["session_id"]
        assert end["started_at"]
        assert end["ended_at"]

    async def test_session_not_found(self, server):
        result = await server._learning_session_end_impl("nonexistent")
        assert "error" in result


class TestLearningResetMCP:
    async def test_reset_all(self, server):
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        await server._learning_observe_impl(learner="test", arm_id="arm:b", outcome="rejected")

        result = await server._learning_reset_impl(learner="test")
        assert result["learner"] == "test"
        assert result["deleted"] == 2
        assert result["status"] == "reset"

        # Posteriors should be empty after reset (new learner instance)
        posteriors = await server._learning_posteriors_impl(learner="test")
        assert posteriors["posteriors"] == {}

    async def test_reset_specific_arms(self, server):
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        await server._learning_observe_impl(learner="test", arm_id="arm:b", outcome="accepted")

        result = await server._learning_reset_impl(learner="test", arm_ids=["arm:a"])
        assert result["deleted"] == 1

        # arm:b still has data
        posteriors = await server._learning_posteriors_impl(learner="test")
        assert "arm:b" in posteriors["posteriors"]
        assert "arm:a" not in posteriors["posteriors"]

    async def test_reset_by_context(self, server):
        ctx = {"task": "typing"}
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        await server._learning_observe_impl(
            learner="test", arm_id="arm:b", outcome="accepted", context=ctx
        )

        result = await server._learning_reset_impl(learner="test", context=ctx)
        assert result["deleted"] == 1

        # Default context arm survives
        posteriors = await server._learning_posteriors_impl(learner="test")
        assert "arm:a" in posteriors["posteriors"]

    async def test_reset_evicts_learner_cache(self, server):
        await server._learning_observe_impl(learner="test", arm_id="arm:a", outcome="accepted")
        assert "test" in server._learners

        await server._learning_reset_impl(learner="test")
        assert "test" not in server._learners

        # Next call auto-creates a fresh learner
        await server._learning_observe_impl(learner="test", arm_id="arm:x", outcome="accepted")
        assert "test" in server._learners


class TestLearningSeedArmsMCP:
    async def test_seed_arms_apply_on_first_use(self, server):
        """seed_arms/seed_boost should boost priors when learner is first created."""
        result = await server._learning_select_impl(
            learner="seeded",
            candidates=[{"id": "arm:a"}, {"id": "arm:b"}],
            k=1,
            seed_arms=["arm:a"],
            seed_boost=100.0,  # large boost makes Beta(100,1) vs Beta(1,1) deterministic
        )
        # arm:a should be selected because it has a heavily boosted prior
        assert result["selected_arms"][0]["id"] == "arm:a"

        # Verify the boosted posterior
        posteriors = await server._learning_posteriors_impl(learner="seeded")
        assert posteriors["posteriors"]["arm:a"]["alpha"] == 100.0

    async def test_seed_arms_ignored_on_cached_learner(self, server):
        """seed_arms on a second call should not re-create the learner."""
        await server._learning_observe_impl(learner="cached", arm_id="arm:a", outcome="accepted")
        # Second call with seed_arms -- learner already cached, seeds ignored
        await server._learning_select_impl(
            learner="cached",
            candidates=[{"id": "arm:a"}, {"id": "arm:b"}],
            k=1,
            seed_arms=["arm:b"],
            seed_boost=10.0,
        )
        posteriors = await server._learning_posteriors_impl(learner="cached")
        # arm:b should NOT have boosted alpha (learner was cached)
        assert "arm:b" not in posteriors["posteriors"]


class TestLearnerAutoCreation:
    async def test_different_learners_independent(self, server):
        await server._learning_observe_impl(learner="alpha", arm_id="arm:a", outcome="accepted")
        await server._learning_observe_impl(learner="beta", arm_id="arm:a", outcome="rejected")

        alpha_p = await server._learning_posteriors_impl(learner="alpha")
        beta_p = await server._learning_posteriors_impl(learner="beta")

        assert alpha_p["posteriors"]["arm:a"]["alpha"] == 2.0
        assert beta_p["posteriors"]["arm:a"]["alpha"] == 1.0
