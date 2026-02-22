"""Tests for LearningStore protocol, JsonLearningStore, and SqliteLearningStore."""

from __future__ import annotations

import asyncio

import pytest

from qortex.learning.store import (
    JsonLearningStore,
    LearningStore,
    SqliteLearningStore,
)
from qortex.learning.types import ArmState


@pytest.fixture(params=["sqlite", "json"], ids=["sqlite", "json"])
def store(request, tmp_path) -> LearningStore:
    if request.param == "sqlite":
        return SqliteLearningStore("test-store", str(tmp_path))
    return JsonLearningStore("test-store", str(tmp_path))


@pytest.fixture
def state_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_sqlite_satisfies_protocol(self, tmp_path):
        s = SqliteLearningStore("p", str(tmp_path))
        assert isinstance(s, LearningStore)

    def test_json_satisfies_protocol(self, tmp_path):
        s = JsonLearningStore("p", str(tmp_path))
        assert isinstance(s, LearningStore)


# ---------------------------------------------------------------------------
# CRUD — runs against both backends
# ---------------------------------------------------------------------------


class TestStoreCRUD:
    async def test_get_missing_returns_default(self, store):
        state = await store.get("nonexistent")
        assert state.alpha == 1.0
        assert state.beta == 1.0
        assert state.pulls == 0

    async def test_put_then_get(self, store):
        s = ArmState(alpha=3.0, beta=2.0, pulls=5, total_reward=3.0, last_updated="t1")
        await store.put("arm:a", s)
        got = await store.get("arm:a")
        assert got.alpha == 3.0
        assert got.beta == 2.0
        assert got.pulls == 5
        assert got.total_reward == 3.0
        assert got.last_updated == "t1"

    async def test_put_overwrites(self, store):
        await store.put("arm:a", ArmState(alpha=2.0))
        await store.put("arm:a", ArmState(alpha=9.0))
        state = await store.get("arm:a")
        assert state.alpha == 9.0

    async def test_get_all_empty(self, store):
        assert await store.get_all() == {}

    async def test_get_all_returns_context_arms(self, store):
        await store.put("arm:a", ArmState(alpha=2.0))
        await store.put("arm:b", ArmState(alpha=3.0))
        all_arms = await store.get_all()
        assert set(all_arms.keys()) == {"arm:a", "arm:b"}
        assert all_arms["arm:a"].alpha == 2.0

    async def test_get_all_contexts_empty(self, store):
        assert await store.get_all_contexts() == []

    async def test_get_all_contexts(self, store):
        await store.put("arm:a", ArmState(), context={"task": "x"})
        await store.put("arm:b", ArmState(), context={"task": "y"})
        contexts = await store.get_all_contexts()
        assert len(contexts) == 2

    async def test_save_persists_data(self, store):
        await store.put("arm:a", ArmState(alpha=5.0))
        await store.save()
        # Verify data is still readable after explicit save
        state = await store.get("arm:a")
        assert state.alpha == 5.0


# ---------------------------------------------------------------------------
# Context partitioning
# ---------------------------------------------------------------------------


class TestContextPartitioning:
    async def test_same_arm_different_contexts_independent(self, store):
        ctx_a = {"task": "typing"}
        ctx_b = {"task": "linting"}

        await store.put("arm:x", ArmState(alpha=10.0), context=ctx_a)
        await store.put("arm:x", ArmState(alpha=1.0), context=ctx_b)

        assert (await store.get("arm:x", context=ctx_a)).alpha == 10.0
        assert (await store.get("arm:x", context=ctx_b)).alpha == 1.0

    async def test_get_all_scoped_to_context(self, store):
        ctx_a = {"task": "typing"}
        ctx_b = {"task": "linting"}

        await store.put("arm:a", ArmState(alpha=2.0), context=ctx_a)
        await store.put("arm:b", ArmState(alpha=3.0), context=ctx_b)

        a_arms = await store.get_all(context=ctx_a)
        assert "arm:a" in a_arms
        assert "arm:b" not in a_arms

    async def test_default_context_is_none(self, store):
        await store.put("arm:a", ArmState(alpha=5.0))
        await store.put("arm:a", ArmState(alpha=9.0), context={"task": "x"})

        assert (await store.get("arm:a")).alpha == 5.0
        assert (await store.get("arm:a", context={"task": "x"})).alpha == 9.0


# ---------------------------------------------------------------------------
# get_all_states
# ---------------------------------------------------------------------------


class TestGetAllStates:
    async def test_empty(self, store):
        assert await store.get_all_states() == {}

    async def test_returns_nested_dict(self, store):
        await store.put("arm:a", ArmState(alpha=2.0))
        await store.put("arm:b", ArmState(alpha=3.0), context={"task": "x"})

        all_states = await store.get_all_states()
        assert len(all_states) == 2

        # Find keys (context hashes)
        for _ctx_hash, arms in all_states.items():
            assert isinstance(arms, dict)
            for _arm_id, state in arms.items():
                assert isinstance(state, ArmState)

    async def test_all_states_has_correct_values(self, store):
        await store.put("arm:a", ArmState(alpha=2.0, pulls=3, total_reward=2.0))
        await store.put("arm:b", ArmState(alpha=4.0, pulls=5, total_reward=4.0))

        all_states = await store.get_all_states()
        # Both arms in default context
        default_ctx = list(all_states.values())[0]
        assert default_ctx["arm:a"].alpha == 2.0
        assert default_ctx["arm:b"].pulls == 5


# ---------------------------------------------------------------------------
# SQLite-specific: persistence across connections
# ---------------------------------------------------------------------------


class TestSqlitePersistence:
    async def test_data_survives_new_instance(self, state_dir):
        store1 = SqliteLearningStore("persist-test", state_dir)
        await store1.put("arm:a", ArmState(alpha=7.0, beta=2.0, pulls=10, total_reward=7.0))
        await store1.save()
        del store1

        # New instance, same path
        store2 = SqliteLearningStore("persist-test", state_dir)
        got = await store2.get("arm:a")
        assert got.alpha == 7.0
        assert got.beta == 2.0
        assert got.pulls == 10
        assert got.total_reward == 7.0

    async def test_context_partitioning_persists(self, state_dir):
        store1 = SqliteLearningStore("ctx-persist", state_dir)
        await store1.put("arm:a", ArmState(alpha=5.0), context={"task": "x"})
        await store1.put("arm:a", ArmState(alpha=9.0), context={"task": "y"})
        await store1.save()
        del store1

        store2 = SqliteLearningStore("ctx-persist", state_dir)
        assert (await store2.get("arm:a", context={"task": "x"})).alpha == 5.0
        assert (await store2.get("arm:a", context={"task": "y"})).alpha == 9.0

    async def test_lazy_connection(self, state_dir):
        store = SqliteLearningStore("lazy-test", state_dir)
        # First call triggers connection; verify by checking DB file exists after
        import os

        db_path = os.path.join(state_dir, "lazy-test.db")
        assert not os.path.exists(db_path)
        await store.get("arm:a")
        assert os.path.exists(db_path)


# ---------------------------------------------------------------------------
# Name sanitization (path traversal prevention)
# ---------------------------------------------------------------------------


class TestNameSanitization:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "../../evil",
            "../passwd",
            "foo/bar",
            "foo\\bar",
            "",
            ".hidden",
            "name with spaces",
        ],
    )
    def test_rejects_unsafe_names_sqlite(self, tmp_path, bad_name):
        with pytest.raises(ValueError, match="Invalid learner name"):
            SqliteLearningStore(bad_name, str(tmp_path))

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../../evil",
            "../passwd",
            "foo/bar",
            "",
            ".hidden",
        ],
    )
    def test_rejects_unsafe_names_json(self, tmp_path, bad_name):
        with pytest.raises(ValueError, match="Invalid learner name"):
            JsonLearningStore(bad_name, str(tmp_path))

    @pytest.mark.parametrize(
        "good_name",
        [
            "my-learner",
            "prompt_optimizer",
            "v2.1",
            "learner-A",
            "test123",
        ],
    )
    def test_accepts_safe_names(self, tmp_path, good_name):
        SqliteLearningStore(good_name, str(tmp_path))
        JsonLearningStore(good_name, str(tmp_path))


# ---------------------------------------------------------------------------
# delete() — runs against both backends
# ---------------------------------------------------------------------------


class TestStoreDelete:
    async def test_delete_all(self, store):
        """arm_ids=None, context=None -> delete everything."""
        await store.put("arm:a", ArmState(alpha=2.0))
        await store.put("arm:b", ArmState(alpha=3.0), context={"task": "x"})
        count = await store.delete()
        await store.save()
        assert count == 2
        assert await store.get_all() == {}
        assert await store.get_all(context={"task": "x"}) == {}

    async def test_delete_by_context(self, store):
        """arm_ids=None, context=set -> delete all arms for that context."""
        await store.put("arm:a", ArmState(alpha=2.0))
        await store.put("arm:b", ArmState(alpha=3.0), context={"task": "x"})
        await store.put("arm:c", ArmState(alpha=4.0), context={"task": "x"})
        count = await store.delete(context={"task": "x"})
        await store.save()
        assert count == 2
        # Default context untouched
        assert (await store.get("arm:a")).alpha == 2.0
        assert await store.get_all(context={"task": "x"}) == {}

    async def test_delete_specific_arms_default_context(self, store):
        """arm_ids=set, context=None -> delete those arms in default context."""
        await store.put("arm:a", ArmState(alpha=2.0))
        await store.put("arm:b", ArmState(alpha=3.0))
        await store.put("arm:c", ArmState(alpha=4.0))
        count = await store.delete(arm_ids=["arm:a", "arm:b"])
        await store.save()
        assert count == 2
        assert (await store.get("arm:a")).pulls == 0  # back to default
        assert (await store.get("arm:c")).alpha == 4.0  # untouched

    async def test_delete_specific_arms_specific_context(self, store):
        """arm_ids=set, context=set -> delete those arms in that context."""
        ctx = {"task": "x"}
        await store.put("arm:a", ArmState(alpha=5.0), context=ctx)
        await store.put("arm:b", ArmState(alpha=6.0), context=ctx)
        await store.put("arm:a", ArmState(alpha=9.0))  # default context, untouched
        count = await store.delete(arm_ids=["arm:a"], context=ctx)
        await store.save()
        assert count == 1
        assert (await store.get("arm:a", context=ctx)).pulls == 0  # deleted
        assert (await store.get("arm:b", context=ctx)).alpha == 6.0  # untouched
        assert (await store.get("arm:a")).alpha == 9.0  # default untouched

    async def test_delete_nonexistent_returns_zero(self, store):
        count = await store.delete(arm_ids=["nope"])
        assert count == 0

    async def test_delete_empty_arm_ids_is_noop(self, store):
        """arm_ids=[] should not crash (especially SQLite IN clause)."""
        await store.put("arm:a", ArmState(alpha=2.0))
        count = await store.delete(arm_ids=[])
        assert count == 0
        assert (await store.get("arm:a")).alpha == 2.0


# ---------------------------------------------------------------------------
# Concurrent access (regression for #93)
# ---------------------------------------------------------------------------


class TestSqliteConcurrency:
    async def test_concurrent_observe_no_crash(self, tmp_path):
        """Multiple async tasks hitting get/put simultaneously must not raise."""
        store = SqliteLearningStore("concurrent-test", str(tmp_path))

        async def observe_arm(arm_idx: int) -> str:
            arm_id = f"tool:cat:{arm_idx}"
            state = await store.get(arm_id)
            new_state = ArmState(
                alpha=state.alpha + 1.0,
                beta=state.beta,
                pulls=state.pulls + 1,
                total_reward=state.total_reward + 1.0,
                last_updated="t",
            )
            await store.put(arm_id, new_state)
            await store.save()
            return arm_id

        results = await asyncio.gather(*[observe_arm(i) for i in range(50)])

        assert len(results) == 50
        # All 50 arms persisted
        all_states = await store.get_all()
        assert len(all_states) == 50

    async def test_concurrent_same_arm_no_crash(self, tmp_path):
        """Multiple async tasks updating the same arm must not raise."""
        store = SqliteLearningStore("same-arm-test", str(tmp_path))

        async def observe(_: int) -> None:
            state = await store.get("arm:shared")
            new_state = ArmState(
                alpha=state.alpha + 0.5,
                beta=state.beta,
                pulls=state.pulls + 1,
                total_reward=state.total_reward + 0.5,
                last_updated="t",
            )
            await store.put("arm:shared", new_state)
            await store.save()

        await asyncio.gather(*[observe(i) for i in range(30)])

        final = await store.get("arm:shared")
        assert final.pulls >= 1  # at least some updates landed
