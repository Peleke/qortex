"""Tests for LearningStore protocol, JsonLearningStore, and SqliteLearningStore."""

from __future__ import annotations

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
    def test_get_missing_returns_default(self, store):
        state = store.get("nonexistent")
        assert state.alpha == 1.0
        assert state.beta == 1.0
        assert state.pulls == 0

    def test_put_then_get(self, store):
        s = ArmState(alpha=3.0, beta=2.0, pulls=5, total_reward=3.0, last_updated="t1")
        store.put("arm:a", s)
        got = store.get("arm:a")
        assert got.alpha == 3.0
        assert got.beta == 2.0
        assert got.pulls == 5
        assert got.total_reward == 3.0
        assert got.last_updated == "t1"

    def test_put_overwrites(self, store):
        store.put("arm:a", ArmState(alpha=2.0))
        store.put("arm:a", ArmState(alpha=9.0))
        assert store.get("arm:a").alpha == 9.0

    def test_get_all_empty(self, store):
        assert store.get_all() == {}

    def test_get_all_returns_context_arms(self, store):
        store.put("arm:a", ArmState(alpha=2.0))
        store.put("arm:b", ArmState(alpha=3.0))
        all_arms = store.get_all()
        assert set(all_arms.keys()) == {"arm:a", "arm:b"}
        assert all_arms["arm:a"].alpha == 2.0

    def test_get_all_contexts_empty(self, store):
        assert store.get_all_contexts() == []

    def test_get_all_contexts(self, store):
        store.put("arm:a", ArmState(), context={"task": "x"})
        store.put("arm:b", ArmState(), context={"task": "y"})
        contexts = store.get_all_contexts()
        assert len(contexts) == 2

    def test_save_persists_data(self, store):
        store.put("arm:a", ArmState(alpha=5.0))
        store.save()
        # Verify data is still readable after explicit save
        assert store.get("arm:a").alpha == 5.0


# ---------------------------------------------------------------------------
# Context partitioning
# ---------------------------------------------------------------------------


class TestContextPartitioning:
    def test_same_arm_different_contexts_independent(self, store):
        ctx_a = {"task": "typing"}
        ctx_b = {"task": "linting"}

        store.put("arm:x", ArmState(alpha=10.0), context=ctx_a)
        store.put("arm:x", ArmState(alpha=1.0), context=ctx_b)

        assert store.get("arm:x", context=ctx_a).alpha == 10.0
        assert store.get("arm:x", context=ctx_b).alpha == 1.0

    def test_get_all_scoped_to_context(self, store):
        ctx_a = {"task": "typing"}
        ctx_b = {"task": "linting"}

        store.put("arm:a", ArmState(alpha=2.0), context=ctx_a)
        store.put("arm:b", ArmState(alpha=3.0), context=ctx_b)

        a_arms = store.get_all(context=ctx_a)
        assert "arm:a" in a_arms
        assert "arm:b" not in a_arms

    def test_default_context_is_none(self, store):
        store.put("arm:a", ArmState(alpha=5.0))
        store.put("arm:a", ArmState(alpha=9.0), context={"task": "x"})

        assert store.get("arm:a").alpha == 5.0
        assert store.get("arm:a", context={"task": "x"}).alpha == 9.0


# ---------------------------------------------------------------------------
# get_all_states
# ---------------------------------------------------------------------------


class TestGetAllStates:
    def test_empty(self, store):
        assert store.get_all_states() == {}

    def test_returns_nested_dict(self, store):
        store.put("arm:a", ArmState(alpha=2.0))
        store.put("arm:b", ArmState(alpha=3.0), context={"task": "x"})

        all_states = store.get_all_states()
        assert len(all_states) == 2

        # Find keys (context hashes)
        for _ctx_hash, arms in all_states.items():
            assert isinstance(arms, dict)
            for _arm_id, state in arms.items():
                assert isinstance(state, ArmState)

    def test_all_states_has_correct_values(self, store):
        store.put("arm:a", ArmState(alpha=2.0, pulls=3, total_reward=2.0))
        store.put("arm:b", ArmState(alpha=4.0, pulls=5, total_reward=4.0))

        all_states = store.get_all_states()
        # Both arms in default context
        default_ctx = list(all_states.values())[0]
        assert default_ctx["arm:a"].alpha == 2.0
        assert default_ctx["arm:b"].pulls == 5


# ---------------------------------------------------------------------------
# SQLite-specific: persistence across connections
# ---------------------------------------------------------------------------


class TestSqlitePersistence:
    def test_data_survives_new_instance(self, state_dir):
        store1 = SqliteLearningStore("persist-test", state_dir)
        store1.put("arm:a", ArmState(alpha=7.0, beta=2.0, pulls=10, total_reward=7.0))
        store1.save()
        del store1

        # New instance, same path
        store2 = SqliteLearningStore("persist-test", state_dir)
        got = store2.get("arm:a")
        assert got.alpha == 7.0
        assert got.beta == 2.0
        assert got.pulls == 10
        assert got.total_reward == 7.0

    def test_context_partitioning_persists(self, state_dir):
        store1 = SqliteLearningStore("ctx-persist", state_dir)
        store1.put("arm:a", ArmState(alpha=5.0), context={"task": "x"})
        store1.put("arm:a", ArmState(alpha=9.0), context={"task": "y"})
        store1.save()
        del store1

        store2 = SqliteLearningStore("ctx-persist", state_dir)
        assert store2.get("arm:a", context={"task": "x"}).alpha == 5.0
        assert store2.get("arm:a", context={"task": "y"}).alpha == 9.0

    def test_lazy_connection(self, state_dir):
        store = SqliteLearningStore("lazy-test", state_dir)
        # First call triggers connection; verify by checking DB file exists after
        import os

        db_path = os.path.join(state_dir, "lazy-test.db")
        assert not os.path.exists(db_path)
        store.get("arm:a")
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
    def test_delete_all(self, store):
        """arm_ids=None, context=None -> delete everything."""
        store.put("arm:a", ArmState(alpha=2.0))
        store.put("arm:b", ArmState(alpha=3.0), context={"task": "x"})
        count = store.delete()
        store.save()
        assert count == 2
        assert store.get_all() == {}
        assert store.get_all(context={"task": "x"}) == {}

    def test_delete_by_context(self, store):
        """arm_ids=None, context=set -> delete all arms for that context."""
        store.put("arm:a", ArmState(alpha=2.0))
        store.put("arm:b", ArmState(alpha=3.0), context={"task": "x"})
        store.put("arm:c", ArmState(alpha=4.0), context={"task": "x"})
        count = store.delete(context={"task": "x"})
        store.save()
        assert count == 2
        # Default context untouched
        assert store.get("arm:a").alpha == 2.0
        assert store.get_all(context={"task": "x"}) == {}

    def test_delete_specific_arms_default_context(self, store):
        """arm_ids=set, context=None -> delete those arms in default context."""
        store.put("arm:a", ArmState(alpha=2.0))
        store.put("arm:b", ArmState(alpha=3.0))
        store.put("arm:c", ArmState(alpha=4.0))
        count = store.delete(arm_ids=["arm:a", "arm:b"])
        store.save()
        assert count == 2
        assert store.get("arm:a").pulls == 0  # back to default
        assert store.get("arm:c").alpha == 4.0  # untouched

    def test_delete_specific_arms_specific_context(self, store):
        """arm_ids=set, context=set -> delete those arms in that context."""
        ctx = {"task": "x"}
        store.put("arm:a", ArmState(alpha=5.0), context=ctx)
        store.put("arm:b", ArmState(alpha=6.0), context=ctx)
        store.put("arm:a", ArmState(alpha=9.0))  # default context, untouched
        count = store.delete(arm_ids=["arm:a"], context=ctx)
        store.save()
        assert count == 1
        assert store.get("arm:a", context=ctx).pulls == 0  # deleted
        assert store.get("arm:b", context=ctx).alpha == 6.0  # untouched
        assert store.get("arm:a").alpha == 9.0  # default untouched

    def test_delete_nonexistent_returns_zero(self, store):
        count = store.delete(arm_ids=["nope"])
        assert count == 0

    def test_delete_empty_arm_ids_is_noop(self, store):
        """arm_ids=[] should not crash (especially SQLite IN clause)."""
        store.put("arm:a", ArmState(alpha=2.0))
        count = store.delete(arm_ids=[])
        assert count == 0
        assert store.get("arm:a").alpha == 2.0


# ---------------------------------------------------------------------------
# Concurrent access (regression for #93)
# ---------------------------------------------------------------------------


class TestSqliteConcurrency:
    def test_concurrent_observe_no_crash(self, tmp_path):
        """Multiple threads hitting get/put simultaneously must not raise."""
        import concurrent.futures

        store = SqliteLearningStore("concurrent-test", str(tmp_path))

        def observe_arm(arm_idx: int) -> str:
            arm_id = f"tool:cat:{arm_idx}"
            state = store.get(arm_id)
            new_state = ArmState(
                alpha=state.alpha + 1.0,
                beta=state.beta,
                pulls=state.pulls + 1,
                total_reward=state.total_reward + 1.0,
                last_updated="t",
            )
            store.put(arm_id, new_state)
            store.save()
            return arm_id

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(observe_arm, i) for i in range(50)]
            results = [f.result() for f in futures]

        assert len(results) == 50
        # All 50 arms persisted
        all_states = store.get_all()
        assert len(all_states) == 50

    def test_concurrent_same_arm_no_crash(self, tmp_path):
        """Multiple threads updating the same arm must not raise."""
        import concurrent.futures

        store = SqliteLearningStore("same-arm-test", str(tmp_path))

        def observe(_: int) -> None:
            state = store.get("arm:shared")
            new_state = ArmState(
                alpha=state.alpha + 0.5,
                beta=state.beta,
                pulls=state.pulls + 1,
                total_reward=state.total_reward + 0.5,
                last_updated="t",
            )
            store.put("arm:shared", new_state)
            store.save()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(observe, i) for i in range(30)]
            for f in futures:
                f.result()  # must not raise InterfaceError

        final = store.get("arm:shared")
        assert final.pulls >= 1  # at least some updates landed
