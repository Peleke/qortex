"""Tests for PostgresLearningStore.

Integration tests require a running PostgreSQL instance.
Run with: uv run pytest tests/test_pg_learning_store.py -v -m integration
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

POSTGRES_DSN = os.environ.get(
    "TEST_POSTGRES_DSN",
    "postgresql://qortex:qortex@localhost:5432/qortex",
)


@pytest.fixture
async def pool():
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=3)
    yield pool
    await pool.close()


@pytest.fixture
async def store(pool):
    from qortex.learning.pg_store import PostgresLearningStore

    s = PostgresLearningStore("test_learner", pool)
    await s._ensure_schema()
    # Clean table for this learner before each test
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM learning_arm_states WHERE learner_name = $1",
            "test_learner",
        )
    return s


async def test_get_default(store):

    state = await store.get("nonexistent_arm")
    assert state.alpha == pytest.approx(1.0)
    assert state.beta == pytest.approx(1.0)
    assert state.pulls == 0


async def test_put_and_get(store):
    from qortex.learning.types import ArmState

    state = ArmState(
        alpha=3.0, beta=2.0, pulls=5, total_reward=3.5, last_updated="2025-01-01T00:00:00+00:00"
    )
    await store.put("arm_a", state)

    loaded = await store.get("arm_a")
    assert loaded.alpha == pytest.approx(3.0)
    assert loaded.beta == pytest.approx(2.0)
    assert loaded.pulls == 5
    assert loaded.total_reward == pytest.approx(3.5)


async def test_put_upsert(store):
    from qortex.learning.types import ArmState

    await store.put("arm_a", ArmState(alpha=1.0, beta=1.0, pulls=0))
    await store.put("arm_a", ArmState(alpha=5.0, beta=3.0, pulls=10, total_reward=7.0))

    loaded = await store.get("arm_a")
    assert loaded.alpha == pytest.approx(5.0)
    assert loaded.pulls == 10


async def test_get_all(store):
    from qortex.learning.types import ArmState

    await store.put("arm_a", ArmState(alpha=2.0, beta=1.0, pulls=3))
    await store.put("arm_b", ArmState(alpha=4.0, beta=2.0, pulls=6))

    all_arms = await store.get_all()
    assert len(all_arms) == 2
    assert "arm_a" in all_arms
    assert "arm_b" in all_arms


async def test_context_partitioning(store):
    from qortex.learning.types import ArmState

    ctx1 = {"task": "routing"}
    ctx2 = {"task": "scoring"}

    await store.put("arm_a", ArmState(alpha=2.0, pulls=1), context=ctx1)
    await store.put("arm_a", ArmState(alpha=5.0, pulls=10), context=ctx2)

    s1 = await store.get("arm_a", context=ctx1)
    s2 = await store.get("arm_a", context=ctx2)
    assert s1.alpha == pytest.approx(2.0)
    assert s2.alpha == pytest.approx(5.0)


async def test_get_all_contexts(store):
    from qortex.learning.types import ArmState

    await store.put("arm_a", ArmState(), context={"task": "a"})
    await store.put("arm_b", ArmState(), context={"task": "b"})

    contexts = await store.get_all_contexts()
    assert len(contexts) == 2


async def test_get_all_states(store):
    from qortex.learning.types import ArmState

    await store.put("arm_a", ArmState(alpha=2.0), context={"task": "x"})
    await store.put("arm_b", ArmState(alpha=3.0), context={"task": "x"})

    states = await store.get_all_states()
    assert len(states) >= 1


async def test_delete_all(store):
    from qortex.learning.types import ArmState

    await store.put("arm_a", ArmState())
    await store.put("arm_b", ArmState())

    count = await store.delete()
    assert count == 2

    all_arms = await store.get_all()
    assert len(all_arms) == 0


async def test_delete_by_arm_ids(store):
    from qortex.learning.types import ArmState

    await store.put("arm_a", ArmState())
    await store.put("arm_b", ArmState())
    await store.put("arm_c", ArmState())

    count = await store.delete(arm_ids=["arm_a", "arm_c"])
    assert count == 2

    all_arms = await store.get_all()
    assert len(all_arms) == 1
    assert "arm_b" in all_arms


async def test_delete_by_context(store):
    from qortex.learning.types import ArmState

    ctx = {"task": "routing"}
    await store.put("arm_a", ArmState(), context=ctx)
    await store.put("arm_b", ArmState())

    count = await store.delete(context=ctx)
    assert count == 1


async def test_delete_empty_arm_ids(store):
    count = await store.delete(arm_ids=[])
    assert count == 0


async def test_save_noop(store):
    # save() is a no-op for postgres
    await store.save()


async def test_learner_isolation(pool):
    """Different learner names don't see each other's data."""
    from qortex.learning.pg_store import PostgresLearningStore
    from qortex.learning.types import ArmState

    store_a = PostgresLearningStore("learner_a", pool)
    store_b = PostgresLearningStore("learner_b", pool)

    await store_a._ensure_schema()

    # Clean up
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM learning_arm_states WHERE learner_name IN ('learner_a', 'learner_b')"
        )

    await store_a.put("arm_x", ArmState(alpha=10.0))
    await store_b.put("arm_x", ArmState(alpha=20.0))

    state_a = await store_a.get("arm_x")
    state_b = await store_b.get("arm_x")

    assert state_a.alpha == pytest.approx(10.0)
    assert state_b.alpha == pytest.approx(20.0)

    all_a = await store_a.get_all()
    all_b = await store_b.get_all()
    assert len(all_a) == 1
    assert len(all_b) == 1
