"""Tests for PostgresInteroceptionStore.

Integration tests require a running PostgreSQL instance.
Run with: uv run pytest tests/test_pg_interoception_store.py -v -m integration
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
    """Create an isolated asyncpg pool for testing."""
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(POSTGRES_DSN, min_size=1, max_size=3)
    yield pool
    await pool.close()


@pytest.fixture
async def store(pool):
    """Create a PostgresInteroceptionStore with clean tables."""
    from qortex.hippocampus.pg_store import PostgresInteroceptionStore

    s = PostgresInteroceptionStore(pool)
    await s._ensure_schema()
    # Clean tables before each test
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM interoception_factors")
        await conn.execute("DELETE FROM interoception_edge_buffer")
    return s


async def test_save_and_load_factors(store):
    await store.save_factor("node_a", 1.5)
    await store.save_factor("node_b", 0.8)

    factors = await store.load_factors()
    assert factors["node_a"] == pytest.approx(1.5)
    assert factors["node_b"] == pytest.approx(0.8)


async def test_save_factor_upsert(store):
    await store.save_factor("node_a", 1.0)
    await store.save_factor("node_a", 2.0)

    factors = await store.load_factors()
    assert factors["node_a"] == pytest.approx(2.0)


async def test_save_factors_batch(store):
    batch = {f"node_{i}": float(i) for i in range(10)}
    await store.save_factors(batch)

    factors = await store.load_factors()
    assert len(factors) == 10
    assert factors["node_5"] == pytest.approx(5.0)


async def test_save_factors_empty(store):
    await store.save_factors({})
    factors = await store.load_factors()
    assert factors == {}


async def test_save_and_load_edges(store):
    from qortex.hippocampus.buffer import EdgeStats

    buffer = {
        ("a", "b"): EdgeStats(hit_count=3, scores=[0.9, 0.8, 0.7], last_seen="2025-01-01"),
        ("c", "d"): EdgeStats(hit_count=1, scores=[0.5], last_seen="2025-01-02"),
    }
    await store.save_edges(buffer)

    loaded = await store.load_edges()
    assert len(loaded) == 2
    assert loaded[("a", "b")].hit_count == 3
    assert loaded[("a", "b")].scores == [0.9, 0.8, 0.7]
    assert loaded[("c", "d")].hit_count == 1


async def test_edge_upsert(store):
    from qortex.hippocampus.buffer import EdgeStats

    await store.save_edges({("a", "b"): EdgeStats(hit_count=1, scores=[0.5], last_seen="t1")})
    await store.save_edges({("a", "b"): EdgeStats(hit_count=5, scores=[0.9], last_seen="t2")})

    loaded = await store.load_edges()
    assert loaded[("a", "b")].hit_count == 5


async def test_remove_edges(store):
    from qortex.hippocampus.buffer import EdgeStats

    buffer = {
        ("a", "b"): EdgeStats(hit_count=1, scores=[0.5], last_seen="t"),
        ("c", "d"): EdgeStats(hit_count=1, scores=[0.5], last_seen="t"),
        ("e", "f"): EdgeStats(hit_count=1, scores=[0.5], last_seen="t"),
    }
    await store.save_edges(buffer)
    await store.remove_edges([("a", "b"), ("e", "f")])

    loaded = await store.load_edges()
    assert len(loaded) == 1
    assert ("c", "d") in loaded


async def test_remove_edges_empty(store):
    # Should not raise
    await store.remove_edges([])


async def test_close_is_noop(store):
    await store.close()
    # Should still work after close (pool is shared)
    factors = await store.load_factors()
    assert isinstance(factors, dict)
