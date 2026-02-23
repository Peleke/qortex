"""Tests for the shared asyncpg pool singleton."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_pool():
    """Reset the pool singleton before and after each test."""
    import qortex.core.pool as pool_mod

    pool_mod._pool = None
    yield
    pool_mod._pool = None


async def test_get_shared_pool_creates_once():
    """First call creates pool, second returns same instance."""
    import qortex.core.pool as pool_mod

    mock_pool = MagicMock()
    mock_asyncpg = MagicMock()
    mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

    with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
        p1 = await pool_mod.get_shared_pool("postgresql://test")
        p2 = await pool_mod.get_shared_pool("postgresql://test")

    assert p1 is p2
    assert p1 is mock_pool
    mock_asyncpg.create_pool.assert_called_once()


async def test_close_shared_pool():
    """close_shared_pool closes and resets the singleton."""
    import qortex.core.pool as pool_mod

    mock_pool = AsyncMock()
    pool_mod._pool = mock_pool

    await pool_mod.close_shared_pool()

    mock_pool.close.assert_called_once()
    assert pool_mod._pool is None


async def test_close_shared_pool_noop_when_none():
    """close_shared_pool does nothing when no pool exists."""
    import qortex.core.pool as pool_mod

    await pool_mod.close_shared_pool()  # should not raise
    assert pool_mod._pool is None


async def test_get_shared_pool_passes_init():
    """The init callback is forwarded to asyncpg.create_pool."""
    import qortex.core.pool as pool_mod

    init_fn = AsyncMock()
    mock_asyncpg = MagicMock()
    mock_asyncpg.create_pool = AsyncMock(return_value=MagicMock())

    with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
        await pool_mod.get_shared_pool(
            "postgresql://test", init=init_fn, min_size=3, max_size=20
        )

    mock_asyncpg.create_pool.assert_called_once_with(
        "postgresql://test",
        min_size=3,
        max_size=20,
        init=init_fn,
    )
