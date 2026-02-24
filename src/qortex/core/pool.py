"""Shared asyncpg pool singleton for all postgres-backed stores.

One pool, one database, three subsystems (vec, interoception, learning).
Tests create isolated pools directly — this singleton is for production wiring.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from qortex.observe.logging import get_logger

logger = get_logger(__name__)

_pool: Any = None
_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Lazy-init the asyncio lock (must be created inside a running loop)."""
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def get_shared_pool(
    dsn: str,
    *,
    min_size: int = 2,
    max_size: int = 10,
    init: Callable | None = None,
) -> Any:
    """Get or create the shared asyncpg connection pool.

    First call creates the pool; subsequent calls return the same instance.
    The ``init`` callback runs on each new connection (e.g. pgvector codec registration).
    """
    global _pool
    if _pool is not None:
        return _pool

    async with _get_lock():
        # Double-check after acquiring lock
        if _pool is not None:
            return _pool

        import asyncpg

        _pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            init=init,
        )
        logger.info(
            "shared_pool.created",
            dsn_host=dsn.split("@")[-1] if "@" in dsn else "local",
            max_size=max_size,
        )
        return _pool


async def close_shared_pool() -> None:
    """Close the shared pool if it exists. Safe to call multiple times."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("shared_pool.closed")
