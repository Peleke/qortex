"""Shared asyncpg pool singleton for all postgres-backed stores.

One pool, one database, three subsystems (vec, interoception, learning).
Tests create isolated pools directly — this singleton is for production wiring.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_pool: Any = None


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

    import asyncpg

    _pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        init=init,
    )
    logger.info("shared_pool.created", extra={"dsn": dsn.split("@")[-1], "max_size": max_size})
    return _pool


async def close_shared_pool() -> None:
    """Close the shared pool if it exists. Safe to call multiple times."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("shared_pool.closed")
