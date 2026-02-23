"""PostgresInteroceptionStore: asyncpg persistence for interoception state.

Same semantics as InteroceptionStore (SQLite), but backed by shared asyncpg pool.
All methods are async. Pool is externally managed (shared singleton).

Requires: asyncpg (already in qortex[vec-pgvector] or qortex[source-postgres])
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from qortex.observe.logging import get_logger

if TYPE_CHECKING:
    from qortex.hippocampus.buffer import EdgeStats

logger = get_logger(__name__)


@runtime_checkable
class AsyncInteroceptionStore(Protocol):
    """Async protocol for interoception state persistence.

    Mirror of InteroceptionStore with async methods for postgres backend.
    """

    async def load_factors(self) -> dict[str, float]: ...
    async def save_factor(self, node_id: str, weight: float) -> None: ...
    async def save_factors(self, factors: dict[str, float]) -> None: ...
    async def load_edges(self) -> dict[tuple[str, str], EdgeStats]: ...
    async def save_edges(self, buffer: dict[tuple[str, str], EdgeStats]) -> None: ...
    async def remove_edges(self, keys: list[tuple[str, str]]) -> None: ...
    async def close(self) -> None: ...


class PostgresInteroceptionStore:
    """Asyncpg-backed persistence for interoception state.

    Uses a shared asyncpg pool. Schema is auto-created on first access.

    Tables:
    - interoception_factors: node_id -> weight (PPR teleportation)
    - interoception_edge_buffer: (src_id, tgt_id) -> EdgeStats
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS interoception_factors (
                    node_id    TEXT PRIMARY KEY,
                    weight     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS interoception_edge_buffer (
                    src_id    TEXT NOT NULL,
                    tgt_id    TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    scores    JSONB NOT NULL DEFAULT '[]'::jsonb,
                    last_seen TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (src_id, tgt_id)
                )
            """)

        self._schema_ready = True

    # -- Teleportation factors ------------------------------------------------

    async def load_factors(self) -> dict[str, float]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT node_id, weight FROM interoception_factors"
            )
        return {row["node_id"]: float(row["weight"]) for row in rows}

    async def save_factor(self, node_id: str, weight: float) -> None:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interoception_factors (node_id, weight, updated_at)
                VALUES ($1, $2, now())
                ON CONFLICT (node_id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    updated_at = now()
                """,
                node_id,
                weight,
            )

    async def save_factors(self, factors: dict[str, float]) -> None:
        if not factors:
            return
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO interoception_factors (node_id, weight, updated_at)
                VALUES ($1, $2, now())
                ON CONFLICT (node_id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    updated_at = now()
                """,
                [(nid, w) for nid, w in factors.items()],
            )

    # -- Edge buffer ----------------------------------------------------------

    async def load_edges(self) -> dict[tuple[str, str], EdgeStats]:
        from qortex.hippocampus.buffer import EdgeStats as ES

        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT src_id, tgt_id, hit_count, scores, last_seen "
                "FROM interoception_edge_buffer"
            )

        result: dict[tuple[str, str], ES] = {}
        for row in rows:
            scores = row["scores"] if isinstance(row["scores"], list) else []
            last_seen = row["last_seen"].isoformat() if row["last_seen"] else ""
            result[(row["src_id"], row["tgt_id"])] = ES(
                hit_count=row["hit_count"],
                scores=scores,
                last_seen=last_seen,
            )
        return result

    async def save_edges(self, buffer: dict[tuple[str, str], EdgeStats]) -> None:
        if not buffer:
            return
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO interoception_edge_buffer
                    (src_id, tgt_id, hit_count, scores, last_seen)
                VALUES ($1, $2, $3, $4::jsonb, now())
                ON CONFLICT (src_id, tgt_id) DO UPDATE SET
                    hit_count = EXCLUDED.hit_count,
                    scores = EXCLUDED.scores,
                    last_seen = now()
                """,
                [
                    (src, tgt, stats.hit_count, json.dumps(stats.scores))
                    for (src, tgt), stats in buffer.items()
                ],
            )

    async def remove_edges(self, keys: list[tuple[str, str]]) -> None:
        if not keys:
            return
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            await conn.executemany(
                "DELETE FROM interoception_edge_buffer WHERE src_id = $1 AND tgt_id = $2",
                keys,
            )

    # -- Lifecycle ------------------------------------------------------------

    async def close(self) -> None:
        """No-op — pool is shared and managed externally."""
