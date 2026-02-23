"""PgVector-backed async vector index.

Uses asyncpg for native async PostgreSQL access with the pgvector extension.
HNSW index for fast approximate nearest neighbor search.

Requires: pip install qortex[vec-pgvector]
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np

from qortex.observe.tracing import traced

_SAFE_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")

logger = logging.getLogger(__name__)


def _try_emit(event: Any) -> None:
    """Emit an observability event if the emitter is configured."""
    try:
        from qortex.observe import emit

        emit(event)
    except Exception:
        pass


class PgVectorIndex:
    """Async vector index backed by pgvector.

    Uses asyncpg with the pgvector codec for native async PostgreSQL access.
    Creates an HNSW index for fast approximate nearest neighbor search.

    Requires:
        PostgreSQL with pgvector extension installed.
        pip install qortex[vec-pgvector]
    """

    def __init__(
        self,
        dsn: str,
        *,
        dimensions: int = 384,
        table_name: str = "qortex_vectors",
        pool_size: int = 10,
    ) -> None:
        if not _SAFE_IDENT.match(table_name):
            raise ValueError(
                f"table_name must be a safe SQL identifier (lowercase alphanumeric + underscores), "
                f"got: {table_name!r}"
            )
        self._dsn = dsn
        self._dimensions = dimensions
        self._table_name = table_name
        self._pool_size = pool_size
        self._pool: Any = None
        self._schema_ready = False

    async def _ensure_pool(self) -> None:
        """Lazily create the connection pool with pgvector codec registered."""
        if self._pool is not None:
            return

        import asyncpg
        from pgvector.asyncpg import register_vector

        async def _init_connection(conn: asyncpg.Connection) -> None:
            await register_vector(conn)

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=1,
            max_size=self._pool_size,
            init=_init_connection,
        )

    async def _ensure_schema(self) -> None:
        """Create extension, table, and HNSW index if not present."""
        if self._schema_ready:
            return

        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({self._dimensions}),
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_embedding
                ON {self._table_name}
                USING hnsw (embedding vector_cosine_ops)
            """)

        self._schema_ready = True

    @traced("vec.add")
    async def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Add vectors with upsert semantics."""
        t0 = time.perf_counter()
        if len(ids) != len(embeddings):
            raise ValueError(f"ids ({len(ids)}) and embeddings ({len(embeddings)}) must match")

        await self._ensure_schema()

        vectors = [np.array(e, dtype=np.float32) for e in embeddings]

        async with self._pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._table_name} (id, embedding)
                VALUES ($1, $2)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = now()
                """,
                list(zip(ids, vectors)),
            )

        total = await self.size()

        from qortex.observe.events import VecIndexUpdated

        _try_emit(
            VecIndexUpdated(
                count_added=len(ids),
                total_size=total,
                latency_ms=(time.perf_counter() - t0) * 1000,
                index_type="pgvector",
            )
        )

    @traced("vec.search")
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search using HNSW cosine distance."""
        t0 = time.perf_counter()
        await self._ensure_schema()

        query_vec = np.array(query_embedding, dtype=np.float32)

        async with self._pool.acquire() as conn:
            # pgvector <=> is cosine distance (1 - similarity).
            # ORDER BY distance uses the HNSW index; threshold filter in outer query.
            rows = await conn.fetch(
                f"""
                SELECT id, similarity FROM (
                    SELECT id, 1 - (embedding <=> $1) AS similarity
                    FROM {self._table_name}
                    ORDER BY embedding <=> $1
                    LIMIT $2
                ) sub
                WHERE similarity >= $3
                """,
                query_vec,
                top_k,
                threshold,
            )

        results = [(row["id"], float(row["similarity"])) for row in rows]

        elapsed = (time.perf_counter() - t0) * 1000
        top_score = results[0][1] if results else 0.0
        bottom_score = results[-1][1] if results else 0.0

        from qortex.observe.events import VecSearchResults

        _try_emit(
            VecSearchResults(
                candidates=len(results),
                top_score=top_score,
                score_spread=top_score - bottom_score,
                latency_ms=elapsed,
                index_type="pgvector",
            )
        )

        return results

    @traced("vec.remove")
    async def remove(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        await self._ensure_schema()

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._table_name} WHERE id = ANY($1::text[])",
                ids,
            )

    async def size(self) -> int:
        """Count of vectors in the index."""
        await self._ensure_schema()

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {self._table_name}")
            return count or 0

    async def persist(self) -> None:
        """No-op — PostgreSQL auto-commits."""

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._schema_ready = False
