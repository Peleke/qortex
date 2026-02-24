"""PostgresLearningStore: asyncpg persistence for bandit arm states.

Same protocol as SqliteLearningStore, but backed by shared asyncpg pool.
Key difference: learner_name is part of the composite PK (one table for all learners).

Requires: asyncpg (provided by qortex[vec-pgvector] or standalone)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qortex.learning.types import ArmState, context_hash
from qortex.observe.tracing import traced


class PostgresLearningStore:
    """Asyncpg-backed arm state persistence.

    Unlike SQLite (one .db per learner), Postgres shares one table
    with learner_name in the composite primary key.
    """

    def __init__(self, learner_name: str, pool: Any) -> None:
        self._name = learner_name
        self._pool = pool
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_arm_states (
                    learner_name TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    arm_id       TEXT NOT NULL,
                    alpha        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    beta         DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    pulls        INTEGER NOT NULL DEFAULT 0,
                    total_reward DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    last_updated TIMESTAMPTZ DEFAULT now(),
                    PRIMARY KEY (learner_name, context_hash, arm_id)
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_learning_arm_states_context
                    ON learning_arm_states(learner_name, context_hash)
            """)

        self._schema_ready = True

    @traced("learning.pg.get")
    async def get(self, arm_id: str, context: dict | None = None) -> ArmState:
        await self._ensure_schema()
        ctx = context_hash(context or {})
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT alpha, beta, pulls, total_reward, last_updated "
                "FROM learning_arm_states "
                "WHERE learner_name = $1 AND context_hash = $2 AND arm_id = $3",
                self._name,
                ctx,
                arm_id,
            )
        if row is None:
            return ArmState()
        return ArmState(
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            pulls=row["pulls"],
            total_reward=float(row["total_reward"]),
            last_updated=row["last_updated"].isoformat() if row["last_updated"] else "",
        )

    @traced("learning.pg.get_all")
    async def get_all(self, context: dict | None = None) -> dict[str, ArmState]:
        await self._ensure_schema()
        ctx = context_hash(context or {})
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT arm_id, alpha, beta, pulls, total_reward, last_updated "
                "FROM learning_arm_states "
                "WHERE learner_name = $1 AND context_hash = $2",
                self._name,
                ctx,
            )
        return {
            row["arm_id"]: ArmState(
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                pulls=row["pulls"],
                total_reward=float(row["total_reward"]),
                last_updated=row["last_updated"].isoformat() if row["last_updated"] else "",
            )
            for row in rows
        }

    @traced("learning.pg.put")
    async def put(
        self, arm_id: str, state: ArmState, context: dict | None = None
    ) -> None:
        await self._ensure_schema()
        ctx = context_hash(context or {})
        ts = datetime.fromisoformat(state.last_updated) if state.last_updated else datetime.now(UTC)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO learning_arm_states
                    (learner_name, context_hash, arm_id, alpha, beta, pulls, total_reward, last_updated)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (learner_name, context_hash, arm_id) DO UPDATE SET
                    alpha = EXCLUDED.alpha,
                    beta = EXCLUDED.beta,
                    pulls = EXCLUDED.pulls,
                    total_reward = EXCLUDED.total_reward,
                    last_updated = EXCLUDED.last_updated
                """,
                self._name,
                ctx,
                arm_id,
                state.alpha,
                state.beta,
                state.pulls,
                state.total_reward,
                ts,
            )

    @traced("learning.pg.get_all_contexts")
    async def get_all_contexts(self) -> list[str]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT context_hash FROM learning_arm_states "
                "WHERE learner_name = $1",
                self._name,
            )
        return [row["context_hash"] for row in rows]

    @traced("learning.pg.get_all_states")
    async def get_all_states(self) -> dict[str, dict[str, ArmState]]:
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT context_hash, arm_id, alpha, beta, pulls, total_reward, last_updated "
                "FROM learning_arm_states "
                "WHERE learner_name = $1",
                self._name,
            )
        result: dict[str, dict[str, ArmState]] = {}
        for row in rows:
            ctx = row["context_hash"]
            if ctx not in result:
                result[ctx] = {}
            result[ctx][row["arm_id"]] = ArmState(
                alpha=float(row["alpha"]),
                beta=float(row["beta"]),
                pulls=row["pulls"],
                total_reward=float(row["total_reward"]),
                last_updated=row["last_updated"].isoformat() if row["last_updated"] else "",
            )
        return result

    @traced("learning.pg.delete")
    async def delete(
        self, arm_ids: list[str] | None = None, context: dict | None = None
    ) -> int:
        if arm_ids is not None and len(arm_ids) == 0:
            return 0
        await self._ensure_schema()
        async with self._pool.acquire() as conn:
            if arm_ids is None and context is None:
                result = await conn.execute(
                    "DELETE FROM learning_arm_states WHERE learner_name = $1",
                    self._name,
                )
            elif arm_ids is None:
                ctx = context_hash(context or {})
                result = await conn.execute(
                    "DELETE FROM learning_arm_states "
                    "WHERE learner_name = $1 AND context_hash = $2",
                    self._name,
                    ctx,
                )
            elif context is None:
                ctx = context_hash({})
                result = await conn.execute(
                    "DELETE FROM learning_arm_states "
                    "WHERE learner_name = $1 AND context_hash = $2 "
                    "AND arm_id = ANY($3::text[])",
                    self._name,
                    ctx,
                    arm_ids,
                )
            else:
                ctx = context_hash(context)
                result = await conn.execute(
                    "DELETE FROM learning_arm_states "
                    "WHERE learner_name = $1 AND context_hash = $2 "
                    "AND arm_id = ANY($3::text[])",
                    self._name,
                    ctx,
                    arm_ids,
                )
        # asyncpg returns "DELETE N"
        return int(result.split()[-1]) if result else 0

    async def save(self) -> None:
        """No-op — Postgres auto-commits per statement."""
