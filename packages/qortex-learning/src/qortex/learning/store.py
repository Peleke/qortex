"""Learning state persistence: protocol + JSON and SQLite backends.

LearningStore is the protocol. Code against it.
Primary: SqliteLearningStore (ACID, concurrent-safe, async via aiosqlite)
Fallback: JsonLearningStore (simple, async-compatible)
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite

from qortex.learning.types import ArmState, context_hash

_SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


def _sanitize_name(learner_name: str) -> str:
    """Validate learner_name is safe for use in file paths."""
    if not _SAFE_NAME.match(learner_name) or ".." in learner_name:
        raise ValueError(
            f"Invalid learner name {learner_name!r}: must be alphanumeric "
            f"with optional dots, hyphens, underscores; no path separators."
        )
    return learner_name


@runtime_checkable
class LearningStore(Protocol):
    """Protocol for learning state persistence."""

    async def get(self, arm_id: str, context: dict | None = None) -> ArmState: ...
    async def get_all(self, context: dict | None = None) -> dict[str, ArmState]: ...
    async def put(self, arm_id: str, state: ArmState, context: dict | None = None) -> None: ...
    async def get_all_contexts(self) -> list[str]: ...
    async def get_all_states(self) -> dict[str, dict[str, ArmState]]: ...
    async def delete(self, arm_ids: list[str] | None = None, context: dict | None = None) -> int: ...
    async def save(self) -> None: ...


# ---------------------------------------------------------------------------
# JSON backend
# ---------------------------------------------------------------------------


class JsonLearningStore:
    """Persists arm states as JSON, partitioned by context hash.

    File layout: {state_dir}/{learner_name}.json
    JSON structure: { context_hash: { arm_id: ArmState.to_dict() } }
    """

    def __init__(self, learner_name: str, state_dir: str = "") -> None:
        self._name = _sanitize_name(learner_name)
        if state_dir:
            self._dir = Path(state_dir)
        else:
            self._dir = Path("~/.qortex/learning").expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{self._name}.json"
        self._data: dict[str, dict[str, ArmState]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            for ctx, arms in raw.items():
                self._data[ctx] = {
                    arm_id: ArmState.from_dict(state) for arm_id, state in arms.items()
                }

    async def save(self) -> None:
        out: dict[str, dict[str, dict]] = {}
        for ctx, arms in self._data.items():
            out[ctx] = {arm_id: state.to_dict() for arm_id, state in arms.items()}

        def _write() -> None:
            self._path.write_text(json.dumps(out, indent=2))

        await asyncio.to_thread(_write)

    async def get(self, arm_id: str, context: dict | None = None) -> ArmState:
        ctx = context_hash(context or {})
        return self._data.get(ctx, {}).get(arm_id, ArmState())

    async def get_all(self, context: dict | None = None) -> dict[str, ArmState]:
        ctx = context_hash(context or {})
        return dict(self._data.get(ctx, {}))

    async def put(self, arm_id: str, state: ArmState, context: dict | None = None) -> None:
        ctx = context_hash(context or {})
        if ctx not in self._data:
            self._data[ctx] = {}
        self._data[ctx][arm_id] = state

    async def delete(self, arm_ids: list[str] | None = None, context: dict | None = None) -> int:
        """Delete arm states. Returns number of entries removed."""
        if arm_ids is not None and len(arm_ids) == 0:
            return 0
        count = 0
        if arm_ids is None and context is None:
            for arms in self._data.values():
                count += len(arms)
            self._data.clear()
        elif arm_ids is None:
            ctx = context_hash(context or {})
            if ctx in self._data:
                count = len(self._data[ctx])
                del self._data[ctx]
        elif context is None:
            ctx = context_hash({})
            if ctx in self._data:
                for arm_id in arm_ids:
                    if arm_id in self._data[ctx]:
                        del self._data[ctx][arm_id]
                        count += 1
                if not self._data[ctx]:
                    del self._data[ctx]
        else:
            ctx = context_hash(context)
            if ctx in self._data:
                for arm_id in arm_ids:
                    if arm_id in self._data[ctx]:
                        del self._data[ctx][arm_id]
                        count += 1
                if not self._data[ctx]:
                    del self._data[ctx]
        return count

    async def close(self) -> None:
        """No-op for JSON backend (no connection to close)."""
        pass

    async def get_all_contexts(self) -> list[str]:
        return list(self._data.keys())

    async def get_all_states(self) -> dict[str, dict[str, ArmState]]:
        return {ctx: dict(arms) for ctx, arms in self._data.items()}


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SqliteLearningStore:
    """ACID-safe arm state persistence with aiosqlite.

    File layout: {state_dir}/{learner_name}.db
    Context partitioning via composite primary key (context_hash, arm_id).
    Fully async — no threading.Lock needed, aiosqlite serializes internally.
    """

    def __init__(self, learner_name: str, state_dir: str = "") -> None:
        self._name = _sanitize_name(learner_name)
        if state_dir:
            self._dir = Path(state_dir)
        else:
            self._dir = Path("~/.qortex/learning").expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / f"{self._name}.db"
        self._conn: aiosqlite.Connection | None = None

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._conn is not None:
            return self._conn
        self._conn = await aiosqlite.connect(str(self._db_path))
        await self._conn.execute("PRAGMA journal_mode = WAL")
        await self._conn.execute("PRAGMA busy_timeout = 3000")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS arm_states (
                context_hash TEXT NOT NULL,
                arm_id TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                pulls INTEGER NOT NULL DEFAULT 0,
                total_reward REAL NOT NULL DEFAULT 0.0,
                last_updated TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (context_hash, arm_id)
            )
        """)
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_arm_states_context ON arm_states(context_hash)"
        )
        await self._conn.commit()
        return self._conn

    async def get(self, arm_id: str, context: dict | None = None) -> ArmState:
        conn = await self._ensure_connection()
        ctx = context_hash(context or {})
        async with conn.execute(
            "SELECT alpha, beta, pulls, total_reward, last_updated "
            "FROM arm_states WHERE context_hash = ? AND arm_id = ?",
            (ctx, arm_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return ArmState()
        return ArmState(
            alpha=row[0],
            beta=row[1],
            pulls=row[2],
            total_reward=row[3],
            last_updated=row[4],
        )

    async def get_all(self, context: dict | None = None) -> dict[str, ArmState]:
        conn = await self._ensure_connection()
        ctx = context_hash(context or {})
        async with conn.execute(
            "SELECT arm_id, alpha, beta, pulls, total_reward, last_updated "
            "FROM arm_states WHERE context_hash = ?",
            (ctx,),
        ) as cursor:
            rows = await cursor.fetchall()
        return {
            row[0]: ArmState(
                alpha=row[1],
                beta=row[2],
                pulls=row[3],
                total_reward=row[4],
                last_updated=row[5],
            )
            for row in rows
        }

    async def put(self, arm_id: str, state: ArmState, context: dict | None = None) -> None:
        conn = await self._ensure_connection()
        ctx = context_hash(context or {})
        await conn.execute(
            "INSERT OR REPLACE INTO arm_states "
            "(context_hash, arm_id, alpha, beta, pulls, total_reward, last_updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ctx,
                arm_id,
                state.alpha,
                state.beta,
                state.pulls,
                state.total_reward,
                state.last_updated,
            ),
        )
        await conn.commit()

    async def get_all_contexts(self) -> list[str]:
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT DISTINCT context_hash FROM arm_states"
        ) as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_all_states(self) -> dict[str, dict[str, ArmState]]:
        conn = await self._ensure_connection()
        async with conn.execute(
            "SELECT context_hash, arm_id, alpha, beta, pulls, total_reward, last_updated "
            "FROM arm_states"
        ) as cursor:
            rows = await cursor.fetchall()
        result: dict[str, dict[str, ArmState]] = {}
        for row in rows:
            ctx = row[0]
            if ctx not in result:
                result[ctx] = {}
            result[ctx][row[1]] = ArmState(
                alpha=row[2],
                beta=row[3],
                pulls=row[4],
                total_reward=row[5],
                last_updated=row[6],
            )
        return result

    async def delete(self, arm_ids: list[str] | None = None, context: dict | None = None) -> int:
        """Delete arm states. Returns number of entries removed."""
        if arm_ids is not None and len(arm_ids) == 0:
            return 0
        conn = await self._ensure_connection()
        if arm_ids is None and context is None:
            cursor = await conn.execute("DELETE FROM arm_states")
        elif arm_ids is None:
            ctx = context_hash(context or {})
            cursor = await conn.execute(
                "DELETE FROM arm_states WHERE context_hash = ?", (ctx,)
            )
        elif context is None:
            ctx = context_hash({})
            placeholders = ",".join("?" for _ in arm_ids)
            cursor = await conn.execute(
                f"DELETE FROM arm_states WHERE context_hash = ? AND arm_id IN ({placeholders})",
                (ctx, *arm_ids),
            )
        else:
            ctx = context_hash(context)
            placeholders = ",".join("?" for _ in arm_ids)
            cursor = await conn.execute(
                f"DELETE FROM arm_states WHERE context_hash = ? AND arm_id IN ({placeholders})",
                (ctx, *arm_ids),
            )
        await conn.commit()
        return cursor.rowcount

    async def close(self) -> None:
        """Close the underlying aiosqlite connection.

        Call this during teardown to avoid RuntimeError('Event loop is closed')
        from aiosqlite's background thread.
        """
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None

    async def save(self) -> None:
        if self._conn is not None:
            await self._conn.commit()
