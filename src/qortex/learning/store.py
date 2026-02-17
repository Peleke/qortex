"""Learning state persistence: protocol + JSON and SQLite backends.

LearningStore is the protocol. Code against it.
Primary: SqliteLearningStore (ACID, concurrent-safe)
Fallback: JsonLearningStore (simple, no locking)
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

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

    def get(self, arm_id: str, context: dict | None = None) -> ArmState: ...
    def get_all(self, context: dict | None = None) -> dict[str, ArmState]: ...
    def put(self, arm_id: str, state: ArmState, context: dict | None = None) -> None: ...
    def get_all_contexts(self) -> list[str]: ...
    def get_all_states(self) -> dict[str, dict[str, ArmState]]: ...
    def delete(self, arm_ids: list[str] | None = None, context: dict | None = None) -> int: ...
    def save(self) -> None: ...


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

    def save(self) -> None:
        out: dict[str, dict[str, dict]] = {}
        for ctx, arms in self._data.items():
            out[ctx] = {arm_id: state.to_dict() for arm_id, state in arms.items()}
        self._path.write_text(json.dumps(out, indent=2))

    def get(self, arm_id: str, context: dict | None = None) -> ArmState:
        ctx = context_hash(context or {})
        return self._data.get(ctx, {}).get(arm_id, ArmState())

    def get_all(self, context: dict | None = None) -> dict[str, ArmState]:
        ctx = context_hash(context or {})
        return dict(self._data.get(ctx, {}))

    def put(self, arm_id: str, state: ArmState, context: dict | None = None) -> None:
        ctx = context_hash(context or {})
        if ctx not in self._data:
            self._data[ctx] = {}
        self._data[ctx][arm_id] = state

    def delete(self, arm_ids: list[str] | None = None, context: dict | None = None) -> int:
        """Delete arm states. Returns number of entries removed."""
        if arm_ids is not None and len(arm_ids) == 0:
            return 0
        count = 0
        if arm_ids is None and context is None:
            # Full reset
            for arms in self._data.values():
                count += len(arms)
            self._data.clear()
        elif arm_ids is None:
            # Delete all arms for a specific context
            ctx = context_hash(context or {})
            if ctx in self._data:
                count = len(self._data[ctx])
                del self._data[ctx]
        elif context is None:
            # Delete specific arms in default context
            ctx = context_hash({})
            if ctx in self._data:
                for arm_id in arm_ids:
                    if arm_id in self._data[ctx]:
                        del self._data[ctx][arm_id]
                        count += 1
                if not self._data[ctx]:
                    del self._data[ctx]
        else:
            # Delete specific arms in specific context
            ctx = context_hash(context)
            if ctx in self._data:
                for arm_id in arm_ids:
                    if arm_id in self._data[ctx]:
                        del self._data[ctx][arm_id]
                        count += 1
                if not self._data[ctx]:
                    del self._data[ctx]
        return count

    def get_all_contexts(self) -> list[str]:
        return list(self._data.keys())

    def get_all_states(self) -> dict[str, dict[str, ArmState]]:
        return {ctx: dict(arms) for ctx, arms in self._data.items()}


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


class SqliteLearningStore:
    """ACID-safe arm state persistence with SQLite.

    File layout: {state_dir}/{learner_name}.db
    Context partitioning via composite primary key (context_hash, arm_id).
    """

    def __init__(self, learner_name: str, state_dir: str = "") -> None:
        self._name = _sanitize_name(learner_name)
        if state_dir:
            self._dir = Path(state_dir)
        else:
            self._dir = Path("~/.qortex/learning").expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / f"{self._name}.db"
        self._conn: sqlite3.Connection | None = None
        # Serialize all connection access. check_same_thread=False allows
        # cross-thread usage but does NOT serialize it; concurrent
        # conn.execute() calls from FastMCP's thread pool cause
        # InterfaceError ("bad parameter or other API misuse").
        self._lock = threading.Lock()

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        # check_same_thread=False: FastMCP dispatches tool handlers on a
        # thread pool, so consecutive calls (select, observe) may land on
        # different threads AND run concurrently. All public methods acquire
        # self._lock before touching the connection. WAL mode allows
        # concurrent readers at the SQLite level if needed in the future.
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 3000")
        self._conn.execute("""
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
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_arm_states_context ON arm_states(context_hash)"
        )
        self._conn.commit()
        return self._conn

    def get(self, arm_id: str, context: dict | None = None) -> ArmState:
        with self._lock:
            conn = self._ensure_connection()
            ctx = context_hash(context or {})
            row = conn.execute(
                "SELECT alpha, beta, pulls, total_reward, last_updated "
                "FROM arm_states WHERE context_hash = ? AND arm_id = ?",
                (ctx, arm_id),
            ).fetchone()
            if row is None:
                return ArmState()
            return ArmState(
                alpha=row[0],
                beta=row[1],
                pulls=row[2],
                total_reward=row[3],
                last_updated=row[4],
            )

    def get_all(self, context: dict | None = None) -> dict[str, ArmState]:
        with self._lock:
            conn = self._ensure_connection()
            ctx = context_hash(context or {})
            rows = conn.execute(
                "SELECT arm_id, alpha, beta, pulls, total_reward, last_updated "
                "FROM arm_states WHERE context_hash = ?",
                (ctx,),
            ).fetchall()
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

    def put(self, arm_id: str, state: ArmState, context: dict | None = None) -> None:
        with self._lock:
            conn = self._ensure_connection()
            ctx = context_hash(context or {})
            conn.execute(
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
            conn.commit()

    def get_all_contexts(self) -> list[str]:
        with self._lock:
            conn = self._ensure_connection()
            rows = conn.execute("SELECT DISTINCT context_hash FROM arm_states").fetchall()
            return [row[0] for row in rows]

    def get_all_states(self) -> dict[str, dict[str, ArmState]]:
        with self._lock:
            conn = self._ensure_connection()
            rows = conn.execute(
                "SELECT context_hash, arm_id, alpha, beta, pulls, total_reward, last_updated "
                "FROM arm_states"
            ).fetchall()
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

    def delete(self, arm_ids: list[str] | None = None, context: dict | None = None) -> int:
        """Delete arm states. Returns number of entries removed."""
        if arm_ids is not None and len(arm_ids) == 0:
            return 0
        with self._lock:
            conn = self._ensure_connection()
            if arm_ids is None and context is None:
                cursor = conn.execute("DELETE FROM arm_states")
            elif arm_ids is None:
                ctx = context_hash(context or {})
                cursor = conn.execute("DELETE FROM arm_states WHERE context_hash = ?", (ctx,))
            elif context is None:
                ctx = context_hash({})
                placeholders = ",".join("?" for _ in arm_ids)
                cursor = conn.execute(
                    f"DELETE FROM arm_states WHERE context_hash = ? AND arm_id IN ({placeholders})",
                    (ctx, *arm_ids),
                )
            else:
                ctx = context_hash(context)
                placeholders = ",".join("?" for _ in arm_ids)
                cursor = conn.execute(
                    f"DELETE FROM arm_states WHERE context_hash = ? AND arm_id IN ({placeholders})",
                    (ctx, *arm_ids),
                )
            conn.commit()
            return cursor.rowcount

    def save(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.commit()
