"""InteroceptionStore: SQLite persistence for interoception state.

Replaces flat JSON files (factors.json, edge_buffer.json) with a single
SQLite database using WAL mode for crash safety and concurrent access.

Same pattern as SqliteLearningStore: WAL, busy_timeout, threading.Lock,
check_same_thread=False.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from qortex.observe.logging import get_logger

if TYPE_CHECKING:
    from qortex.hippocampus.buffer import EdgeStats

logger = get_logger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS teleportation_factors (
    node_id    TEXT PRIMARY KEY,
    weight     REAL NOT NULL DEFAULT 1.0,
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edge_buffer (
    src_id    TEXT NOT NULL,
    tgt_id    TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    scores    TEXT NOT NULL DEFAULT '[]',
    last_seen TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (src_id, tgt_id)
);
"""


class InteroceptionStore:
    """SQLite-backed persistence for interoception state.

    Manages two tables:
    - teleportation_factors: node_id -> weight mappings
    - edge_buffer: (src_id, tgt_id) -> EdgeStats
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 3000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- Teleportation factors ------------------------------------------------

    def load_factors(self) -> dict[str, float]:
        """Load all teleportation factors from the database."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT node_id, weight FROM teleportation_factors"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def save_factor(self, node_id: str, weight: float) -> None:
        """Upsert a single teleportation factor (for persist_on_update)."""
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO teleportation_factors (node_id, weight, updated_at) "
                "VALUES (?, ?, ?)",
                (node_id, weight, now),
            )
            self._conn.commit()

    def save_factors(self, factors: dict[str, float]) -> None:
        """Batch upsert all teleportation factors (for shutdown)."""
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO teleportation_factors (node_id, weight, updated_at) "
                "VALUES (?, ?, ?)",
                [(nid, w, now) for nid, w in factors.items()],
            )
            self._conn.commit()

    # -- Edge buffer ----------------------------------------------------------

    def load_edges(self) -> dict[tuple[str, str], EdgeStats]:
        """Load all buffered edges from the database."""
        from qortex.hippocampus.buffer import EdgeStats as ES

        with self._lock:
            rows = self._conn.execute(
                "SELECT src_id, tgt_id, hit_count, scores, last_seen FROM edge_buffer"
            ).fetchall()

        result: dict[tuple[str, str], ES] = {}
        for src_id, tgt_id, hit_count, scores_json, last_seen in rows:
            try:
                scores = json.loads(scores_json)
            except (json.JSONDecodeError, TypeError):
                scores = []
            result[(src_id, tgt_id)] = ES(
                hit_count=hit_count,
                scores=scores,
                last_seen=last_seen,
            )
        return result

    def save_edges(self, buffer: dict[tuple[str, str], EdgeStats]) -> None:
        """Batch upsert all buffered edges."""
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO edge_buffer (src_id, tgt_id, hit_count, scores, last_seen) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (src, tgt, stats.hit_count, json.dumps(stats.scores), stats.last_seen)
                    for (src, tgt), stats in buffer.items()
                ],
            )
            self._conn.commit()

    def remove_edges(self, keys: list[tuple[str, str]]) -> None:
        """Delete promoted edges from the buffer table."""
        if not keys:
            return
        with self._lock:
            self._conn.executemany(
                "DELETE FROM edge_buffer WHERE src_id = ? AND tgt_id = ?",
                keys,
            )
            self._conn.commit()

    # -- Lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Commit pending changes and close the connection."""
        with self._lock:
            self._conn.commit()
            self._conn.close()
