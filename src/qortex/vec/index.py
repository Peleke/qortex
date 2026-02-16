"""Vector index abstractions and implementations."""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

from qortex.observe.tracing import traced

logger = logging.getLogger(__name__)


def _try_emit(event) -> None:
    """Emit an observability event if the emitter is configured."""
    try:
        from qortex.observe import emit

        emit(event)
    except Exception:
        pass  # observability is optional


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector similarity search."""

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Add vectors to the index.

        Args:
            ids: Unique identifiers for each vector.
            embeddings: Vectors to add (must match index dimensions).
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity (0.0 to 1.0).

        Returns:
            List of (id, score) tuples sorted by descending similarity.
        """
        ...

    def remove(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        ...

    def size(self) -> int:
        """Number of vectors in the index."""
        ...

    def persist(self) -> None:
        """Persist index to storage. No-op for in-memory implementations."""
        ...


class NumpyVectorIndex:
    """In-memory brute-force cosine similarity index.

    Uses numpy for batch cosine sim. Zero external deps beyond numpy.

    Good for:
    - Testing and mocking (no persistence = predictable state)
    - Ephemeral agent runs (nothing persists = security value)
    - Small datasets (< 100k vectors)

    Not suitable for:
    - Production persistence (use SqliteVecIndex)
    - Large-scale search (no approximate NN)
    """

    def __init__(self, dimensions: int) -> None:
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("numpy required: pip install qortex[vec]") from e

        self._np = np
        self._dimensions = dimensions
        self._ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}
        # Stacked matrix: (n_vectors, dimensions), normalized for cosine sim
        self._matrix: np.ndarray = np.zeros((0, dimensions), dtype=np.float32)

    @traced("vec.add")
    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Add vectors. Overwrites if ID already exists."""
        t0 = time.perf_counter()
        if len(ids) != len(embeddings):
            raise ValueError(f"ids ({len(ids)}) and embeddings ({len(embeddings)}) must match")

        np = self._np

        # Remove existing IDs first (upsert semantics)
        existing = [i for i in ids if i in self._id_to_idx]
        if existing:
            self.remove(existing)

        new_vecs = np.array(embeddings, dtype=np.float32)
        if new_vecs.shape[1] != self._dimensions:
            raise ValueError(f"Expected {self._dimensions} dimensions, got {new_vecs.shape[1]}")

        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        new_vecs = new_vecs / norms

        # Append
        start_idx = len(self._ids)
        self._ids.extend(ids)
        for i, id_ in enumerate(ids):
            self._id_to_idx[id_] = start_idx + i

        if self._matrix.shape[0] == 0:
            self._matrix = new_vecs
        else:
            self._matrix = np.vstack([self._matrix, new_vecs])

        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span.set_attribute("vec.count_added", len(ids))
            span.set_attribute("vec.total_size", len(self._ids))
        except ImportError:
            pass

        from qortex.observe.events import VecIndexUpdated

        _try_emit(
            VecIndexUpdated(
                count_added=len(ids),
                total_size=len(self._ids),
                latency_ms=(time.perf_counter() - t0) * 1000,
                index_type="numpy",
            )
        )

    @traced("vec.search")
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Brute-force cosine similarity search."""
        t0 = time.perf_counter()
        if len(self._ids) == 0:
            return []

        np = self._np
        query = np.array(query_embedding, dtype=np.float32)

        # Normalize query
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        # Cosine similarity = dot product of normalized vectors
        scores = self._matrix @ query

        # Filter by threshold and get top-k
        mask = scores >= threshold
        valid_indices = np.where(mask)[0]
        valid_scores = scores[valid_indices]

        # Sort by descending score
        top_indices = valid_indices[np.argsort(-valid_scores)[:top_k]]
        top_scores = scores[top_indices]

        results = [(self._ids[idx], float(top_scores[i])) for i, idx in enumerate(top_indices)]

        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span.set_attribute("vec.top_k", top_k)
            span.set_attribute("vec.threshold", threshold)
            span.set_attribute("vec.result_count", len(results))
            if results:
                span.set_attribute("vec.top_score", results[0][1])
        except ImportError:
            pass

        # Vec observability: search quality signal
        elapsed = (time.perf_counter() - t0) * 1000
        top_score = float(top_scores[0]) if len(top_scores) > 0 else 0.0
        bottom_score = float(top_scores[-1]) if len(top_scores) > 0 else 0.0
        from qortex.observe.events import VecSearchResults

        _try_emit(
            VecSearchResults(
                candidates=len(results),
                top_score=top_score,
                score_spread=top_score - bottom_score,
                latency_ms=elapsed,
                index_type="numpy",
            )
        )

        return results

    @traced("vec.remove")
    def remove(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        np = self._np
        indices_to_remove = set()
        for id_ in ids:
            if id_ in self._id_to_idx:
                indices_to_remove.add(self._id_to_idx[id_])

        if not indices_to_remove:
            return

        # Build mask of indices to keep
        keep_mask = np.ones(len(self._ids), dtype=bool)
        for idx in indices_to_remove:
            keep_mask[idx] = False

        # Rebuild
        new_ids = [id_ for i, id_ in enumerate(self._ids) if keep_mask[i]]
        self._matrix = self._matrix[keep_mask]
        self._ids = new_ids
        self._id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

    def size(self) -> int:
        return len(self._ids)

    def persist(self) -> None:
        """No-op for in-memory index."""
        pass


class SqliteVecIndex:
    """Persistent vector index backed by sqlite-vec.

    Production default. Same library OpenClaw uses (battle-tested).
    Stored at configurable path (default: .qortex/vectors.db).

    Requires: pip install qortex[vec-sqlite]
    """

    def __init__(self, db_path: str = ".qortex/vectors.db", dimensions: int = 384) -> None:
        self._db_path = db_path
        self._dimensions = dimensions
        self._conn = None
        self._lock = __import__("threading").Lock()

    def _ensure_connection(self):
        if self._conn is not None:
            return

        import sqlite3
        from pathlib import Path

        try:
            import sqlite_vec
        except ImportError as e:
            raise ImportError("sqlite-vec required: pip install qortex[vec-sqlite]") from e

        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # FastMCP dispatches tool handlers on a thread pool — consecutive
        # calls may land on different threads.  check_same_thread=False lets
        # the single connection be used cross-thread; the _lock serialises
        # access so SQLite never sees concurrent writes.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Create tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_meta (
                id TEXT PRIMARY KEY,
                row_id INTEGER NOT NULL
            )
        """)
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_index
            USING vec0(embedding float[{self._dimensions}])
        """)
        self._conn.commit()

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        """Add vectors with upsert semantics."""
        import struct

        t0 = time.perf_counter()
        with self._lock:
            self._ensure_connection()
            assert self._conn is not None

            # Remove existing first (upsert)
            existing = []
            for id_ in ids:
                row = self._conn.execute("SELECT row_id FROM vec_meta WHERE id = ?", (id_,)).fetchone()
                if row:
                    existing.append(id_)
            if existing:
                self._remove_locked(existing)

            for id_, emb in zip(ids, embeddings):
                if len(emb) != self._dimensions:
                    raise ValueError(f"Expected {self._dimensions} dims, got {len(emb)}")

                # sqlite-vec expects binary float32
                blob = struct.pack(f"{len(emb)}f", *emb)
                cursor = self._conn.execute("INSERT INTO vec_index(embedding) VALUES (?)", (blob,))
                row_id = cursor.lastrowid
                self._conn.execute("INSERT INTO vec_meta(id, row_id) VALUES (?, ?)", (id_, row_id))

            self._conn.commit()

        from qortex.observe.events import VecIndexUpdated

        _try_emit(
            VecIndexUpdated(
                count_added=len(ids),
                total_size=self.size(),
                latency_ms=(time.perf_counter() - t0) * 1000,
                index_type="sqlite",
            )
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search using sqlite-vec's built-in distance functions."""
        import struct

        t0 = time.perf_counter()
        with self._lock:
            self._ensure_connection()
            assert self._conn is not None

            blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)

            # sqlite-vec returns distance (lower = more similar)
            # We convert to cosine similarity: sim = 1 - distance
            rows = self._conn.execute(
                """
                SELECT vec_meta.id, vec_index.distance
                FROM vec_index
                JOIN vec_meta ON vec_meta.row_id = vec_index.rowid
                WHERE embedding MATCH ?
                AND k = ?
                """,
                (blob, top_k),
            ).fetchall()

        results = []
        for id_, distance in rows:
            similarity = 1.0 - distance
            if similarity >= threshold:
                results.append((id_, similarity))

        # Vec observability: search quality signal
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
                index_type="sqlite",
            )
        )

        return results

    def _remove_locked(self, ids: list[str]) -> None:
        """Remove vectors by ID (caller holds _lock)."""
        assert self._conn is not None
        for id_ in ids:
            row = self._conn.execute("SELECT row_id FROM vec_meta WHERE id = ?", (id_,)).fetchone()
            if row:
                self._conn.execute("DELETE FROM vec_index WHERE rowid = ?", (row[0],))
                self._conn.execute("DELETE FROM vec_meta WHERE id = ?", (id_,))
        self._conn.commit()

    def remove(self, ids: list[str]) -> None:
        """Remove vectors by ID."""
        with self._lock:
            self._ensure_connection()
            self._remove_locked(ids)

    def size(self) -> int:
        with self._lock:
            self._ensure_connection()
            assert self._conn is not None
            row = self._conn.execute("SELECT COUNT(*) FROM vec_meta").fetchone()
            return row[0] if row else 0

    def persist(self) -> None:
        """Commit any pending changes."""
        with self._lock:
            if self._conn:
                self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
