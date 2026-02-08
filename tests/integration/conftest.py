"""Shared fixtures for integration tests.

Detects Docker availability by socket-probing all 5 PostgreSQL ports.
Provides SourceConfig fixtures for MindMirror (4 DBs) and interlinear (1 DB).
"""

from __future__ import annotations

import socket
import time
from typing import Any

import numpy as np
import pytest

from qortex.core.memory import InMemoryBackend
from qortex.sources.base import IngestConfig, SourceConfig

# ---------------------------------------------------------------------------
# Port mapping: mirrors docker-compose.yml (15xxx to avoid prod conflicts)
# ---------------------------------------------------------------------------

PORTS = {
    "mm_main": 15432,
    "mm_movements": 15435,
    "mm_practices": 15436,
    "mm_users": 15437,
    "interlinear": 15433,
}

DB_NAMES = {
    "mm_main": "mindmirror",
    "mm_movements": "swae_movements",
    "mm_practices": "swae_practices",
    "mm_users": "swae_users",
    "interlinear": "interlinear",
}

# Docker credentials (must match docker-compose.yml)
PG_USER = "qortex_test"
PG_PASSWORD = "qortex_test"


def _pg_available(port: int, host: str = "localhost", timeout: float = 1.0) -> bool:
    """Check if a PostgreSQL port is reachable via TCP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except (OSError, ConnectionRefusedError):
        return False


def _pg_ready(port: int, db: str, retries: int = 3, delay: float = 1.0) -> bool:
    """Check if PostgreSQL is ready to accept queries (not just TCP open)."""
    if not _pg_available(port):
        return False
    try:
        import asyncpg
        import asyncio

        async def _check():
            conn = await asyncpg.connect(
                host="localhost", port=port, user=PG_USER, password=PG_PASSWORD,
                database=db, timeout=5,
            )
            # Verify init scripts have run by checking for tables
            count = await conn.fetchval(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
            )
            await conn.close()
            return count > 0

        for i in range(retries):
            try:
                if asyncio.run(_check()):
                    return True
            except Exception:
                if i < retries - 1:
                    time.sleep(delay)
        return False
    except ImportError:
        # asyncpg not installed — fall back to socket check
        return _pg_available(port)


DOCKER_AVAILABLE = all(
    _pg_ready(PORTS[svc], DB_NAMES[svc]) for svc in PORTS
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not DOCKER_AVAILABLE,
        reason=(
            "Docker Compose services not running. Start with:\n"
            "  docker compose -f tests/integration/docker-compose.yml up -d"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Fake embedding model for integration tests
# ---------------------------------------------------------------------------


class FakeEmbedding:
    """Deterministic fake embedding model for integration tests.

    Uses a hash-based approach so the same text always produces the same
    embedding, enabling meaningful similarity comparisons.
    """

    dimensions = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic embeddings from text hashes."""
        results = []
        for text in texts:
            # Use hash to get deterministic seed per text
            seed = abs(hash(text)) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.random(self.dimensions).tolist()
            # Normalize to unit vector
            norm = sum(v * v for v in vec) ** 0.5
            vec = [v / norm for v in vec]
            results.append(vec)
        return results


class FakeVectorIndex:
    """In-memory vector index with actual cosine similarity for integration tests."""

    def __init__(self):
        self.vectors: dict[str, np.ndarray] = {}

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        for vid, emb in zip(ids, embeddings):
            self.vectors[vid] = np.array(emb)

    def search(
        self, embedding: list[float], top_k: int = 10, **kwargs
    ) -> list[tuple[str, float]]:
        """Actual cosine similarity search (matches VectorIndex protocol)."""
        if not self.vectors:
            return []
        q = np.array(embedding)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        scores = []
        for vid, vec in self.vectors.items():
            v_norm = vec / (np.linalg.norm(vec) + 1e-10)
            sim = float(np.dot(q_norm, v_norm))
            scores.append((vid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # Alias for backward compat with tests that use query()
    query = search

    def size(self) -> int:
        return len(self.vectors)

    def remove(self, ids: list[str]) -> None:
        for vid in ids:
            self.vectors.pop(vid, None)

    def __len__(self) -> int:
        return len(self.vectors)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _conn_string(service: str) -> str:
    """Build connection string for a Docker service."""
    port = PORTS[service]
    db = DB_NAMES[service]
    return f"postgresql://{PG_USER}:{PG_PASSWORD}@localhost:{port}/{db}"


@pytest.fixture
def mm_main_config() -> SourceConfig:
    """SourceConfig for MindMirror main (journals, habits, meals)."""
    return SourceConfig(
        source_id="mm_main",
        connection_string=_conn_string("mm_main"),
        domain_map={
            "habit_*": "habits",
            "journal_*": "reflection",
            "food_*": "nutrition",
            "meal*": "nutrition",
        },
        default_domain="mindmirror",
    )


@pytest.fixture
def mm_movements_config() -> SourceConfig:
    """SourceConfig for MindMirror movements (exercise catalog)."""
    return SourceConfig(
        source_id="mm_movements",
        connection_string=_conn_string("mm_movements"),
        domain_map={"*": "exercise"},
    )


@pytest.fixture
def mm_practices_config() -> SourceConfig:
    """SourceConfig for MindMirror practices (workouts)."""
    return SourceConfig(
        source_id="mm_practices",
        connection_string=_conn_string("mm_practices"),
        domain_map={
            "practice_*": "training",
            "prescription_*": "training",
            "movement_*": "training",
        },
        default_domain="training",
    )


@pytest.fixture
def mm_users_config() -> SourceConfig:
    """SourceConfig for MindMirror users."""
    return SourceConfig(
        source_id="mm_users",
        connection_string=_conn_string("mm_users"),
        domain_map={"*": "identity"},
    )


@pytest.fixture
def interlinear_config() -> SourceConfig:
    """SourceConfig for interlinear (Supabase-like single DB)."""
    return SourceConfig(
        source_id="interlinear",
        connection_string=_conn_string("interlinear"),
        domain_map={
            "courses": "language",
            "lessons": "language",
            "vocabulary": "language",
            "lesson_vocabulary_*": "language",
            "grammar_*": "language",
            "exercises": "language",
            "ai_generation_*": "metadata",
        },
    )


@pytest.fixture
def mindmirror_configs(
    mm_main_config, mm_movements_config, mm_practices_config, mm_users_config
) -> dict[str, SourceConfig]:
    """All 4 MindMirror database configs."""
    return {
        "mm_main": mm_main_config,
        "mm_movements": mm_movements_config,
        "mm_practices": mm_practices_config,
        "mm_users": mm_users_config,
    }


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def fake_vec_index() -> FakeVectorIndex:
    return FakeVectorIndex()


@pytest.fixture
def backend() -> InMemoryBackend:
    b = InMemoryBackend()
    b.connect()
    return b
