"""Tests for vector migration (migrate_vec).

Coverage:
- numpy → numpy (simplest case)
- sqlite → numpy (reads from sqlite, writes to memory)
- dry_run mode (reads but writes nothing)
- empty source (zero vectors)
- idempotent (run twice, same result)
"""

from __future__ import annotations

import pytest

from qortex.vec.index import NumpyVectorIndex
from qortex.vec.migrate import MigrateResult, migrate_vec

try:
    import sqlite_vec  # noqa: F401

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

skip_no_sqlite_vec = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec not installed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def numpy_source():
    """Pre-populated numpy vector index as migration source."""
    idx = NumpyVectorIndex(dimensions=3)
    return idx


@pytest.fixture
def numpy_dest():
    """Empty numpy vector index as migration destination."""
    return NumpyVectorIndex(dimensions=3)


# =============================================================================
# numpy → numpy
# =============================================================================


class TestNumpyToNumpy:
    async def test_basic_migration(self, numpy_source, numpy_dest):
        await numpy_source.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        result = await migrate_vec(numpy_source, numpy_dest, batch_size=2)

        assert isinstance(result, MigrateResult)
        assert result.vectors_read == 3
        assert result.vectors_written == 3
        assert result.batches == 2
        assert result.dry_run is False
        assert result.source_type == "NumpyVectorIndex"
        assert result.dest_type == "NumpyVectorIndex"
        assert await numpy_dest.size() == 3

    async def test_empty_source(self, numpy_source, numpy_dest):
        result = await migrate_vec(numpy_source, numpy_dest)
        assert result.vectors_read == 0
        assert result.vectors_written == 0
        assert result.batches == 0
        assert await numpy_dest.size() == 0

    async def test_dry_run(self, numpy_source, numpy_dest):
        await numpy_source.add(["a", "b"], [[1, 0, 0], [0, 1, 0]])
        result = await migrate_vec(numpy_source, numpy_dest, dry_run=True)

        assert result.vectors_read == 2
        assert result.vectors_written == 0
        assert result.dry_run is True
        assert await numpy_dest.size() == 0

    async def test_idempotent(self, numpy_source, numpy_dest):
        await numpy_source.add(["a", "b"], [[1, 0, 0], [0, 1, 0]])

        r1 = await migrate_vec(numpy_source, numpy_dest)
        r2 = await migrate_vec(numpy_source, numpy_dest)

        assert r1.vectors_written == r2.vectors_written
        assert await numpy_dest.size() == 2  # same count, upserted

    async def test_progress_callback(self, numpy_source, numpy_dest):
        await numpy_source.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        progress_calls = []

        def on_progress(batches, vectors):
            progress_calls.append((batches, vectors))

        await migrate_vec(numpy_source, numpy_dest, batch_size=2, on_progress=on_progress)

        assert len(progress_calls) == 2
        assert progress_calls[-1][1] == 3  # total vectors read

    async def test_search_after_migration(self, numpy_source, numpy_dest):
        """Migrated vectors should be searchable in destination."""
        await numpy_source.add(["a", "b"], [[1, 0, 0], [0, 1, 0]])
        await migrate_vec(numpy_source, numpy_dest)

        results = await numpy_dest.search([1, 0, 0], top_k=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)


# =============================================================================
# sqlite → numpy
# =============================================================================


@skip_no_sqlite_vec
class TestSqliteToNumpy:
    def _make_sqlite_index(self, tmp_path, dims=3):
        from qortex.vec.index import SqliteVecIndex

        return SqliteVecIndex(db_path=str(tmp_path / "migrate_source.db"), dimensions=dims)

    async def test_sqlite_to_numpy(self, tmp_path, numpy_dest):
        source = self._make_sqlite_index(tmp_path)
        await source.add(
            ["x", "y", "z"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

        result = await migrate_vec(source, numpy_dest, batch_size=2)

        assert result.vectors_read == 3
        assert result.vectors_written == 3
        assert result.source_type == "SqliteVecIndex"
        assert result.dest_type == "NumpyVectorIndex"
        assert await numpy_dest.size() == 3

    async def test_sqlite_dry_run(self, tmp_path, numpy_dest):
        source = self._make_sqlite_index(tmp_path)
        await source.add(["a"], [[1.0, 0.0, 0.0]])

        result = await migrate_vec(source, numpy_dest, dry_run=True)
        assert result.vectors_read == 1
        assert result.vectors_written == 0
        assert await numpy_dest.size() == 0


# =============================================================================
# service.migrate_vec()
# =============================================================================


class TestServiceMigrateVec:
    async def test_service_migrate_numpy_to_numpy(self):
        from qortex.service import QortexService
        from qortex.vec.index import NumpyVectorIndex

        # Create a service with numpy destination
        dest = NumpyVectorIndex(dimensions=3)
        service = QortexService(vector_index=dest)

        # Pre-populate a numpy source won't work via service since
        # _create_vec_index creates a fresh empty one. Test the basic
        # flow to ensure no crashes.
        result = await service.migrate_vec(source_type="numpy")
        assert result["vectors_read"] == 0
        assert result["vectors_written"] == 0

    async def test_service_migrate_invalid_type(self):
        from qortex.service import QortexService
        from qortex.vec.index import NumpyVectorIndex

        dest = NumpyVectorIndex(dimensions=3)
        service = QortexService(vector_index=dest)

        with pytest.raises(ValueError, match="Unknown vec backend type"):
            await service.migrate_vec(source_type="badtype")
