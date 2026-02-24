"""Tests for vector migration (migrate_vec).

Coverage:
- numpy → numpy (simplest case)
- sqlite → numpy (reads from sqlite, writes to memory)
- dry_run mode (reads but writes nothing)
- empty source (zero vectors)
- idempotent (run twice, same result)
- service.migrate_vec() with real data
- CLI smoke tests
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
    async def test_service_migrate_numpy_to_numpy_empty(self):
        from qortex.service import QortexService

        dest = NumpyVectorIndex(dimensions=3)
        service = QortexService(vector_index=dest)

        result = await service.migrate_vec(source_type="numpy")
        assert result["vectors_read"] == 0
        assert result["vectors_written"] == 0

    async def test_service_migrate_invalid_type(self):
        from qortex.service import QortexService

        dest = NumpyVectorIndex(dimensions=3)
        service = QortexService(vector_index=dest)

        with pytest.raises(ValueError, match="Unknown vec backend type"):
            await service.migrate_vec(source_type="badtype")

    async def test_service_migrate_with_real_data(self):
        """Integration: pre-populate source, migrate via service, verify dest has data."""
        from qortex.service import QortexService

        # Destination that the service owns
        dest = NumpyVectorIndex(dimensions=3)
        service = QortexService(vector_index=dest)

        # Pre-populate a source index, then patch _create_vec_index to return it
        source = NumpyVectorIndex(dimensions=3)
        await source.add(
            ["v1", "v2", "v3"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

        with patch.object(service, "_create_vec_index", return_value=source):
            result = await service.migrate_vec(source_type="numpy", batch_size=2)

        assert result["vectors_read"] == 3
        assert result["vectors_written"] == 3
        assert result["batches"] == 2
        assert result["dry_run"] is False
        assert await dest.size() == 3

        # Verify vectors are searchable in dest
        hits = await dest.search([1.0, 0.0, 0.0], top_k=1)
        assert hits[0][0] == "v1"

    async def test_service_migrate_dry_run_real_data(self):
        """Integration: dry_run reads source but writes nothing to dest."""
        from qortex.service import QortexService

        dest = NumpyVectorIndex(dimensions=3)
        service = QortexService(vector_index=dest)

        source = NumpyVectorIndex(dimensions=3)
        await source.add(["v1"], [[1.0, 0.0, 0.0]])

        with patch.object(service, "_create_vec_index", return_value=source):
            result = await service.migrate_vec(source_type="numpy", dry_run=True)

        assert result["vectors_read"] == 1
        assert result["vectors_written"] == 0
        assert result["dry_run"] is True
        assert await dest.size() == 0


# =============================================================================
# CLI: qortex migrate vec
# =============================================================================


class TestCliMigrateVec:
    def test_migrate_help(self):
        from typer.testing import CliRunner

        from qortex.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--help"])
        assert result.exit_code == 0
        assert "vec" in result.output

    def test_migrate_vec_help(self):
        import re

        from typer.testing import CliRunner

        from qortex.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "vec", "--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes — Rich/Typer inserts color codes that split
        # option names (e.g. "--from" becomes "-" + ANSI + "-from")
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--from" in plain
        assert "--batch-size" in plain
        assert "--dry-run" in plain

    def test_migrate_vec_requires_from(self):
        from typer.testing import CliRunner

        from qortex.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "vec"])
        assert result.exit_code != 0  # --from is required

    def test_migrate_vec_smoke(self):
        """Smoke test: mock service.migrate_vec, verify CLI output."""
        from typer.testing import CliRunner

        from qortex.cli import app

        mock_result = {
            "vectors_read": 5,
            "vectors_written": 5,
            "batches": 1,
            "duration_seconds": 0.1,
            "dry_run": False,
            "source_type": "NumpyVectorIndex",
            "dest_type": "NumpyVectorIndex",
        }

        mock_svc = AsyncMock()
        mock_svc.migrate_vec = AsyncMock(return_value=mock_result)
        mock_svc.vector_index = NumpyVectorIndex(dimensions=3)

        with patch("qortex.cli.migrate._make_service", return_value=mock_svc):
            runner = CliRunner()
            result = runner.invoke(app, ["migrate", "vec", "--from", "numpy"])

        assert result.exit_code == 0
        assert "5 read" in result.output
        assert "5 written" in result.output

    def test_migrate_vec_dry_run_flag(self):
        """Verify --dry-run flag is passed through."""
        from typer.testing import CliRunner

        from qortex.cli import app

        mock_result = {
            "vectors_read": 3,
            "vectors_written": 0,
            "batches": 1,
            "duration_seconds": 0.05,
            "dry_run": True,
            "source_type": "NumpyVectorIndex",
            "dest_type": "NumpyVectorIndex",
        }

        mock_svc = AsyncMock()
        mock_svc.migrate_vec = AsyncMock(return_value=mock_result)
        mock_svc.vector_index = NumpyVectorIndex(dimensions=3)

        with patch("qortex.cli.migrate._make_service", return_value=mock_svc):
            runner = CliRunner()
            result = runner.invoke(app, ["migrate", "vec", "--from", "numpy", "--dry-run"])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        mock_svc.migrate_vec.assert_called_once_with(
            source_type="numpy",
            batch_size=500,
            dry_run=True,
        )
