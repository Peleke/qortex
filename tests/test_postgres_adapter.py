"""Tests for PostgresSourceAdapter, PostgresIngestor, IngestConfig, and MCP source tools.

All tests are mock-based (mock asyncpg) — no real database needed.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qortex.sources.base import (
    ColumnSchema,
    IngestConfig,
    SourceConfig,
    SyncResult,
    TableSchema,
)
from qortex.sources.postgres import IngestResult, PostgresIngestor, PostgresSourceAdapter


# ===========================================================================
# Helpers
# ===========================================================================


def _make_col_row(
    schema: str, table: str, col: str, dtype: str, nullable: str = "YES"
) -> dict:
    """Build a mock information_schema.columns row."""
    return {
        "table_schema": schema,
        "table_name": table,
        "column_name": col,
        "data_type": dtype,
        "is_nullable": nullable,
        "column_default": None,
    }


def _make_pk_row(schema: str, table: str, col: str) -> dict:
    return {"table_schema": schema, "table_name": table, "column_name": col}


def _make_fk_row(
    schema: str, table: str, col: str, ftable: str, fcol: str
) -> dict:
    return {
        "table_schema": schema,
        "table_name": table,
        "column_name": col,
        "foreign_table": ftable,
        "foreign_column": fcol,
    }


class FakeEmbedding:
    """Deterministic fake embedding model for testing."""

    def __init__(self, dims: int = 8) -> None:
        self.dimensions = dims
        self._calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._calls.append(texts)
        return [[float(i) / max(len(t), 1) for i in range(self.dimensions)] for t in texts]


class FakeVectorIndex:
    """In-memory fake vector index for testing."""

    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        for vid, emb in zip(ids, embeddings):
            self._store[vid] = emb

    def search(self, query: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        results = [(vid, 0.9) for vid in list(self._store.keys())[:top_k]]
        return results

    def size(self) -> int:
        return len(self._store)

    def remove(self, ids: list[str]) -> None:
        for vid in ids:
            self._store.pop(vid, None)


def _mock_asyncpg_connection(
    columns: list[dict] | None = None,
    pk_rows: list[dict] | None = None,
    fk_rows: list[dict] | None = None,
    table_rows: dict[str, list[dict]] | None = None,
    row_counts: dict[str, int] | None = None,
) -> AsyncMock:
    """Build a mock asyncpg connection with configurable responses."""
    conn = AsyncMock()

    columns = columns or []
    pk_rows = pk_rows or []
    fk_rows = fk_rows or []
    table_rows = table_rows or {}
    row_counts = row_counts or {}

    async def mock_fetch(query: str, *args: Any) -> list[dict]:
        query_lower = query.strip().lower()
        if "information_schema.columns" in query_lower:
            return columns
        if "primary key" in query_lower:
            return pk_rows
        if "foreign key" in query_lower:
            return fk_rows
        if "select *" in query_lower:
            for tname, rows in table_rows.items():
                if tname.lower() in query_lower:
                    batch_size = args[0] if args else 500
                    offset = args[1] if len(args) > 1 else 0
                    return rows[offset : offset + batch_size]
            return []
        return []

    async def mock_fetchval(query: str, *args: Any) -> int:
        for tname, count in row_counts.items():
            if tname.lower() in query.lower():
                return count
        return 0

    conn.fetch = mock_fetch
    conn.fetchval = mock_fetchval
    conn.close = AsyncMock()

    return conn


def _inject_connection(adapter: PostgresSourceAdapter, config: SourceConfig, conn: Any) -> None:
    """Inject a mock connection into an adapter (bypasses asyncpg.connect)."""
    adapter._config = config
    adapter._conn = conn


# ===========================================================================
# TestPostgresConnect
# ===========================================================================


class TestPostgresConnect:
    async def test_connect_success(self):
        adapter = PostgresSourceAdapter()
        config = SourceConfig(
            source_id="test",
            connection_string="postgresql://localhost/testdb",
        )
        mock_conn = AsyncMock()
        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            await adapter.connect(config)

        assert adapter._conn is mock_conn
        assert adapter._config is config

    async def test_connect_failure(self):
        adapter = PostgresSourceAdapter()
        config = SourceConfig(
            source_id="test",
            connection_string="postgresql://bad_host/db",
        )
        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(side_effect=OSError("Connection refused"))

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            with pytest.raises(OSError, match="Connection refused"):
                await adapter.connect(config)

    async def test_disconnect_idempotent(self):
        adapter = PostgresSourceAdapter()
        await adapter.disconnect()
        assert adapter._conn is None

        config = SourceConfig(source_id="test", connection_string="postgresql://localhost/db")
        mock_conn = AsyncMock()
        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            await adapter.connect(config)
            await adapter.disconnect()
            mock_conn.close.assert_called_once()
            assert adapter._conn is None

        await adapter.disconnect()


# ===========================================================================
# TestPostgresDiscover
# ===========================================================================


class TestPostgresDiscover:
    async def test_basic_schema_discovery(self):
        columns = [
            _make_col_row("public", "users", "id", "uuid", "NO"),
            _make_col_row("public", "users", "name", "text"),
            _make_col_row("public", "users", "email", "text"),
        ]
        pk_rows = [_make_pk_row("public", "users", "id")]

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://")
        conn = _mock_asyncpg_connection(
            columns=columns, pk_rows=pk_rows, row_counts={"users": 100}
        )
        _inject_connection(adapter, config, conn)

        schemas = await adapter.discover()
        assert len(schemas) == 1
        assert schemas[0].name == "users"
        assert len(schemas[0].columns) == 3
        assert schemas[0].pk_columns == ["id"]
        assert schemas[0].row_count == 100

    async def test_fk_detection(self):
        columns = [
            _make_col_row("public", "users", "id", "uuid", "NO"),
            _make_col_row("public", "posts", "id", "uuid", "NO"),
            _make_col_row("public", "posts", "user_id", "uuid"),
            _make_col_row("public", "posts", "title", "text"),
        ]
        pk_rows = [
            _make_pk_row("public", "users", "id"),
            _make_pk_row("public", "posts", "id"),
        ]
        fk_rows = [
            _make_fk_row("public", "posts", "user_id", "users", "id"),
        ]

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://")
        conn = _mock_asyncpg_connection(
            columns=columns, pk_rows=pk_rows, fk_rows=fk_rows
        )
        _inject_connection(adapter, config, conn)

        schemas = await adapter.discover()
        posts = next(t for t in schemas if t.name == "posts")
        fk_col = next(c for c in posts.columns if c.name == "user_id")
        assert fk_col.is_fk is True
        assert fk_col.foreign_table == "users"
        assert fk_col.foreign_column == "id"

    async def test_exclude_tables(self):
        columns = [
            _make_col_row("public", "users", "id", "uuid"),
            _make_col_row("public", "_prisma_migrations", "id", "integer"),
        ]

        adapter = PostgresSourceAdapter()
        config = SourceConfig(
            source_id="test",
            connection_string="mock://",
            exclude_tables=["_prisma_migrations"],
        )
        conn = _mock_asyncpg_connection(columns=columns)
        _inject_connection(adapter, config, conn)

        schemas = await adapter.discover()
        table_names = {t.name for t in schemas}
        assert "users" in table_names
        assert "_prisma_migrations" not in table_names

    async def test_multi_schema(self):
        columns = [
            _make_col_row("public", "users", "id", "uuid"),
            _make_col_row("auth", "roles", "id", "integer"),
            _make_col_row("auth", "roles", "name", "text"),
        ]

        adapter = PostgresSourceAdapter()
        config = SourceConfig(
            source_id="test",
            connection_string="mock://",
            schemas=["public", "auth"],
        )
        conn = _mock_asyncpg_connection(columns=columns)
        _inject_connection(adapter, config, conn)

        schemas = await adapter.discover()
        assert len(schemas) == 2
        names = {t.name for t in schemas}
        assert "users" in names
        assert "roles" in names
        roles = next(t for t in schemas if t.name == "roles")
        assert roles.schema_name == "auth"

    async def test_discover_not_connected_raises(self):
        adapter = PostgresSourceAdapter()
        with pytest.raises(RuntimeError, match="Not connected"):
            await adapter.discover()


# ===========================================================================
# TestPostgresReadRows
# ===========================================================================


class TestPostgresReadRows:
    async def test_yields_dicts(self):
        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://")
        conn = _mock_asyncpg_connection(
            table_rows={
                "food_items": [
                    {"id": 1, "name": "Chicken", "calories": 165},
                    {"id": 2, "name": "Rice", "calories": 130},
                ]
            }
        )
        _inject_connection(adapter, config, conn)

        rows = []
        async for row in adapter.read_rows("food_items"):
            rows.append(row)
        assert len(rows) == 2
        assert rows[0]["name"] == "Chicken"

    async def test_batching(self):
        """With batch_size=2, reads 3 rows in 2 fetches."""
        all_rows = [{"id": i, "name": f"item_{i}"} for i in range(3)]

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://")
        conn = _mock_asyncpg_connection(table_rows={"items": all_rows})
        _inject_connection(adapter, config, conn)

        rows = []
        async for row in adapter.read_rows("items", batch_size=2):
            rows.append(row)
        assert len(rows) == 3

    async def test_empty_table(self):
        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://")
        conn = _mock_asyncpg_connection(table_rows={"empty": []})
        _inject_connection(adapter, config, conn)

        rows = []
        async for row in adapter.read_rows("empty"):
            rows.append(row)
        assert len(rows) == 0


# ===========================================================================
# TestPostgresSync
# ===========================================================================


class TestPostgresSync:
    async def test_creates_vectors(self):
        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="mm", connection_string="mock://")
        adapter._schemas = [
            TableSchema(
                name="food_items",
                columns=[
                    ColumnSchema(name="id", data_type="integer", is_pk=True),
                    ColumnSchema(name="name", data_type="text"),
                    ColumnSchema(name="calories", data_type="integer"),
                ],
            )
        ]
        conn = _mock_asyncpg_connection(
            table_rows={
                "food_items": [
                    {"id": 1, "name": "Chicken", "calories": 165},
                    {"id": 2, "name": "Rice", "calories": 130},
                ]
            }
        )
        _inject_connection(adapter, config, conn)

        result = await adapter.sync(vector_index=vec_index, embedding_model=embedding)
        assert result.tables_synced == 1
        assert result.rows_added == 2
        assert result.vectors_created == 2
        assert vec_index.size() == 2

    async def test_respects_domain_map(self):
        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()

        adapter = PostgresSourceAdapter()
        config = SourceConfig(
            source_id="mm",
            connection_string="mock://",
            domain_map={"food_*": "nutrition", "habit_*": "habits"},
        )
        adapter._schemas = [
            TableSchema(
                name="food_items",
                columns=[
                    ColumnSchema(name="id", data_type="integer", is_pk=True),
                    ColumnSchema(name="name", data_type="text"),
                ],
            )
        ]
        conn = _mock_asyncpg_connection(
            table_rows={"food_items": [{"id": 1, "name": "Chicken"}]}
        )
        _inject_connection(adapter, config, conn)

        result = await adapter.sync(vector_index=vec_index, embedding_model=embedding)
        assert result.rows_added == 1
        assert config.resolve_domain("food_items") == "nutrition"

    async def test_serializes_rows(self):
        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://")
        adapter._schemas = [
            TableSchema(
                name="movements",
                columns=[
                    ColumnSchema(name="id", data_type="integer", is_pk=True),
                    ColumnSchema(name="name", data_type="text"),
                    ColumnSchema(name="difficulty", data_type="text"),
                ],
            )
        ]
        conn = _mock_asyncpg_connection(
            table_rows={
                "movements": [
                    {"id": 1, "name": "Deadlift", "difficulty": "advanced"},
                ]
            }
        )
        _inject_connection(adapter, config, conn)

        result = await adapter.sync(vector_index=vec_index, embedding_model=embedding)
        assert result.vectors_created == 1
        assert len(embedding._calls) == 1
        assert "Deadlift" in embedding._calls[0][0]

    async def test_vec_id_format(self):
        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="mm_main", connection_string="mock://")
        adapter._schemas = [
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(name="id", data_type="uuid", is_pk=True),
                    ColumnSchema(name="name", data_type="text"),
                ],
            )
        ]
        conn = _mock_asyncpg_connection(
            table_rows={"users": [{"id": "abc-123", "name": "Alice"}]}
        )
        _inject_connection(adapter, config, conn)

        await adapter.sync(vector_index=vec_index, embedding_model=embedding)
        assert "mm_main:users:abc-123" in vec_index._store

    async def test_incremental_mode(self):
        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://")
        adapter._schemas = [
            TableSchema(
                name="items",
                columns=[
                    ColumnSchema(name="id", data_type="integer", is_pk=True),
                    ColumnSchema(name="name", data_type="text"),
                ],
            )
        ]
        conn = _mock_asyncpg_connection(
            table_rows={"items": [{"id": 1, "name": "A"}]}
        )
        _inject_connection(adapter, config, conn)

        result = await adapter.sync(
            tables=["items"], mode="incremental",
            vector_index=vec_index, embedding_model=embedding,
        )
        assert result.rows_added == 1

    async def test_batch_size_respected(self):
        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()

        adapter = PostgresSourceAdapter()
        config = SourceConfig(source_id="test", connection_string="mock://", batch_size=2)
        adapter._schemas = [
            TableSchema(
                name="items",
                columns=[
                    ColumnSchema(name="id", data_type="integer", is_pk=True),
                    ColumnSchema(name="val", data_type="text"),
                ],
            )
        ]
        all_rows = [{"id": i, "val": f"v{i}"} for i in range(5)]
        conn = _mock_asyncpg_connection(table_rows={"items": all_rows})
        _inject_connection(adapter, config, conn)

        result = await adapter.sync(vector_index=vec_index, embedding_model=embedding)
        assert result.rows_added == 5
        assert result.vectors_created == 5


# ===========================================================================
# TestIngestConfig
# ===========================================================================


class TestIngestConfig:
    def test_defaults(self):
        ic = IngestConfig()
        assert ic.targets == "both"
        assert ic.mode == "batch"
        assert ic.batch_size == 500
        assert ic.serialize_strategy == "natural_language"
        assert ic.embed_catalog_tables is True
        assert ic.extract_rules is True
        assert ic.user_filter is None

    def test_env_overrides(self):
        with patch.dict(os.environ, {
            "QORTEX_INGEST_TARGETS": "vec",
            "QORTEX_INGEST_MODE": "online",
            "QORTEX_INGEST_BATCH_SIZE": "100",
        }):
            ic = IngestConfig()
            assert ic.targets == "vec"
            assert ic.mode == "online"
            assert ic.batch_size == 100

    def test_invalid_targets_raises(self):
        with pytest.raises(ValueError, match="Invalid targets"):
            IngestConfig(targets="invalid")  # type: ignore

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            IngestConfig(mode="streaming")  # type: ignore

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Invalid serialize_strategy"):
            IngestConfig(serialize_strategy="xml")  # type: ignore

    def test_from_url_factory(self):
        ingestor = PostgresIngestor.from_url(
            "postgresql://localhost:5435/swae_movements",
            source_id="mm_movements",
            domain_map={"*": "exercise"},
            ingest=IngestConfig(targets="vec"),
        )
        assert ingestor._config.source_id == "mm_movements"
        assert ingestor._config.connection_string == "postgresql://localhost:5435/swae_movements"
        assert ingestor._ingest.targets == "vec"


# ===========================================================================
# TestPostgresIngestor
# ===========================================================================


class TestPostgresIngestor:
    async def test_run_vec_only(self):
        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()

        ingestor = PostgresIngestor.from_url(
            "postgresql://localhost/testdb",
            source_id="test",
            domain_map={"*": "test"},
            ingest=IngestConfig(targets="vec"),
            vector_index=vec_index,
            embedding_model=embedding,
        )

        mock_conn = _mock_asyncpg_connection(
            columns=[
                _make_col_row("public", "items", "id", "integer", "NO"),
                _make_col_row("public", "items", "name", "text"),
            ],
            pk_rows=[_make_pk_row("public", "items", "id")],
            table_rows={"items": [{"id": 1, "name": "Test"}]},
            row_counts={"items": 1},
        )

        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            result = await ingestor.run()

        assert result.vec_result is not None
        assert result.vec_result.rows_added == 1
        assert result.graph_result is None

    async def test_run_graph_only_skips_vec(self):
        """targets='graph' skips vec sync, runs graph ingest."""
        from qortex.core.memory import InMemoryBackend

        backend = InMemoryBackend()
        backend.connect()

        ingestor = PostgresIngestor.from_url(
            "postgresql://localhost/testdb",
            source_id="test",
            ingest=IngestConfig(targets="graph"),
            backend=backend,
        )

        mock_conn = _mock_asyncpg_connection(
            columns=[_make_col_row("public", "items", "id", "integer")],
            pk_rows=[_make_pk_row("public", "items", "id")],
            row_counts={"items": 0},
        )

        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            result = await ingestor.run()

        # No vec sync happened
        assert result.vec_result is None

    async def test_run_both_targets(self):
        """targets='both' runs vec sync + graph ingest."""
        from qortex.core.memory import InMemoryBackend

        vec_index = FakeVectorIndex()
        embedding = FakeEmbedding()
        backend = InMemoryBackend()
        backend.connect()

        ingestor = PostgresIngestor.from_url(
            "postgresql://localhost/testdb",
            source_id="test",
            ingest=IngestConfig(targets="both"),
            vector_index=vec_index,
            embedding_model=embedding,
            backend=backend,
        )

        mock_conn = _mock_asyncpg_connection(
            columns=[
                _make_col_row("public", "items", "id", "integer", "NO"),
                _make_col_row("public", "items", "name", "text"),
            ],
            pk_rows=[_make_pk_row("public", "items", "id")],
            table_rows={"items": [{"id": 1, "name": "Test"}]},
            row_counts={"items": 1},
        )

        mock_asyncpg = MagicMock()
        mock_asyncpg.connect = AsyncMock(return_value=mock_conn)

        with patch.dict(sys.modules, {"asyncpg": mock_asyncpg}):
            result = await ingestor.run()

        # Vec sync ran
        assert result.vec_result is not None
        assert result.vec_result.vectors_created >= 1


# ===========================================================================
# TestMCPSourceTools
# ===========================================================================


class TestMCPSourceTools:
    def test_source_discover_not_found(self):
        from qortex.mcp.server import _source_discover_impl

        result = _source_discover_impl("nonexistent")
        assert "error" in result
        assert "not found" in result["error"]

    def test_source_list_empty(self):
        from qortex.mcp.server import _source_list_impl, _source_registry

        _source_registry.clear()
        result = _source_list_impl()
        assert result["sources"] == []

    def test_source_list_with_entries(self):
        from qortex.mcp.server import _source_list_impl, _source_registry

        _source_registry.clear()
        config = SourceConfig(source_id="mm", connection_string="mock://")
        _source_registry.register(config, object())
        _source_registry.cache_schemas(
            "mm", [TableSchema(name="users"), TableSchema(name="items")]
        )

        result = _source_list_impl()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source_id"] == "mm"
        assert result["sources"][0]["tables"] == 2

        _source_registry.clear()

    def test_source_disconnect_not_found(self):
        from qortex.mcp.server import _source_disconnect_impl

        result = _source_disconnect_impl("nonexistent")
        assert "error" in result

    def test_source_sync_not_found(self):
        from qortex.mcp.server import _source_sync_impl

        result = _source_sync_impl("nonexistent")
        assert "error" in result

    def test_source_discover_with_cached(self):
        from qortex.mcp.server import _source_discover_impl, _source_registry

        _source_registry.clear()
        config = SourceConfig(source_id="test_db", connection_string="mock://")
        _source_registry.register(config, object())
        _source_registry.cache_schemas(
            "test_db",
            [
                TableSchema(
                    name="users",
                    columns=[
                        ColumnSchema(name="id", data_type="uuid", is_pk=True),
                        ColumnSchema(name="name", data_type="text"),
                    ],
                    row_count=50,
                ),
            ],
        )

        result = _source_discover_impl("test_db")
        assert "error" not in result
        assert len(result["tables"]) == 1
        assert result["tables"][0]["name"] == "users"
        assert result["tables"][0]["columns"] == 2
        assert result["tables"][0]["row_count"] == 50
        assert result["tables"][0]["pk_columns"] == ["id"]

        _source_registry.clear()
