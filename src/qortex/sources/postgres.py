"""PostgresSourceAdapter: async adapter for PostgreSQL databases.

Uses asyncpg for async I/O. Discovers schemas via information_schema,
serializes rows to text, embeds and upserts to VectorIndex.

PostgresIngestor: unified ergonomic wrapper that composes vec + graph layers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from qortex.sources.base import (
    ColumnSchema,
    IngestConfig,
    SourceConfig,
    SyncResult,
    TableSchema,
)
from qortex.sources.serializer import (
    KeyValueSerializer,
    NaturalLanguageSerializer,
    RowSerializer,
)

logger = logging.getLogger(__name__)


def _quote_ident(name: str) -> str:
    """Safely quote a SQL identifier, escaping embedded double quotes."""
    return '"' + name.replace('"', '""') + '"'


class PostgresSourceAdapter:
    """Async source adapter for PostgreSQL via asyncpg.

    Implements the SourceAdapter protocol with real database I/O.
    """

    def __init__(self) -> None:
        self._conn: Any = None
        self._config: SourceConfig | None = None
        self._schemas: list[TableSchema] = []

    async def connect(self, config: SourceConfig) -> None:
        """Connect to a PostgreSQL database."""
        import asyncpg

        self._config = config
        self._conn = await asyncpg.connect(config.connection_string)

    @property
    def connection(self) -> Any:
        """The underlying asyncpg connection, or None if not connected."""
        return self._conn

    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def discover(self) -> list[TableSchema]:
        """Discover table schemas from information_schema."""
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")
        if self._config is None:
            raise RuntimeError("No config set.")

        schemas_param = self._config.schemas or ["public"]

        # Query columns
        rows = await self._conn.fetch(
            """
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default
            FROM information_schema.columns c
            WHERE c.table_schema = ANY($1::text[])
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
            """,
            schemas_param,
        )

        # Query primary keys
        pk_rows = await self._conn.fetch(
            """
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = ANY($1::text[])
            """,
            schemas_param,
        )
        pk_set: set[tuple[str, str, str]] = {
            (r["table_schema"], r["table_name"], r["column_name"]) for r in pk_rows
        }

        # Query foreign keys
        fk_rows = await self._conn.fetch(
            """
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = ANY($1::text[])
            """,
            schemas_param,
        )
        fk_map: dict[tuple[str, str, str], tuple[str, str]] = {
            (r["table_schema"], r["table_name"], r["column_name"]): (
                r["foreign_table"],
                r["foreign_column"],
            )
            for r in fk_rows
        }

        # Build TableSchema objects
        tables: dict[str, TableSchema] = {}
        for row in rows:
            schema_name = row["table_schema"]
            table_name = row["table_name"]
            full = f"{schema_name}.{table_name}"

            # Skip excluded tables
            if table_name in (self._config.exclude_tables or []):
                continue

            if full not in tables:
                tables[full] = TableSchema(
                    name=table_name,
                    schema_name=schema_name,
                )

            col_name = row["column_name"]
            is_pk = (schema_name, table_name, col_name) in pk_set
            fk_info = fk_map.get((schema_name, table_name, col_name))

            tables[full].columns.append(
                ColumnSchema(
                    name=col_name,
                    data_type=row["data_type"],
                    nullable=row["is_nullable"] == "YES",
                    is_pk=is_pk,
                    is_fk=fk_info is not None,
                    foreign_table=fk_info[0] if fk_info else None,
                    foreign_column=fk_info[1] if fk_info else None,
                )
            )

        # Get row counts
        for table in tables.values():
            try:
                count_row = await self._conn.fetchval(
                    f"SELECT COUNT(*) FROM {_quote_ident(table.schema_name)}.{_quote_ident(table.name)}"
                )
                table.row_count = count_row or 0
            except Exception:
                table.row_count = 0

        self._schemas = list(tables.values())
        return self._schemas

    async def read_rows(
        self,
        table: str,
        batch_size: int = 500,
        offset: int = 0,
        schema_name: str = "public",
    ) -> AsyncIterator[dict[str, Any]]:
        """Read rows from a table, paginating through batches."""
        if self._conn is None:
            raise RuntimeError("Not connected.")

        current_offset = offset
        while True:
            rows = await self._conn.fetch(
                f"SELECT * FROM {_quote_ident(schema_name)}.{_quote_ident(table)} LIMIT $1 OFFSET $2",
                batch_size,
                current_offset,
            )
            if not rows:
                break
            for row in rows:
                yield dict(row)
            if len(rows) < batch_size:
                break
            current_offset += batch_size

    async def sync(
        self,
        tables: list[str] | None = None,
        mode: str = "full",
        vector_index: Any = None,
        embedding_model: Any = None,
        serializer: RowSerializer | None = None,
    ) -> SyncResult:
        """Sync source data into qortex's vector layer.

        Args:
            tables: Tables to sync. None = all discovered tables.
            mode: "full" or "incremental".
            vector_index: VectorIndex to write to.
            embedding_model: EmbeddingModel for generating embeddings.
            serializer: Row serializer. Defaults to NaturalLanguageSerializer.
        """
        if self._conn is None or self._config is None:
            raise RuntimeError("Not connected.")

        if not self._schemas:
            await self.discover()

        if serializer is None:
            if self._config.serialize_strategy == "key_value":
                serializer = KeyValueSerializer()
            else:
                serializer = NaturalLanguageSerializer()

        start = time.monotonic()
        result = SyncResult(source_id=self._config.source_id)

        target_tables = self._schemas
        if tables is not None:
            target_tables = [t for t in self._schemas if t.name in tables]

        batch_size = self._config.batch_size

        for table in target_tables:
            try:
                offset = 0
                table_ids: list[str] = []
                table_texts: list[str] = []
                table_domains: list[str] = []

                pk_cols = table.pk_columns
                if not pk_cols:
                    pk_cols = ["id"]  # fallback

                domain = self._config.resolve_domain(table.name)

                while True:
                    rows = await self._conn.fetch(
                        f"SELECT * FROM {_quote_ident(table.schema_name)}.{_quote_ident(table.name)} "
                        f"LIMIT $1 OFFSET $2",
                        batch_size,
                        offset,
                    )
                    if not rows:
                        break

                    for row in rows:
                        row_dict = dict(row)
                        # Build deterministic vector ID
                        pk_val = ":".join(str(row_dict.get(pk, "")) for pk in pk_cols)
                        vec_id = f"{self._config.source_id}:{table.name}:{pk_val}"

                        text = serializer.serialize(table.name, row_dict, table)
                        if text.strip():
                            table_ids.append(vec_id)
                            table_texts.append(text)
                            table_domains.append(domain)
                            result.rows_added += 1

                    offset += batch_size

                # Embed and upsert
                if table_texts and embedding_model is not None and vector_index is not None:
                    embeddings = embedding_model.embed(table_texts)
                    vector_index.add(table_ids, embeddings)
                    result.vectors_created += len(table_ids)

                result.tables_synced += 1

            except Exception as e:
                result.errors.append(f"{table.name}: {e}")

        result.duration_seconds = round(time.monotonic() - start, 2)
        return result


@dataclass
class IngestResult:
    """Result of a unified ingest operation."""

    source_id: str
    vec_result: SyncResult | None = None
    graph_result: dict[str, int] | None = None
    tables_discovered: int = 0
    duration_seconds: float = 0.0


class PostgresIngestor:
    """Unified ergonomic wrapper for database ingestion.

    Composes PostgresSourceAdapter (vec) + PostgresGraphIngestor (graph)
    into a single run() call.
    """

    def __init__(
        self,
        config: SourceConfig,
        ingest: IngestConfig | None = None,
        vector_index: Any = None,
        embedding_model: Any = None,
        backend: Any = None,
    ) -> None:
        self._config = config
        self._ingest = ingest or IngestConfig()
        self._vector_index = vector_index
        self._embedding_model = embedding_model
        self._backend = backend
        self._adapter = PostgresSourceAdapter()

    @classmethod
    def from_url(
        cls,
        url: str,
        source_id: str,
        domain_map: dict[str, str] | None = None,
        ingest: IngestConfig | None = None,
        vector_index: Any = None,
        embedding_model: Any = None,
        backend: Any = None,
    ) -> PostgresIngestor:
        """Create a PostgresIngestor from a connection URL."""
        config = SourceConfig(
            source_id=source_id,
            connection_string=url,
            domain_map=domain_map or {},
        )
        return cls(
            config=config,
            ingest=ingest,
            vector_index=vector_index,
            embedding_model=embedding_model,
            backend=backend,
        )

    async def run(self) -> IngestResult:
        """Run the full ingest pipeline.

        connect -> discover -> sync vec -> ingest graph -> disconnect.
        """
        start = time.monotonic()
        result = IngestResult(source_id=self._config.source_id)

        try:
            await self._adapter.connect(self._config)
            schemas = await self._adapter.discover()
            result.tables_discovered = len(schemas)

            targets = self._ingest.targets

            # Vec sync
            if targets in ("vec", "both"):
                vec_result = await self._adapter.sync(
                    vector_index=self._vector_index,
                    embedding_model=self._embedding_model,
                )
                result.vec_result = vec_result

            # Graph ingest
            if targets in ("graph", "both"):
                try:
                    from qortex.sources.postgres_graph import (
                        PostgresGraphIngestor as PGGraphIngestor,
                    )

                    graph_ingestor = PGGraphIngestor(
                        config=self._config,
                        ingest_config=self._ingest,
                        backend=self._backend,
                        embedding_model=self._embedding_model,
                    )
                    graph_ingestor._conn = self._adapter.connection
                    schema_graph = await graph_ingestor.discover_schema()
                    mapping = graph_ingestor.map_schema(schema_graph)
                    graph_counts = await graph_ingestor.ingest(mapping, schema_graph)
                    result.graph_result = graph_counts
                except ImportError:
                    raise NotImplementedError(
                        "Graph ingestion requires the graph_ingestor module. "
                        "Use targets='vec' or install the graph layer."
                    )

        finally:
            await self._adapter.disconnect()

        result.duration_seconds = round(time.monotonic() - start, 2)
        return result
