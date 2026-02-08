"""Core types and protocol for source adapters.

SourceAdapter connects to an external database, discovers its schema,
reads rows, and syncs them into qortex's vec + graph layers.

All types here are pure dataclasses — no database drivers needed.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class ColumnSchema:
    """Schema for a single database column."""

    name: str
    data_type: str  # e.g. "text", "integer", "uuid", "jsonb"
    nullable: bool = True
    is_pk: bool = False
    is_fk: bool = False
    foreign_table: str | None = None
    foreign_column: str | None = None


@dataclass
class TableSchema:
    """Schema for a database table."""

    name: str
    columns: list[ColumnSchema] = field(default_factory=list)
    row_count: int = 0
    schema_name: str = "public"

    @property
    def pk_columns(self) -> list[str]:
        """Primary key column names."""
        return [c.name for c in self.columns if c.is_pk]

    @property
    def fk_columns(self) -> list[ColumnSchema]:
        """Foreign key columns."""
        return [c for c in self.columns if c.is_fk]

    @property
    def full_name(self) -> str:
        """schema.table qualified name."""
        return f"{self.schema_name}.{self.name}"


@dataclass
class SyncResult:
    """Result of syncing a source to qortex."""

    source_id: str
    tables_synced: int = 0
    rows_added: int = 0
    rows_updated: int = 0
    rows_deleted: int = 0
    vectors_created: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class SourceConfig:
    """Configuration for connecting a source database.

    domain_map maps glob patterns to domain names:
        {"habits.*": "habits", "swae_movements.*": "exercise"}
    If a table doesn't match any pattern, it maps to default_domain.
    """

    source_id: str
    connection_string: str
    schemas: list[str] = field(default_factory=lambda: ["public"])
    tables: list[str] | None = None  # None = all tables
    exclude_tables: list[str] = field(default_factory=list)
    domain_map: dict[str, str] = field(default_factory=dict)
    default_domain: str = "default"
    serialize_strategy: str = "natural_language"  # or "key_value"
    batch_size: int = 500

    def resolve_domain(self, table_name: str) -> str:
        """Resolve a table name to a domain via domain_map globs."""
        import fnmatch

        for pattern, domain in self.domain_map.items():
            if fnmatch.fnmatch(table_name, pattern):
                return domain
        return self.default_domain

    def __repr__(self) -> str:
        """Redact connection_string to prevent credential leaks in logs."""
        return (
            f"SourceConfig(source_id={self.source_id!r}, "
            f"connection_string='***', "
            f"schemas={self.schemas!r}, "
            f"domain_map={self.domain_map!r})"
        )


# ---------------------------------------------------------------------------
# IngestConfig: user-friendly configuration for database ingestion
# ---------------------------------------------------------------------------

_VALID_TARGETS = {"vec", "graph", "both"}
_VALID_MODES = {"online", "batch"}
_VALID_STRATEGIES = {"natural_language", "key_value"}


@dataclass
class IngestConfig:
    """User-friendly configuration for database ingestion.

    Controls what the unified PostgresIngestor does on ``run()``:
    - targets: which layers to populate ("vec", "graph", or "both")
    - mode: "batch" processes all rows at once, "online" streams changes
    - serialize_strategy: how rows become text for embedding
    - embed_catalog_tables: whether to embed reference/catalog rows
    - extract_rules: whether to extract CHECK constraints as rules
    - user_filter: only ingest rows for this user_id (multi-tenant)

    Environment overrides (following QORTEX_VEC/QORTEX_GRAPH pattern):
        QORTEX_INGEST_TARGETS=vec|graph|both
        QORTEX_INGEST_MODE=online|batch
        QORTEX_INGEST_BATCH_SIZE=500
    """

    targets: Literal["vec", "graph", "both"] = "both"
    mode: Literal["online", "batch"] = "batch"
    batch_size: int = 500
    serialize_strategy: Literal["natural_language", "key_value"] = "natural_language"
    embed_catalog_tables: bool = True
    extract_rules: bool = True
    user_filter: str | None = None

    def __post_init__(self) -> None:
        # Apply environment overrides
        env_targets = os.environ.get("QORTEX_INGEST_TARGETS")
        if env_targets and env_targets in _VALID_TARGETS:
            object.__setattr__(self, "targets", env_targets)

        env_mode = os.environ.get("QORTEX_INGEST_MODE")
        if env_mode and env_mode in _VALID_MODES:
            object.__setattr__(self, "mode", env_mode)

        env_batch = os.environ.get("QORTEX_INGEST_BATCH_SIZE")
        if env_batch:
            try:
                object.__setattr__(self, "batch_size", int(env_batch))
            except ValueError:
                pass

        # Validate
        if self.targets not in _VALID_TARGETS:
            raise ValueError(f"Invalid targets={self.targets!r}. Must be one of: {_VALID_TARGETS}")
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Invalid mode={self.mode!r}. Must be one of: {_VALID_MODES}")
        if self.serialize_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Invalid serialize_strategy={self.serialize_strategy!r}. "
                f"Must be one of: {_VALID_STRATEGIES}"
            )


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol for source database adapters.

    All methods are async — database I/O is inherently async.
    Sync wrappers at the client/MCP boundary use asyncio.run().
    """

    async def connect(self, config: SourceConfig) -> None:
        """Connect to the source database."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the source database."""
        ...

    async def discover(self) -> list[TableSchema]:
        """Discover table schemas from the connected database."""
        ...

    async def read_rows(
        self,
        table: str,
        batch_size: int = 500,
        offset: int = 0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Read rows from a table in batches."""
        ...

    async def sync(
        self,
        tables: list[str] | None = None,
        mode: str = "full",
    ) -> SyncResult:
        """Sync source data into qortex.

        Args:
            tables: Tables to sync. None = all discovered tables.
            mode: "full" = re-sync everything, "incremental" = only changes.
        """
        ...
