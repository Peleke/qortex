"""Graph ingestor types: schema → knowledge graph mapping.

Layer 3 converts database schemas into qortex knowledge graphs:
- Tables → ConceptNodes
- Foreign keys → typed ConceptEdges (BELONGS_TO, INSTANCE_OF, etc.)
- Constraints → ExplicitRules
- Cross-database naming conventions → CrossDatabaseEdges

Types here are pure dataclasses. The PostgresGraphIngestor (PR 5)
implements the async discovery + ingestion logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ForeignKeyInfo:
    """A foreign key relationship in the database schema."""

    constraint_name: str
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    on_delete: str = "NO ACTION"  # CASCADE, RESTRICT, SET NULL, NO ACTION
    on_update: str = "NO ACTION"


@dataclass
class CheckConstraintInfo:
    """A CHECK constraint in the database schema."""

    constraint_name: str
    table_name: str
    check_clause: str  # e.g. "((calories >= 0))"


@dataclass
class UniqueConstraintInfo:
    """A UNIQUE constraint (including composite) in the database schema."""

    constraint_name: str
    table_name: str
    columns: list[str]


@dataclass
class TableSchemaFull:
    """Rich table schema with constraints — extends basic TableSchema.

    Includes FK, CHECK, UNIQUE constraints needed for graph mapping.
    """

    name: str
    schema_name: str = "public"
    columns: list[dict[str, Any]] = field(default_factory=list)
    # Each column dict: {name, data_type, nullable, is_pk, default, ...}
    row_count: int = 0
    foreign_keys: list[ForeignKeyInfo] = field(default_factory=list)
    check_constraints: list[CheckConstraintInfo] = field(default_factory=list)
    unique_constraints: list[UniqueConstraintInfo] = field(default_factory=list)

    @property
    def pk_columns(self) -> list[str]:
        return [c["name"] for c in self.columns if c.get("is_pk")]

    @property
    def full_name(self) -> str:
        return f"{self.schema_name}.{self.name}"

    def column_by_name(self, name: str) -> dict[str, Any] | None:
        for c in self.columns:
            if c["name"] == name:
                return c
        return None


@dataclass
class SchemaGraph:
    """Collection of table schemas from one or more databases."""

    source_id: str
    database_name: str
    tables: list[TableSchemaFull] = field(default_factory=list)

    def get_table(self, name: str) -> TableSchemaFull | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Graph mapping output types
# ---------------------------------------------------------------------------


@dataclass
class TableMapping:
    """Mapping from a database table to a qortex domain + concept template."""

    table_name: str
    domain: str
    name_column: str | None = None  # Column to use as concept name
    description_columns: list[str] = field(default_factory=list)
    is_catalog: bool = False  # True for reference tables (movements, templates)


@dataclass
class EdgeMapping:
    """Mapping from a foreign key to a typed edge."""

    source_table: str
    target_table: str
    fk_column: str
    relation_type: str  # RelationType value
    confidence: float = 1.0


@dataclass
class RuleMapping:
    """A rule derived from a database constraint."""

    table_name: str
    constraint_name: str
    rule_text: str
    category: str = "constraint"
    confidence: float = 1.0


@dataclass
class CrossDatabaseEdge:
    """An edge between tables in different databases.

    Detected by naming convention (e.g. exercise_id in practices
    → movements.id in the movements database).
    """

    source_database: str
    source_table: str
    source_column: str
    target_database: str
    target_table: str
    target_column: str
    relation_type: str  # RelationType value
    confidence: float = 0.8  # Lower than FK-derived edges (convention-based)
    description: str = ""


@dataclass
class GraphMapping:
    """Complete mapping from database schemas to qortex graph structure."""

    tables: list[TableMapping] = field(default_factory=list)
    edges: list[EdgeMapping] = field(default_factory=list)
    rules: list[RuleMapping] = field(default_factory=list)
    cross_db_edges: list[CrossDatabaseEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GraphIngestor protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphIngestor(Protocol):
    """Protocol for database → knowledge graph ingestion.

    discover_schema() reads the database schema (async, DB I/O).
    map_schema() converts schemas to graph mappings (sync, pure logic).
    ingest() writes mappings to the qortex backend (async, DB reads + backend writes).
    """

    async def discover_schema(self) -> SchemaGraph:
        """Discover full schema including constraints from the database."""
        ...

    def map_schema(self, schema: SchemaGraph) -> GraphMapping:
        """Map database schema to graph structure. Pure logic, no I/O."""
        ...

    async def ingest(self, mapping: GraphMapping, schema: SchemaGraph) -> dict[str, int]:
        """Ingest mapped data into qortex backend.

        Returns counts: {concepts, edges, rules}.
        """
        ...
