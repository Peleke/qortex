"""Source adapters: connect external databases to qortex vec + graph layers.

Architecture:
    SourceAdapter protocol → connect/discover/read/sync → VectorIndex + GraphBackend
    RowSerializer protocol → row dict → text for embedding
    GraphIngestor protocol → discover_schema/map_schema/ingest → GraphBackend
    Mapping rules → FK classification, constraint→rule, catalog detection

Implementations:
    PostgresSourceAdapter (async, via asyncpg) — PR 3
    PostgresGraphIngestor (async, via asyncpg) — PR 5
    SqliteSourceAdapter (future)
    MongoSourceAdapter (future)
"""

from qortex.sources.base import (
    ColumnSchema,
    IngestConfig,
    SourceAdapter,
    SourceConfig,
    SyncResult,
    TableSchema,
)
from qortex.sources.graph_ingestor import (
    CheckConstraintInfo,
    CrossDatabaseEdge,
    EdgeMapping,
    ForeignKeyInfo,
    GraphIngestor,
    GraphMapping,
    RuleMapping,
    SchemaGraph,
    TableMapping,
    TableSchemaFull,
    UniqueConstraintInfo,
)
from qortex.sources.mapping_rules import (
    auto_domain,
    classify_fk_relation,
    constraint_to_rule,
    create_user_identity_edges,
    detect_catalog_table,
    detect_cross_db_edges_by_naming,
    find_description_columns,
    find_name_column,
)
from qortex.sources.registry import SourceRegistry
from qortex.sources.serializer import (
    KeyValueSerializer,
    NaturalLanguageSerializer,
    RowSerializer,
)

__all__ = [
    "CheckConstraintInfo",
    "ColumnSchema",
    "CrossDatabaseEdge",
    "EdgeMapping",
    "ForeignKeyInfo",
    "GraphIngestor",
    "GraphMapping",
    "IngestConfig",
    "KeyValueSerializer",
    "NaturalLanguageSerializer",
    "RowSerializer",
    "RuleMapping",
    "SchemaGraph",
    "SourceAdapter",
    "SourceConfig",
    "SourceRegistry",
    "SyncResult",
    "TableMapping",
    "TableSchema",
    "TableSchemaFull",
    "UniqueConstraintInfo",
    "auto_domain",
    "classify_fk_relation",
    "constraint_to_rule",
    "create_user_identity_edges",
    "detect_catalog_table",
    "detect_cross_db_edges_by_naming",
    "find_description_columns",
    "find_name_column",
]

# Conditional PostgresSourceAdapter import
try:
    from qortex.sources.postgres import PostgresIngestor, PostgresSourceAdapter

    __all__ += ["PostgresSourceAdapter", "PostgresIngestor"]
except ImportError:
    pass

# Conditional imports for graph implementations
try:
    from qortex.sources.postgres_graph import PostgresGraphIngestor

    __all__ += ["PostgresGraphIngestor"]
except ImportError:
    pass

try:
    from qortex.sources.cross_db import apply_cross_db_edges, discover_all_cross_db

    __all__ += ["apply_cross_db_edges", "discover_all_cross_db"]
except ImportError:
    pass
