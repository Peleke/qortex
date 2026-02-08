"""Source adapters: connect external databases to qortex vec + graph layers.

Architecture:
    SourceAdapter protocol → connect/discover/read/sync → VectorIndex + GraphBackend
    RowSerializer protocol → row dict → text for embedding

Implementations:
    PostgresSourceAdapter (async, via asyncpg) — PR 3
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
    GraphIngestor,
    GraphMapping,
    SchemaGraph,
    TableSchemaFull,
)
from qortex.sources.mapping_rules import (
    classify_fk_relation,
    constraint_to_rule,
    detect_catalog_table,
    detect_cross_db_edges_by_naming,
)
from qortex.sources.registry import SourceRegistry
from qortex.sources.serializer import (
    KeyValueSerializer,
    NaturalLanguageSerializer,
    RowSerializer,
)

__all__ = [
    "ColumnSchema",
    "GraphIngestor",
    "GraphMapping",
    "IngestConfig",
    "KeyValueSerializer",
    "NaturalLanguageSerializer",
    "RowSerializer",
    "SchemaGraph",
    "SourceAdapter",
    "SourceConfig",
    "SourceRegistry",
    "SyncResult",
    "TableSchema",
    "TableSchemaFull",
    "classify_fk_relation",
    "constraint_to_rule",
    "detect_catalog_table",
    "detect_cross_db_edges_by_naming",
]

# Conditional PostgresSourceAdapter import
try:
    from qortex.sources.postgres import PostgresIngestor, PostgresSourceAdapter

    __all__ += ["PostgresSourceAdapter", "PostgresIngestor"]
except ImportError:
    pass
