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
    SourceAdapter,
    SourceConfig,
    SyncResult,
    TableSchema,
)
from qortex.sources.registry import SourceRegistry
from qortex.sources.serializer import (
    KeyValueSerializer,
    NaturalLanguageSerializer,
    RowSerializer,
)

__all__ = [
    "ColumnSchema",
    "KeyValueSerializer",
    "NaturalLanguageSerializer",
    "RowSerializer",
    "SourceAdapter",
    "SourceConfig",
    "SourceRegistry",
    "SyncResult",
    "TableSchema",
]
