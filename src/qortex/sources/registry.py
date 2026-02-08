"""SourceRegistry: manages source adapters by source_id.

Holds active adapter connections and their configs. Used by the MCP server
and client to manage multiple database connections.
"""

from __future__ import annotations

import logging
from typing import Any

from qortex.sources.base import SourceConfig, TableSchema

logger = logging.getLogger(__name__)


class SourceRegistry:
    """Registry of active source adapter connections."""

    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}  # source_id → SourceAdapter
        self._configs: dict[str, SourceConfig] = {}
        self._schemas: dict[str, list[TableSchema]] = {}  # cached discovery

    def register(self, config: SourceConfig, adapter: Any) -> None:
        """Register an adapter for a source."""
        self._adapters[config.source_id] = adapter
        self._configs[config.source_id] = config

    def get(self, source_id: str) -> Any | None:
        """Get an adapter by source_id."""
        return self._adapters.get(source_id)

    def get_config(self, source_id: str) -> SourceConfig | None:
        """Get config by source_id."""
        return self._configs.get(source_id)

    async def remove_async(self, source_id: str) -> bool:
        """Remove an adapter, disconnecting it first. Returns True if it existed."""
        adapter = self._adapters.pop(source_id, None)
        self._configs.pop(source_id, None)
        self._schemas.pop(source_id, None)
        if adapter is not None:
            if hasattr(adapter, "disconnect"):
                try:
                    await adapter.disconnect()
                except Exception:
                    logger.debug(
                        "Error disconnecting adapter %s on remove", source_id, exc_info=True
                    )
            return True
        return False

    def remove(self, source_id: str) -> bool:
        """Remove an adapter without disconnecting. Use remove_async for cleanup."""
        existed = source_id in self._adapters
        self._adapters.pop(source_id, None)
        self._configs.pop(source_id, None)
        self._schemas.pop(source_id, None)
        return existed

    def list_sources(self) -> list[str]:
        """List all registered source IDs."""
        return list(self._adapters.keys())

    def cache_schemas(self, source_id: str, schemas: list[TableSchema]) -> None:
        """Cache discovered schemas for a source."""
        self._schemas[source_id] = schemas

    def get_schemas(self, source_id: str) -> list[TableSchema] | None:
        """Get cached schemas for a source."""
        return self._schemas.get(source_id)

    async def clear_async(self) -> None:
        """Clear all registered adapters, disconnecting each first."""
        for source_id, adapter in list(self._adapters.items()):
            if hasattr(adapter, "disconnect"):
                try:
                    await adapter.disconnect()
                except Exception:
                    logger.debug(
                        "Error disconnecting adapter %s on clear", source_id, exc_info=True
                    )
        self._adapters.clear()
        self._configs.clear()
        self._schemas.clear()

    def clear(self) -> None:
        """Clear all registered adapters without disconnecting. Use clear_async for cleanup."""
        self._adapters.clear()
        self._configs.clear()
        self._schemas.clear()
