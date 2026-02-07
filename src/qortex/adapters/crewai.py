"""CrewAI adapter: drop-in KnowledgeStorage backed by qortex.

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.crewai import QortexKnowledgeStorage

    client = LocalQortexClient(vector_index, backend, embedding_model)
    storage = QortexKnowledgeStorage(client=client, domains=["security"])

    # Works anywhere crewai expects a KnowledgeStorage
    results = storage.search("What is OAuth2?", limit=5, score_threshold=0.5)
    # → [{"id": "...", "content": "...", "metadata": {...}, "score": 0.92}]

Does NOT require crewai to be installed — returns plain dicts matching
crewai's SearchResult TypedDict shape.
"""

from __future__ import annotations

from typing import Any

from qortex.client import QortexClient


class QortexKnowledgeStorage:
    """Drop-in replacement for crewai's KnowledgeStorage.

    Backed by QortexClient instead of ChromaDB.
    Returns dicts matching crewai's SearchResult TypedDict:
        {id: str, content: str, metadata: dict, score: float}
    """

    def __init__(
        self,
        client: QortexClient,
        domains: list[str] | None = None,
        feedback_source: str = "crewai",
    ) -> None:
        self._client = client
        self._domains = domains
        self._feedback_source = feedback_source
        self._last_query_id: str | None = None

    def search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search for relevant knowledge.

        Args:
            query: Search query text.
            limit: Maximum results.
            metadata_filter: Not yet supported (logged and ignored).
            score_threshold: Minimum score (0.0 to 1.0).

        Returns:
            List of SearchResult dicts: {id, content, metadata, score}
        """
        result = self._client.query(
            context=query,
            domains=self._domains,
            top_k=limit,
            min_confidence=score_threshold,
        )
        self._last_query_id = result.query_id

        return [item.to_crewai_result() for item in result.items]

    def save(self, documents: list[str]) -> None:
        """Save documents. CrewAI calls this to populate storage.

        For qortex, use client.ingest() instead — this is a no-op
        since qortex ingestion is file-based, not string-based.
        """
        pass

    def reset(self) -> None:
        """Reset storage. No-op for qortex (persistent storage)."""
        pass

    def feedback(self, outcomes: dict[str, str]) -> None:
        """Report feedback for the last search. Closes the learning loop."""
        if self._last_query_id is None:
            return
        self._client.feedback(
            query_id=self._last_query_id,
            outcomes=outcomes,
            source=self._feedback_source,
        )

    @property
    def last_query_id(self) -> str | None:
        return self._last_query_id
