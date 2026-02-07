"""Agno adapter: qortex as a knowledge source for Agno agents.

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.agno import QortexKnowledge

    client = LocalQortexClient(vector_index, backend, embedding_model)
    knowledge = QortexKnowledge(client=client, domains=["security"])

    # Works anywhere agno expects a knowledge source
    docs = knowledge.retrieve("What is OAuth2?")
    # → [{"content": "...", "id": "...", "name": "...", "meta_data": {...}, "reranking_score": 0.92}]

Does NOT require agno to be installed — returns plain dicts matching
agno's Document shape when agno is not available.
"""

from __future__ import annotations

from typing import Any

from qortex.client import QortexClient


class QortexKnowledge:
    """Qortex as an Agno knowledge source.

    Implements agno's knowledge protocol: retrieve() → list[Document].
    Returns agno Document instances if agno is installed, otherwise
    returns dicts with the same shape.
    """

    def __init__(
        self,
        client: QortexClient,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
        feedback_source: str = "agno",
    ) -> None:
        self._client = client
        self._domains = domains
        self._top_k = top_k
        self._min_confidence = min_confidence
        self._feedback_source = feedback_source
        self._last_query_id: str | None = None

    def retrieve(self, query: str, **kwargs) -> list[Any]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query text.
            **kwargs: Additional args (top_k, min_confidence overrides).

        Returns:
            List of agno Documents (or dicts with same shape).
        """
        top_k = kwargs.get("top_k", self._top_k)
        min_confidence = kwargs.get("min_confidence", self._min_confidence)

        result = self._client.query(
            context=query,
            domains=self._domains,
            top_k=top_k,
            min_confidence=min_confidence,
        )
        self._last_query_id = result.query_id

        return [item.to_agno_document() for item in result.items]

    def build_context(self, query: str = "", **kwargs) -> str:
        """Build a context string from retrieved documents.

        Agno calls this to build context for prompt injection.
        """
        docs = self.retrieve(query, **kwargs)
        parts = []
        for doc in docs:
            content = doc.content if hasattr(doc, "content") else doc.get("content", "")
            parts.append(content)
        return "\n\n".join(parts)

    def feedback(self, outcomes: dict[str, str]) -> None:
        """Report feedback for the last retrieval. Closes the learning loop."""
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
