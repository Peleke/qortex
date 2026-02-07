"""Mastra adapter: qortex as a MastraVector-compatible store.

This provides a Python-side implementation that maps Mastra's vector store
interface to qortex. The TypeScript @qortex/mastra package (future) will
implement MastraVector by talking to qortex MCP.

Mastra's MastraVector has 9 methods:
    query, upsert, createIndex, listIndexes, describeIndex,
    deleteIndex, updateVector, deleteVector, deleteVectors

We implement the ones that map cleanly. The rest raise NotImplementedError
with clear explanations of why (qortex is file-based ingestion, not
arbitrary vector upsert).

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.mastra import QortexVectorStore

    client = LocalQortexClient(vector_index, backend, embedding_model)
    store = QortexVectorStore(client=client)

    # Works like Mastra's vector stores
    results = store.query(index_name="security", query_vector=[...], top_k=10)
    # → [{"id": "...", "score": 0.92, "metadata": {...}, "document": "..."}]

    indexes = store.list_indexes()
    # → [{"name": "security", "dimension": 384, "metric": "cosine", "count": 42}]

Does NOT require Mastra to be installed.
"""

from __future__ import annotations

from typing import Any

from qortex.client import QortexClient


class QortexVectorStore:
    """Qortex as a Mastra-compatible vector store.

    Maps Mastra's MastraVector interface to QortexClient.
    Mastra domains = qortex domains (1:1 mapping).

    Upgrade path:
        Level 0: Vec-only (this adapter) — parity with PgVector/Chroma
        Level 1: query(mode="graph") → PPR-enhanced results (future)
        Level 2: Feedback → teleportation factors → improves over time
    """

    def __init__(
        self,
        client: QortexClient,
        feedback_source: str = "mastra",
    ) -> None:
        self._client = client
        self._feedback_source = feedback_source
        self._last_query_id: str | None = None

    def query(
        self,
        index_name: str,
        query_vector: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
        include_vector: bool = False,
    ) -> list[dict[str, Any]]:
        """Query vectors by similarity.

        Mastra signature: query(indexName, queryVector, topK, filter, includeVector)
        qortex uses text queries (embedding happens server-side), so we prefer
        query_text. If only query_vector is provided, we pass it through
        but this is less efficient.

        Args:
            index_name: Domain name (Mastra calls these "indexes").
            query_vector: Raw embedding vector (optional, prefer query_text).
            query_text: Text query (preferred — qortex embeds server-side).
            top_k: Maximum results.
            filter: Metadata filter (not yet supported).
            include_vector: Include vector in results (not supported).

        Returns:
            List of Mastra QueryResult dicts: {id, score, metadata, document}
        """
        if query_text is None and query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")

        if query_text is None:
            raise NotImplementedError(
                "query_vector without query_text is not yet supported. "
                "qortex embeds server-side — pass query_text for meaningful results. "
                "Direct vector search will be supported when QortexClient exposes vector_query()."
            )

        result = self._client.query(
            context=query_text,
            domains=[index_name],
            top_k=top_k,
        )
        self._last_query_id = result.query_id

        return [item.to_mastra_result() for item in result.items]

    def upsert(
        self,
        index_name: str,
        vectors: list[dict[str, Any]],
    ) -> None:
        """Upsert vectors into an index.

        Mastra's upsert takes {id, vector, metadata} dicts. Qortex is
        file-based ingestion, not arbitrary vector upsert. Use
        client.ingest() for file-based ingestion.

        For direct vector insertion, this writes to the underlying
        VectorIndex if the client exposes it.
        """
        raise NotImplementedError(
            "qortex uses file-based ingestion (client.ingest()), not arbitrary vector upsert. "
            "Use client.ingest(source_path, domain) to ingest files."
        )

    def create_index(
        self,
        name: str,
        dimension: int = 384,
        metric: str = "cosine",
    ) -> None:
        """Create a new index (domain).

        In qortex, domains are auto-created on first ingest. This is a no-op
        that records the intent.
        """
        # Domains auto-create on ingest in qortex
        pass

    def list_indexes(self) -> list[dict[str, Any]]:
        """List available indexes (domains).

        Returns Mastra's IndexStats shape:
            {name, dimension, metric, count}
        """
        domains = self._client.domains()
        dimension = 384  # default; could be queried from embedding model

        return [
            {
                "name": d.name,
                "dimension": dimension,
                "metric": "cosine",
                "count": d.concept_count,
            }
            for d in domains
        ]

    def describe_index(self, name: str) -> dict[str, Any]:
        """Describe a specific index (domain)."""
        domains = self._client.domains()
        for d in domains:
            if d.name == name:
                return {
                    "name": d.name,
                    "dimension": 384,
                    "metric": "cosine",
                    "count": d.concept_count,
                }
        raise ValueError(f"Index (domain) '{name}' not found")

    def delete_index(self, name: str) -> None:
        """Delete an index (domain). Not yet supported."""
        raise NotImplementedError("Domain deletion not yet supported via QortexClient")

    def feedback(self, outcomes: dict[str, str]) -> None:
        """Report feedback for the last query. Closes the learning loop.

        This is qortex's unique advantage over Mastra — feedback drives
        teleportation factor updates that improve future retrieval.
        """
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
