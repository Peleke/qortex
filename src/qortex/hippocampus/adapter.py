"""Retrieval adapters: the seam between vector search and graph reasoning.

Architecture:
    VecOnlyAdapter  — pure vector similarity, no graph. Default for qortex[vec].
    HippoRAGAdapter — vec seeds → PPR over graph → rule collection. Requires qortex[all].

The MCP server and QortexClient both use RetrievalAdapter, so swapping between
vec-only and HippoRAG is transparent to consumers.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from qortex.core.backend import GraphBackend
from qortex.core.models import ConceptNode

logger = logging.getLogger(__name__)


@dataclass
class RetrievalItem:
    """A single item returned by retrieval."""

    id: str
    content: str
    score: float
    domain: str
    node_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""

    items: list[RetrievalItem]
    query_id: str
    activated_nodes: list[str] = field(default_factory=list)


@runtime_checkable
class RetrievalAdapter(Protocol):
    """The adapter between vector search and graph-based reasoning.

    Vec layer finds candidates. This adapter optionally enriches them
    via graph structure (PPR, rule collection, transitive activation).
    When no graph is available, passes through vec results unchanged.
    """

    def retrieve(
        self,
        query: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        """Retrieve relevant items for a query context."""
        ...

    def feedback(self, query_id: str, outcomes: dict[str, str]) -> None:
        """Report outcomes for retrieved items.

        Args:
            query_id: The query_id from RetrievalResult.
            outcomes: Mapping of item_id → "accepted" | "rejected" | "partial".
        """
        ...


class VecOnlyAdapter:
    """Pure vector similarity search, no graph reasoning.

    Default adapter when only qortex[vec] is installed. Searches the
    VectorIndex directly for candidate IDs, then resolves node metadata
    from the GraphBackend.

    Vec layer and graph layer are independent — this adapter composes them.
    """

    def __init__(self, vector_index, backend: GraphBackend, embedding_model) -> None:
        self.vector_index = vector_index  # VectorIndex: similarity search
        self.backend = backend  # GraphBackend: node metadata lookup
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        query_id = str(uuid.uuid4())

        # 1. Embed query
        query_embedding = self.embedding_model.embed([query])[0]

        # 2. Search VectorIndex directly (not backend.vector_search)
        # Over-fetch to allow for domain filtering
        fetch_k = top_k * 2 if domains else top_k
        vec_results = self.vector_index.search(
            query_embedding, top_k=fetch_k, threshold=min_confidence
        )

        # 3. Resolve node metadata from backend and filter by domain
        items: list[RetrievalItem] = []
        for node_id, score in vec_results:
            node = self.backend.get_node(node_id)
            if node is None:
                continue
            if domains and node.domain not in domains:
                continue
            items.append(
                RetrievalItem(
                    id=node.id,
                    content=f"{node.name}: {node.description}",
                    score=score,
                    domain=node.domain,
                    node_id=node.id,
                    metadata=node.properties,
                )
            )
            if len(items) >= top_k:
                break

        return RetrievalResult(
            items=items,
            query_id=query_id,
            activated_nodes=[item.node_id for item in items if item.node_id],
        )

    def feedback(self, query_id: str, outcomes: dict[str, str]) -> None:
        """No-op for vec-only — no teleportation factors to update."""
        logger.debug("VecOnlyAdapter: feedback ignored (no graph), query_id=%s", query_id)


class HippoRAGAdapter:
    """Vec seeds → PPR over graph → rule collection. Requires qortex[all].

    1. Embed query → VectorIndex.search() → seed nodes
    2. Online cosine-sim graph gen (fills KG gaps) + persistent KG edges
    3. PPR over merged graph (with teleportation factors)
    4. Collect rules from activated concepts
    5. Return combined results

    Vec layer and graph layer are independent — this adapter composes them
    for graph-enhanced retrieval.

    Implementation: Phase 4.
    """

    def __init__(
        self,
        vector_index,
        backend: GraphBackend,
        embedding_model,
        teleportation_factors: dict[str, float] | None = None,
    ) -> None:
        self.vector_index = vector_index  # VectorIndex: find seed nodes
        self.backend = backend  # GraphBackend: PPR + rules + node metadata
        self.embedding_model = embedding_model
        self.teleportation_factors = teleportation_factors or {}

    def retrieve(
        self,
        query: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        raise NotImplementedError("HippoRAGAdapter: Phase 4 — vec seeds → PPR → rules")

    def feedback(self, query_id: str, outcomes: dict[str, str]) -> None:
        raise NotImplementedError("HippoRAGAdapter: Phase 4 — teleportation factor updates")


def get_adapter(
    vector_index,
    backend: GraphBackend,
    embedding_model=None,
    teleportation_factors: dict[str, float] | None = None,
) -> RetrievalAdapter:
    """Factory: returns HippoRAGAdapter if graph supports MAGE, VecOnlyAdapter otherwise."""
    if embedding_model is None:
        raise ValueError("No embedding model — cannot create retrieval adapter")

    if backend.supports_mage():
        return HippoRAGAdapter(vector_index, backend, embedding_model, teleportation_factors)

    return VecOnlyAdapter(vector_index, backend, embedding_model)
