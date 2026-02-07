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
    """Passthrough: pure vector search, no graph.

    Default when qortex[vec] only (no graph backend or HippoRAG).
    Embeds query → vector_search on backend → wrap as RetrievalResult.
    """

    def __init__(self, backend: GraphBackend, embedding_model) -> None:
        self.backend = backend
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        query_id = str(uuid.uuid4())

        # Embed query
        query_embedding = self.embedding_model.embed([query])[0]

        # Search each domain (or all)
        if domains:
            all_results: list[tuple[ConceptNode, float]] = []
            for domain in domains:
                results = self.backend.vector_search(
                    query_embedding, domain=domain, top_k=top_k, threshold=min_confidence
                )
                all_results.extend(results)
            # Re-sort by score and truncate
            all_results.sort(key=lambda x: -x[1])
            all_results = all_results[:top_k]
        else:
            all_results = self.backend.vector_search(
                query_embedding, domain=None, top_k=top_k, threshold=min_confidence
            )

        items = [
            RetrievalItem(
                id=node.id,
                content=f"{node.name}: {node.description}",
                score=score,
                domain=node.domain,
                node_id=node.id,
                metadata=node.properties,
            )
            for node, score in all_results
        ]

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

    1. Embed query → vector_search → seed nodes
    2. PPR from seeds (with teleportation factors)
    3. Collect rules from activated concepts
    4. Return combined results

    Implementation: Phase 4.
    """

    def __init__(
        self,
        backend: GraphBackend,
        embedding_model,
        teleportation_factors: dict[str, float] | None = None,
    ) -> None:
        self.backend = backend
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
    backend: GraphBackend,
    embedding_model=None,
    teleportation_factors: dict[str, float] | None = None,
) -> RetrievalAdapter:
    """Factory: returns HippoRAGAdapter if graph supports MAGE, VecOnlyAdapter otherwise."""
    if embedding_model is None:
        raise ValueError("No embedding model — cannot create retrieval adapter")

    if backend.supports_mage():
        return HippoRAGAdapter(backend, embedding_model, teleportation_factors)

    return VecOnlyAdapter(backend, embedding_model)
