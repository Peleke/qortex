"""Retrieval adapters: the seam between vector search and graph reasoning.

Architecture:
    VecOnlyAdapter  — pure vector similarity, no graph. Default for qortex[vec].
    GraphRAGAdapter — vec seeds → PPR over graph → rule collection. Requires qortex[all].

The MCP server and QortexClient both use RetrievalAdapter, so swapping between
vec-only and GraphRAG is transparent to consumers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from qortex.core.backend import GraphBackend
from qortex.observability import emit
from qortex.observability.events import (
    FeedbackReceived,
    KGCoverageComputed,
    OnlineEdgesGenerated,
    QueryCompleted,
    QueryStarted,
    VecSearchCompleted,
    VecSeedYield,
)
from qortex.observability.logging import get_logger

if TYPE_CHECKING:
    from qortex.hippocampus.buffer import EdgePromotionBuffer
    from qortex.hippocampus.factors import TeleportationFactors
    from qortex.hippocampus.interoception import InteroceptionProvider

logger = get_logger(__name__)


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
        t0 = time.perf_counter()

        emit(QueryStarted(
            query_id=query_id,
            query_text=query,
            domains=tuple(domains) if domains else None,
            mode="vec",
            top_k=top_k,
            timestamp=datetime.now(UTC).isoformat(),
        ))

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

        elapsed = (time.perf_counter() - t0) * 1000
        activated = [item.node_id for item in items if item.node_id]

        emit(QueryCompleted(
            query_id=query_id,
            latency_ms=elapsed,
            seed_count=len(vec_results),
            result_count=len(items),
            activated_nodes=len(activated),
            mode="vec",
            timestamp=datetime.now(UTC).isoformat(),
        ))

        return RetrievalResult(
            items=items,
            query_id=query_id,
            activated_nodes=activated,
        )

    def feedback(self, query_id: str, outcomes: dict[str, str]) -> None:
        """No-op for vec-only — no teleportation factors to update."""
        logger.debug("vec.feedback.ignored", query_id=query_id)


class GraphRAGAdapter:
    """Vec seeds → PPR over merged graph → rule collection.

    The full hybrid retrieval pipeline:

    1. Embed query → VectorIndex.search() → seed node IDs + scores
    2. Online cosine-sim graph gen between candidates (fills KG gaps)
    3. Persistent KG edges for these nodes (typed: REQUIRES, REFINES...)
    4. Merge online + persistent edges (persistent get weight bonus)
    5. PPR over merged graph (with teleportation factors from feedback)
    6. Score: vec_sim × PPR_activation
    7. Collect rules from activated concepts
    8. Buffer online edges for future promotion

    Vec layer and graph layer are independent — this adapter composes them
    for graph-enhanced retrieval.
    """

    def __init__(
        self,
        vector_index,
        backend: GraphBackend,
        embedding_model,
        factors: TeleportationFactors | None = None,
        edge_buffer: EdgePromotionBuffer | None = None,
        online_sim_threshold: float = 0.7,
        kg_weight_bonus: float = 0.3,
        interoception: InteroceptionProvider | None = None,
    ) -> None:
        from qortex.hippocampus.interoception import (
            LocalInteroceptionProvider,
        )

        self.vector_index = vector_index
        self.backend = backend
        self.embedding_model = embedding_model
        self.online_sim_threshold = online_sim_threshold
        self.kg_weight_bonus = kg_weight_bonus

        # Interoception layer: prefer explicit, else wrap legacy params
        if interoception is not None:
            self._interoception = interoception
            if factors is not None or edge_buffer is not None:
                logger.warning(
                    "Both interoception= and factors=/edge_buffer= provided. "
                    "Using interoception=, ignoring factors=/edge_buffer=."
                )
        elif factors is not None or edge_buffer is not None:
            # Backward compat: wrap legacy params in LocalInteroceptionProvider
            provider = LocalInteroceptionProvider()
            if factors is not None:
                provider._factors = factors
            if edge_buffer is not None:
                provider._buffer = edge_buffer
            self._interoception = provider
        else:
            self._interoception = LocalInteroceptionProvider()

        # Give interoception a backend reference for auto-flush
        if hasattr(self._interoception, "set_backend"):
            self._interoception.set_backend(self.backend)

        # Cache: query_id → list of item_ids (for feedback routing)
        self._query_cache: dict[str, list[str]] = {}

    def retrieve(
        self,
        query: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        query_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        emit(QueryStarted(
            query_id=query_id,
            query_text=query,
            domains=tuple(domains) if domains else None,
            mode="graph",
            top_k=top_k,
            timestamp=datetime.now(UTC).isoformat(),
        ))

        # 1. Vec layer: embed query → vector search → seed candidates
        query_embedding = self.embedding_model.embed([query])[0]
        fetch_k = max(top_k * 3, 30)  # Over-fetch for graph expansion
        vec_results = self.vector_index.search(
            query_embedding,
            top_k=fetch_k,
            threshold=min_confidence,
        )

        if not vec_results:
            return RetrievalResult(items=[], query_id=query_id)

        # Resolve nodes and filter by domain
        seed_nodes: list[tuple[str, float]] = []  # (node_id, vec_score)
        for node_id, score in vec_results:
            node = self.backend.get_node(node_id)
            if node is None:
                continue
            if domains and node.domain not in domains:
                continue
            seed_nodes.append((node_id, score))

        # Seed yield: how many vec results survived domain filtering
        emit(VecSeedYield(
            query_id=query_id,
            vec_candidates=len(vec_results),
            seeds_after_filter=len(seed_nodes),
            yield_ratio=len(seed_nodes) / max(len(vec_results), 1),
        ))

        if not seed_nodes:
            return RetrievalResult(items=[], query_id=query_id)

        seed_ids = [nid for nid, _ in seed_nodes]
        vec_scores = dict(seed_nodes)

        vec_elapsed = (time.perf_counter() - t0) * 1000
        emit(VecSearchCompleted(
            query_id=query_id,
            candidates=len(vec_results),
            fetch_k=fetch_k,
            latency_ms=vec_elapsed,
        ))

        # 2. Online edge generation: cosine sim between candidate pairs
        online_edges = self._build_online_edges(seed_ids)

        emit(OnlineEdgesGenerated(
            query_id=query_id,
            edge_count=len(online_edges),
            threshold=self.online_sim_threshold,
            seed_count=len(seed_ids),
        ))

        # 3. Buffer online edges for future promotion
        for src, tgt, weight in online_edges:
            self._interoception.record_online_edge(src, tgt, weight)

        # 4. Compute KG coverage (research metric)
        persistent_count = self._count_persistent_edges(seed_ids)
        total_edges = len(online_edges) + persistent_count
        kg_coverage = persistent_count / max(total_edges, 1)

        emit(KGCoverageComputed(
            query_id=query_id,
            persistent_edges=persistent_count,
            online_edges=len(online_edges),
            coverage=kg_coverage,
        ))

        # 5. PPR over merged graph (persistent edges in backend + online extras)
        seed_weights = self._interoception.get_seed_weights(seed_ids)

        ppr_scores = self.backend.personalized_pagerank(
            source_nodes=seed_ids,
            damping_factor=0.85,
            max_iterations=100,
            domain=domains[0] if domains and len(domains) == 1 else None,
            seed_weights=seed_weights,
            extra_edges=online_edges,
            query_id=query_id,
        )

        # 6. Combined scoring: vec_sim × PPR_activation
        combined: dict[str, float] = {}
        for node_id in set(list(vec_scores.keys()) + list(ppr_scores.keys())):
            vec_s = vec_scores.get(node_id, 0.0)
            ppr_s = ppr_scores.get(node_id, 0.0)
            # Hybrid score: both signals contribute
            combined[node_id] = vec_s * 0.5 + ppr_s * 0.5

        # Sort by combined score
        ranked = sorted(combined.items(), key=lambda x: -x[1])

        # 7. Build result items
        items: list[RetrievalItem] = []
        for node_id, score in ranked[:top_k]:
            node = self.backend.get_node(node_id)
            if node is None:
                continue
            items.append(
                RetrievalItem(
                    id=node.id,
                    content=f"{node.name}: {node.description}",
                    score=score,
                    domain=node.domain,
                    node_id=node.id,
                    metadata={
                        **node.properties,
                        "vec_score": round(vec_scores.get(node_id, 0.0), 4),
                        "ppr_score": round(ppr_scores.get(node_id, 0.0), 4),
                        "kg_coverage": round(kg_coverage, 4),
                    },
                )
            )

        # Cache query for feedback routing
        self._query_cache[query_id] = [item.id for item in items]

        elapsed = (time.perf_counter() - t0) * 1000
        emit(QueryCompleted(
            query_id=query_id,
            latency_ms=elapsed,
            seed_count=len(seed_ids),
            result_count=len(items),
            activated_nodes=len(ppr_scores),
            mode="graph",
            timestamp=datetime.now(UTC).isoformat(),
        ))

        return RetrievalResult(
            items=items,
            query_id=query_id,
            activated_nodes=list(ppr_scores.keys()),
        )

    def feedback(self, query_id: str, outcomes: dict[str, str]) -> None:
        """Update teleportation factors from feedback via interoception.

        Accepted items get boosted, rejected items get penalized.
        This biases future PPR toward nodes that produce good results.
        """
        emit(FeedbackReceived(
            query_id=query_id,
            outcomes=len(outcomes),
            accepted=sum(1 for v in outcomes.values() if v == "accepted"),
            rejected=sum(1 for v in outcomes.values() if v == "rejected"),
            partial=sum(1 for v in outcomes.values() if v == "partial"),
            source="adapter",
        ))
        self._interoception.report_outcome(query_id, outcomes)
        logger.debug(
            "graph.feedback.routed",
            query_id=query_id,
            outcome_count=len(outcomes),
        )

    @property
    def factors(self) -> TeleportationFactors:
        """Backward-compat: proxy to interoception's factors."""
        return self._interoception.factors

    @property
    def edge_buffer(self) -> EdgePromotionBuffer:
        """Backward-compat: proxy to interoception's buffer."""
        return self._interoception.buffer

    def _build_online_edges(
        self,
        seed_ids: list[str],
    ) -> list[tuple[str, str, float]]:
        """Generate ephemeral cosine-sim edges between candidate pairs.

        O(K²) where K = len(seed_ids). For K=30 this is 435 pairs — fast.
        """
        if len(seed_ids) < 2:
            return []

        # Get embeddings for all seeds
        embeddings: dict[str, list[float]] = {}
        for nid in seed_ids:
            emb = self.backend.get_embedding(nid)
            if emb is not None:
                embeddings[nid] = emb

        if len(embeddings) < 2:
            return []

        edges: list[tuple[str, str, float]] = []
        ids = list(embeddings.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = _cosine_similarity(embeddings[ids[i]], embeddings[ids[j]])
                if sim >= self.online_sim_threshold:
                    edges.append((ids[i], ids[j], sim))

        return edges

    def _count_persistent_edges(self, node_ids: list[str]) -> int:
        """Count how many persistent KG edges exist between candidate nodes."""
        count = 0
        node_set = set(node_ids)
        for nid in node_ids:
            for edge in self.backend.get_edges(nid, direction="out"):
                if edge.target_id in node_set:
                    count += 1
        return count


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Pure Python — no numpy required."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_adapter(
    vector_index,
    backend: GraphBackend,
    embedding_model=None,
    factors: TeleportationFactors | None = None,
    edge_buffer: EdgePromotionBuffer | None = None,
    interoception: InteroceptionProvider | None = None,
) -> RetrievalAdapter:
    """Factory: returns the best available adapter.

    - If graph supports MAGE: GraphRAGAdapter (PPR via Memgraph)
    - If InMemoryBackend with edges: GraphRAGAdapter (PPR via power iteration)
    - Otherwise: VecOnlyAdapter (pure cosine similarity)
    """
    if embedding_model is None:
        raise ValueError("No embedding model — cannot create retrieval adapter")

    # GraphRAGAdapter works with any backend that has PPR — including InMemoryBackend
    return GraphRAGAdapter(
        vector_index,
        backend,
        embedding_model,
        factors=factors,
        edge_buffer=edge_buffer,
        interoception=interoception,
    )
