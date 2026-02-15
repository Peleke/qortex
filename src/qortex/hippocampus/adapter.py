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

import numpy as np

from qortex.core.backend import GraphBackend
from qortex.observe import emit
from qortex.observe.events import (
    FeedbackReceived,
    KGCoverageComputed,
    OnlineEdgesGenerated,
    QueryCompleted,
    QueryFailed,
    QueryStarted,
    VecSearchCompleted,
    VecSeedYield,
)
from qortex.observe.logging import get_logger
from qortex.observe.snapshot import config_snapshot_hash
from qortex.observe.tracing import _config_hash, get_overhead_timer, traced

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

        emit(
            QueryStarted(
                query_id=query_id,
                query_text=query,
                domains=tuple(domains) if domains else None,
                mode="vec",
                top_k=top_k,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        # 1. Embed query
        try:
            query_embedding = self.embedding_model.embed([query])[0]
        except Exception as exc:
            emit(
                QueryFailed(
                    query_id=query_id,
                    error=str(exc),
                    stage="embedding",
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            raise

        # 2. Search VectorIndex directly (not backend.vector_search)
        # Over-fetch to allow for domain filtering
        fetch_k = top_k * 2 if domains else top_k
        try:
            vec_results = self.vector_index.search(
                query_embedding, top_k=fetch_k, threshold=min_confidence
            )
        except Exception as exc:
            emit(
                QueryFailed(
                    query_id=query_id,
                    error=str(exc),
                    stage="vec_search",
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            raise

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

        emit(
            QueryCompleted(
                query_id=query_id,
                latency_ms=elapsed,
                seed_count=len(vec_results),
                result_count=len(items),
                activated_nodes=len(activated),
                mode="vec",
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

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

    @traced("retrieval.query")
    def retrieve(
        self,
        query: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> RetrievalResult:
        # Set config snapshot hash for this query's span tree
        _config_hash.set(
            config_snapshot_hash(
                learner_configs={"interoception": type(self._interoception).__name__},
            )
        )

        query_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        emit(
            QueryStarted(
                query_id=query_id,
                query_text=query,
                domains=tuple(domains) if domains else None,
                mode="graph",
                top_k=top_k,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        # 1. Vec layer: embed query → vector search → seed candidates
        seed_ids, vec_scores, vec_results_count = self._vec_search(
            query,
            top_k,
            min_confidence,
            domains,
            query_id,
            t0,
        )

        if seed_ids is None:
            return RetrievalResult(items=[], query_id=query_id)

        # 2. Online edge generation: cosine sim between candidate pairs
        online_edges = self._build_online_edges(seed_ids)

        emit(
            OnlineEdgesGenerated(
                query_id=query_id,
                edge_count=len(online_edges),
                threshold=self.online_sim_threshold,
                seed_count=len(seed_ids),
            )
        )

        # 3. Buffer online edges for future promotion
        for src, tgt, weight in online_edges:
            self._interoception.record_online_edge(src, tgt, weight)

        # 4. Compute KG coverage (research metric)
        persistent_count = self._count_persistent_edges(seed_ids)
        total_edges = len(online_edges) + persistent_count
        kg_coverage = persistent_count / max(total_edges, 1)

        emit(
            KGCoverageComputed(
                query_id=query_id,
                persistent_edges=persistent_count,
                online_edges=len(online_edges),
                coverage=kg_coverage,
            )
        )

        # 5. PPR over merged graph (persistent edges in backend + online extras)
        ppr_scores = self._run_ppr(seed_ids, online_edges, domains, query_id)

        # 6. Combined scoring + result assembly
        items = self._score_and_build(
            vec_scores,
            ppr_scores,
            top_k,
            kg_coverage,
        )

        # Cache query for feedback routing
        self._query_cache[query_id] = [item.id for item in items]

        elapsed = (time.perf_counter() - t0) * 1000
        timer = get_overhead_timer()
        overhead = timer.overhead_seconds() if timer else None

        emit(
            QueryCompleted(
                query_id=query_id,
                latency_ms=elapsed,
                seed_count=len(seed_ids),
                result_count=len(items),
                activated_nodes=len(ppr_scores),
                mode="graph",
                timestamp=datetime.now(UTC).isoformat(),
                overhead_seconds=overhead,
            )
        )

        return RetrievalResult(
            items=items,
            query_id=query_id,
            activated_nodes=list(ppr_scores.keys()),
        )

    @traced("retrieval.vec_search")
    def _vec_search(
        self,
        query: str,
        top_k: int,
        min_confidence: float,
        domains: list[str] | None,
        query_id: str,
        t0: float,
    ) -> tuple[list[str] | None, dict[str, float], int]:
        """Embed query, search vectors, filter by domain.

        Returns (seed_ids, vec_scores, vec_results_count).
        Returns (None, {}, 0) if no results survive filtering.
        """
        try:
            query_embedding = self.embedding_model.embed([query])[0]
        except Exception as exc:
            emit(
                QueryFailed(
                    query_id=query_id,
                    error=str(exc),
                    stage="embedding",
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            raise

        fetch_k = max(top_k * 3, 30)  # Over-fetch for graph expansion
        try:
            vec_results = self.vector_index.search(
                query_embedding,
                top_k=fetch_k,
                threshold=min_confidence,
            )
        except Exception as exc:
            emit(
                QueryFailed(
                    query_id=query_id,
                    error=str(exc),
                    stage="vec_search",
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            raise

        if not vec_results:
            return None, {}, 0

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
        emit(
            VecSeedYield(
                query_id=query_id,
                vec_candidates=len(vec_results),
                seeds_after_filter=len(seed_nodes),
                yield_ratio=len(seed_nodes) / max(len(vec_results), 1),
            )
        )

        if not seed_nodes:
            return None, {}, len(vec_results)

        seed_ids = [nid for nid, _ in seed_nodes]
        vec_scores = dict(seed_nodes)

        vec_elapsed = (time.perf_counter() - t0) * 1000
        emit(
            VecSearchCompleted(
                query_id=query_id,
                candidates=len(vec_results),
                fetch_k=fetch_k,
                latency_ms=vec_elapsed,
            )
        )

        return seed_ids, vec_scores, len(vec_results)

    @traced("retrieval.ppr", external=True)
    def _run_ppr(
        self,
        seed_ids: list[str],
        online_edges: list[tuple[str, str, float]],
        domains: list[str] | None,
        query_id: str,
    ) -> dict[str, float]:
        """Run Personalized PageRank over the merged graph."""
        seed_weights = self._interoception.get_seed_weights(seed_ids)

        try:
            return self.backend.personalized_pagerank(
                source_nodes=seed_ids,
                damping_factor=0.85,
                max_iterations=100,
                domain=domains[0] if domains and len(domains) == 1 else None,
                seed_weights=seed_weights,
                extra_edges=online_edges,
                query_id=query_id,
            )
        except Exception as exc:
            emit(
                QueryFailed(
                    query_id=query_id,
                    error=str(exc),
                    stage="ppr",
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            raise

    @traced("retrieval.scoring")
    def _score_and_build(
        self,
        vec_scores: dict[str, float],
        ppr_scores: dict[str, float],
        top_k: int,
        kg_coverage: float,
    ) -> list[RetrievalItem]:
        """Combined scoring: vec_sim x PPR_activation, then build result items.

        PPR returns a probability distribution (sums to ~1) while vec returns
        cosine similarities in [0,1]. We normalize PPR to [0,1] before blending
        so both signals contribute equally at the configured weight.
        """
        # Normalize PPR scores to [0,1] to match vec score scale
        max_ppr = max(ppr_scores.values()) if ppr_scores else 1.0
        if max_ppr > 0:
            ppr_norm = {k: v / max_ppr for k, v in ppr_scores.items()}
        else:
            ppr_norm = ppr_scores

        combined: dict[str, float] = {}
        for node_id in set(list(vec_scores.keys()) + list(ppr_norm.keys())):
            vec_s = vec_scores.get(node_id, 0.0)
            ppr_s = ppr_norm.get(node_id, 0.0)
            combined[node_id] = vec_s * 0.5 + ppr_s * 0.5

        ranked = sorted(combined.items(), key=lambda x: -x[1])

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
        return items

    def feedback(self, query_id: str, outcomes: dict[str, str]) -> None:
        """Update teleportation factors from feedback via interoception.

        Accepted items get boosted, rejected items get penalized.
        This biases future PPR toward nodes that produce good results.
        """
        emit(
            FeedbackReceived(
                query_id=query_id,
                outcomes=len(outcomes),
                accepted=sum(1 for v in outcomes.values() if v == "accepted"),
                rejected=sum(1 for v in outcomes.values() if v == "rejected"),
                partial=sum(1 for v in outcomes.values() if v == "partial"),
                source="adapter",
            )
        )
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

    @traced("retrieval.online_edges")
    def _build_online_edges(
        self,
        seed_ids: list[str],
    ) -> list[tuple[str, str, float]]:
        """Generate ephemeral cosine-sim edges between candidate pairs.

        O(K²) where K = len(seed_ids). Uses numpy batch matmul for the
        heavy lifting (all-pairs cosine via normalized matrix multiply).
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

        ids = list(embeddings.keys())
        matrix = np.array([embeddings[nid] for nid in ids])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normed = matrix / (norms + 1e-9)
        sim_matrix = normed @ normed.T

        edges: list[tuple[str, str, float]] = []
        threshold = self.online_sim_threshold
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if sim_matrix[i, j] >= threshold:
                    edges.append((ids[i], ids[j], float(sim_matrix[i, j])))

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
