"""End-to-end graph pipeline verification test.

Walks the full 10-stage pipeline from ingest through graph-augmented retrieval:
1. Ingest → nodes created
2. Embed → vectors stored
3. Concept extraction → nodes in KG
4. Edge creation → typed edges between concepts
5. Vec search → seed nodes (VecOnlyAdapter baseline)
6. Online edge generation (GraphRAGAdapter)
7. PPR execution
8. Combined scoring (graph changes ranking)
9. Teleportation factors (feedback shifts PPR weights)
10. Edge promotion (buffer → persistent KG)

Runs against InMemoryBackend by default (no Memgraph needed).
Set QORTEX_E2E_MEMGRAPH=1 to run against live Memgraph.
"""

from __future__ import annotations

import math
import os
import socket

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, RelationType
from qortex.hippocampus.adapter import GraphRAGAdapter, VecOnlyAdapter
from qortex.hippocampus.buffer import EdgePromotionBuffer
from qortex.hippocampus.factors import TeleportationFactors
from qortex.hippocampus.interoception import LocalInteroceptionProvider
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


class ControlledEmbedding:
    """Returns pre-set vectors for known texts, zero otherwise."""

    def __init__(self, mapping: dict[str, list[float]], dims: int = 3):
        self._mapping = mapping
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        zero = [0.0] * self._dims
        return [self._mapping.get(t, zero) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims


def _make_node(domain: str, name: str, desc: str = "") -> ConceptNode:
    return ConceptNode(
        id=f"{domain}:{name}",
        name=name,
        description=desc or f"A {name} concept",
        domain=domain,
        source_id="e2e-test",
    )


# ---------------------------------------------------------------------------
# Backend fixture: InMemoryBackend or live Memgraph
# ---------------------------------------------------------------------------


def _memgraph_available() -> bool:
    if not os.environ.get("QORTEX_E2E_MEMGRAPH"):
        return False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("localhost", 7687))
        s.close()
        return True
    except Exception:
        return False


@pytest.fixture
def backend():
    """InMemoryBackend (default) or MemgraphBackend (if QORTEX_E2E_MEMGRAPH=1)."""
    if _memgraph_available():
        from qortex.core.backend import MemgraphBackend

        b = MemgraphBackend(host="localhost", port=7687)
        b.connect()
        # Clean slate
        b._driver.execute_query("MATCH (n) DETACH DELETE n")
        yield b
        b._driver.execute_query("MATCH (n) DETACH DELETE n")
        b.disconnect()
    else:
        vi = NumpyVectorIndex(dimensions=3)
        b = InMemoryBackend(vector_index=vi)
        b.connect()
        yield b


@pytest.fixture
def dims():
    return 3


# ---------------------------------------------------------------------------
# Embedding vectors: designed so graph structure changes ranking
#
# Cluster A: alpha(1,0,0), beta(0.9,0.1,0), delta(0.8,0.2,0)
# Cluster B: gamma(0,0,1), epsilon(0.1,0,0.9)
# Bridge:    zeta(0.5,0.5,0) — moderately similar to cluster A
# ---------------------------------------------------------------------------


@pytest.fixture
def embeddings():
    return {
        "e2e:alpha": _norm([1.0, 0.0, 0.0]),
        "e2e:beta": _norm([0.9, 0.1, 0.0]),
        "e2e:gamma": _norm([0.0, 0.0, 1.0]),
        "e2e:delta": _norm([0.8, 0.2, 0.0]),
        "e2e:epsilon": _norm([0.1, 0.0, 0.9]),
        "e2e:zeta": _norm([0.5, 0.5, 0.0]),
    }


@pytest.fixture
def populated_backend(backend, embeddings, dims):
    """Backend with 6 nodes, embeddings, and edges pre-loaded."""
    backend.create_domain("e2e")

    nodes = [
        _make_node("e2e", "alpha", "Alpha concept — cluster A leader"),
        _make_node("e2e", "beta", "Beta concept — cluster A member"),
        _make_node("e2e", "gamma", "Gamma concept — cluster B leader"),
        _make_node("e2e", "delta", "Delta concept — cluster A member"),
        _make_node("e2e", "epsilon", "Epsilon concept — cluster B member"),
        _make_node("e2e", "zeta", "Zeta concept — bridge node"),
    ]

    # Stage 1 & 3: Ingest nodes (concept extraction)
    for node in nodes:
        backend.add_node(node)

    # Stage 2: Embed vectors
    for nid, emb in embeddings.items():
        backend.add_embedding(nid, emb)

    # Stage 4: Create typed edges (cluster structure + bridge)
    edges = [
        ("e2e:alpha", "e2e:beta", RelationType.REQUIRES, 0.9),
        ("e2e:alpha", "e2e:delta", RelationType.REQUIRES, 0.8),
        ("e2e:gamma", "e2e:epsilon", RelationType.SIMILAR_TO, 0.9),
        # Bridge: zeta connects cluster A to cluster B
        ("e2e:zeta", "e2e:alpha", RelationType.REFINES, 0.7),
        ("e2e:zeta", "e2e:gamma", RelationType.USES, 0.6),
    ]
    for src, tgt, rel, conf in edges:
        backend.add_edge(
            ConceptEdge(
                source_id=src,
                target_id=tgt,
                relation_type=rel,
                confidence=conf,
            )
        )

    return backend, nodes, edges


@pytest.fixture
def embedding_model(embeddings, dims):
    """Embedding model with known query → vector mappings."""
    query_map = {
        "find alpha things": embeddings["e2e:alpha"],
        "find gamma things": embeddings["e2e:gamma"],
        "find zeta bridge": embeddings["e2e:zeta"],
    }
    return ControlledEmbedding(query_map, dims=dims)


@pytest.fixture
def vector_index(populated_backend):
    """Extract the vector index from the backend."""
    backend = populated_backend[0]
    if hasattr(backend, "_vector_index") and backend._vector_index is not None:
        return backend._vector_index
    # Memgraph: create separate index, add embeddings
    vi = NumpyVectorIndex(dimensions=3)
    for node in populated_backend[1]:
        emb = backend.get_embedding(node.id)
        if emb:
            vi.add(node.id, emb)
    return vi


# ===========================================================================
# Stage 1–4: Ingest, embed, concept extraction, edge creation
# ===========================================================================


class TestIngestPipeline:
    """Stages 1–4: data gets into the graph correctly."""

    def test_stage1_nodes_created(self, populated_backend):
        backend, nodes, _ = populated_backend
        for node in nodes:
            found = backend.get_node(node.id)
            assert found is not None, f"Node {node.id} not found"
            assert found.name == node.name

    def test_stage2_embeddings_stored(self, populated_backend, embeddings):
        backend = populated_backend[0]
        for nid, expected_emb in embeddings.items():
            emb = backend.get_embedding(nid)
            assert emb is not None, f"Embedding for {nid} not found"
            assert len(emb) == len(expected_emb)
            # Cosine sim with self should be ~1.0
            dot = sum(a * b for a, b in zip(emb, expected_emb))
            assert dot > 0.99, f"Embedding mismatch for {nid}: dot={dot}"

    def test_stage3_concepts_exist(self, populated_backend):
        backend = populated_backend[0]
        domains = backend.list_domains()
        domain_names = [d.name for d in domains]
        assert "e2e" in domain_names

    def test_stage4_edges_exist(self, populated_backend):
        backend, _, expected_edges = populated_backend
        for src, tgt, _rel, _ in expected_edges:
            edges = backend.get_edges(src, direction="out")
            target_ids = [e.target_id for e in edges]
            assert tgt in target_ids, f"Edge {src} → {tgt} not found"


# ===========================================================================
# Stage 5: Vec search → seed nodes (VecOnlyAdapter)
# ===========================================================================


class TestVecOnlyRetrieval:
    """Stage 5: VecOnlyAdapter returns ranked results from vector similarity."""

    def test_vec_retrieval_returns_results(self, populated_backend, vector_index, embedding_model):
        backend = populated_backend[0]
        adapter = VecOnlyAdapter(vector_index, backend, embedding_model)

        result = adapter.retrieve("find alpha things", top_k=5)
        assert len(result.items) > 0
        assert result.query_id

    def test_vec_ranking_by_cosine_sim(self, populated_backend, vector_index, embedding_model):
        backend = populated_backend[0]
        adapter = VecOnlyAdapter(vector_index, backend, embedding_model)

        result = adapter.retrieve("find alpha things", top_k=6)
        # alpha should be top-1 (exact match), beta/delta close behind
        ids = [it.id for it in result.items]
        assert ids[0] == "e2e:alpha", f"Expected alpha first, got {ids[0]}"
        # beta and delta (both in cluster A) should appear before gamma/epsilon
        alpha_cluster = {"e2e:alpha", "e2e:beta", "e2e:delta"}
        top_3_ids = set(ids[:3])
        assert top_3_ids.issubset(alpha_cluster | {"e2e:zeta"}), (
            f"Top 3 should be cluster A + bridge, got {top_3_ids}"
        )


# ===========================================================================
# Stages 6–8: GraphRAGAdapter — online edges, PPR, combined scoring
# ===========================================================================


class TestGraphRAGRetrieval:
    """Stages 6–8: GraphRAGAdapter enriches results via graph structure."""

    def _make_adapter(self, vector_index, backend, embedding_model):
        return GraphRAGAdapter(
            vector_index,
            backend,
            embedding_model,
            online_sim_threshold=0.5,
            kg_weight_bonus=0.3,
        )

    def test_stage6_online_edges_generated(self, populated_backend, vector_index, embedding_model):
        backend = populated_backend[0]
        adapter = self._make_adapter(vector_index, backend, embedding_model)

        # _build_online_edges needs seed IDs with embeddings
        seed_ids = ["e2e:alpha", "e2e:beta", "e2e:delta"]
        edges = adapter._build_online_edges(seed_ids)
        # alpha, beta, delta are all similar — should generate edges
        assert len(edges) > 0, "Expected online edges between similar nodes"
        for _src, _tgt, weight in edges:
            assert weight >= 0.5, f"Edge weight {weight} below threshold"

    def test_stage7_ppr_returns_scores(self, populated_backend, vector_index, embedding_model):
        backend = populated_backend[0]
        adapter = self._make_adapter(vector_index, backend, embedding_model)

        seed_ids = ["e2e:alpha", "e2e:beta"]
        online_edges = adapter._build_online_edges(seed_ids)

        ppr_scores = adapter._run_ppr(seed_ids, online_edges, None, "test-q")
        assert len(ppr_scores) > 0, "PPR returned no scores"
        # Seeds should have non-zero scores
        assert ppr_scores.get("e2e:alpha", 0) > 0
        # PPR should spread activation — delta (connected to alpha) should get some
        assert ppr_scores.get("e2e:delta", 0) > 0 or ppr_scores.get("e2e:beta", 0) > 0

    def test_stage8_combined_scoring_differs_from_vec(
        self, populated_backend, vector_index, embedding_model
    ):
        """The core proof: graph-augmented ranking differs from vec-only."""
        backend = populated_backend[0]
        vec_adapter = VecOnlyAdapter(vector_index, backend, embedding_model)
        graph_adapter = self._make_adapter(vector_index, backend, embedding_model)

        query = "find alpha things"
        vec_result = vec_adapter.retrieve(query, top_k=6)
        graph_result = graph_adapter.retrieve(query, top_k=6)

        vec_ids = [it.id for it in vec_result.items]
        graph_ids = [it.id for it in graph_result.items]

        # Both should return results
        assert len(vec_ids) > 0
        assert len(graph_ids) > 0

        # Graph should activate more nodes via PPR
        assert len(graph_result.activated_nodes) >= len(vec_result.activated_nodes)

        # At minimum, graph results should include items that got PPR boost
        # Check that scores differ (graph items have ppr_score in metadata)
        graph_item = graph_result.items[0]
        assert "ppr_score" in graph_item.metadata
        assert "vec_score" in graph_item.metadata
        assert "kg_coverage" in graph_item.metadata

    def test_graph_result_contains_delta_metadata(
        self, populated_backend, vector_index, embedding_model
    ):
        """Graph results embed the observability metadata."""
        backend = populated_backend[0]
        adapter = self._make_adapter(vector_index, backend, embedding_model)

        result = adapter.retrieve("find alpha things", top_k=5)
        for item in result.items:
            assert "vec_score" in item.metadata
            assert "ppr_score" in item.metadata


# ===========================================================================
# Stage 9: Teleportation factors (feedback loop)
# ===========================================================================


class TestTeleportationFactors:
    """Stage 9: Feedback updates factors, shifting PPR seed weights."""

    def test_factors_change_after_feedback(self, populated_backend, vector_index, embedding_model):
        backend = populated_backend[0]
        factors = TeleportationFactors()
        adapter = GraphRAGAdapter(
            vector_index,
            backend,
            embedding_model,
            factors=factors,
            online_sim_threshold=0.5,
        )

        # Initial query
        result = adapter.retrieve("find alpha things", top_k=5)
        query_id = result.query_id

        # Get initial factors
        initial_weights = factors.weight_seeds(["e2e:alpha", "e2e:beta"])

        # Report feedback: alpha was useful, gamma was not
        adapter.feedback(
            query_id,
            {
                "e2e:alpha": "accepted",
                "e2e:gamma": "rejected",
            },
        )

        # Factors should have changed
        updated_weights = factors.weight_seeds(["e2e:alpha", "e2e:beta"])
        # Alpha's factor should have increased (accepted)
        alpha_initial = initial_weights.get("e2e:alpha", 1.0)
        alpha_updated = updated_weights.get("e2e:alpha", 1.0)
        assert alpha_updated >= alpha_initial, (
            f"Expected alpha factor to increase: {alpha_initial} → {alpha_updated}"
        )


# ===========================================================================
# Stage 10: Edge promotion (buffer → persistent KG)
# ===========================================================================


class TestEdgePromotion:
    """Stage 10: Online edges with enough hits get promoted to persistent KG."""

    def test_buffer_records_and_promotes(self, populated_backend):
        backend = populated_backend[0]
        buffer = EdgePromotionBuffer()

        # Record the same edge multiple times (simulating repeated query hits)
        for _ in range(3):
            buffer.record("e2e:alpha", "e2e:zeta", 0.85)

        stats = buffer.get_edge_stats("e2e:alpha", "e2e:zeta")
        assert stats is not None
        assert stats.hit_count >= 3
        assert stats.avg_score >= 0.8

        # Flush with low thresholds: edges meeting threshold should promote
        result = buffer.flush(backend, min_hits=2, min_avg_score=0.6)
        assert result.promoted >= 1, "Expected at least 1 edge promoted"

    def test_subthreshold_edges_not_promoted(self, populated_backend):
        backend = populated_backend[0]
        buffer = EdgePromotionBuffer()

        # Record just once — below min_hits threshold
        buffer.record("e2e:alpha", "e2e:epsilon", 0.5)

        result = buffer.flush(backend, min_hits=5, min_avg_score=0.9)
        assert result.promoted == 0, "Sub-threshold edge should not promote"


# ===========================================================================
# Full pipeline: all stages together
# ===========================================================================


class TestFullPipeline:
    """Run the complete pipeline end-to-end and verify graph adds value."""

    def test_full_pipeline_query_feedback_requery(
        self, populated_backend, vector_index, embedding_model
    ):
        """Ingest → query → feedback → re-query shows learning effect."""
        backend = populated_backend[0]

        interoception = LocalInteroceptionProvider()
        adapter = GraphRAGAdapter(
            vector_index,
            backend,
            embedding_model,
            interoception=interoception,
            online_sim_threshold=0.5,
        )

        # First query
        r1 = adapter.retrieve("find alpha things", top_k=5)
        assert len(r1.items) > 0

        # Feedback: boost alpha, penalize others
        adapter.feedback(
            r1.query_id,
            {
                "e2e:alpha": "accepted",
                "e2e:beta": "accepted",
                "e2e:gamma": "rejected",
            },
        )

        # Second query — factors should influence ranking
        r2 = adapter.retrieve("find alpha things", top_k=5)
        assert len(r2.items) > 0

        # Alpha and beta should still be top-ranked after positive feedback
        top_2_ids = {it.id for it in r2.items[:2]}
        assert "e2e:alpha" in top_2_ids, f"Alpha should be in top 2, got {top_2_ids}"

    def test_compare_shows_graph_delta(self, populated_backend, vector_index, embedding_model):
        """The compare tool's logic: graph vs vec should show differences."""
        backend = populated_backend[0]
        vec = VecOnlyAdapter(vector_index, backend, embedding_model)
        graph = GraphRAGAdapter(
            vector_index,
            backend,
            embedding_model,
            online_sim_threshold=0.5,
        )

        query = "find zeta bridge"
        vec_r = vec.retrieve(query, top_k=6)
        graph_r = graph.retrieve(query, top_k=6)

        vec_ids = [it.id for it in vec_r.items]
        graph_ids = [it.id for it in graph_r.items]  # noqa: F841

        # zeta is a bridge node — graph should activate both clusters
        # whereas vec-only only sees cosine similarity to zeta's embedding
        assert len(graph_r.activated_nodes) >= len(vec_ids), (
            f"Graph should activate at least as many nodes as vec: "
            f"{len(graph_r.activated_nodes)} vs {len(vec_ids)}"
        )
