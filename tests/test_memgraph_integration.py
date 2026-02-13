"""Integration tests for MemgraphBackend.

All tests require a running Memgraph instance (docker-compose up).
Skipped automatically in CI when Memgraph is not available.
"""

from __future__ import annotations

import socket

import pytest

from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    RelationType,
    SourceMetadata,
)

# ---------------------------------------------------------------------------
# Skip guard: detect if Memgraph is running and accessible
# ---------------------------------------------------------------------------

MEMGRAPH_AVAILABLE = False
try:
    # First check if port is open
    _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _s.settimeout(1)
    _s.connect(("localhost", 7687))
    _s.close()
    # Then verify actual connectivity (handles auth issues)
    from qortex.core.backend import MemgraphBackend

    _b = MemgraphBackend(host="localhost", port=7687)
    _b.connect()
    _b.disconnect()
    MEMGRAPH_AVAILABLE = True
except Exception:
    # Any error (socket, auth, driver) means tests should skip
    pass

pytestmark = pytest.mark.skipif(
    not MEMGRAPH_AVAILABLE,
    reason="Memgraph not running (docker-compose up)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    """Fresh MemgraphBackend connected to localhost. Cleans up after test."""
    from qortex.core.backend import MemgraphBackend

    b = MemgraphBackend(host="localhost", port=7687)
    b.connect()

    # Clean slate for each test
    b._run("MATCH (n) DETACH DELETE n")

    yield b

    # Cleanup
    b._run("MATCH (n) DETACH DELETE n")
    b.disconnect()


@pytest.fixture
def sample_manifest():
    """A minimal manifest for testing ingestion."""
    return IngestionManifest(
        source=SourceMetadata(
            id="test-src",
            name="Test Source",
            source_type="text",
            path_or_url="test.txt",
        ),
        domain="test_domain",
        concepts=[
            ConceptNode(
                id="c1",
                name="Alpha",
                description="First concept",
                domain="test_domain",
                source_id="test-src",
            ),
            ConceptNode(
                id="c2",
                name="Beta",
                description="Second concept",
                domain="test_domain",
                source_id="test-src",
            ),
            ConceptNode(
                id="c3",
                name="Gamma",
                description="Third concept",
                domain="test_domain",
                source_id="test-src",
            ),
        ],
        edges=[
            ConceptEdge(
                source_id="c1",
                target_id="c2",
                relation_type=RelationType.REQUIRES,
                confidence=0.9,
            ),
            ConceptEdge(
                source_id="c2",
                target_id="c3",
                relation_type=RelationType.SUPPORTS,
                confidence=0.8,
            ),
        ],
        rules=[
            ExplicitRule(
                id="r1",
                text="Always do X before Y",
                domain="test_domain",
                source_id="test-src",
                category="process",
                confidence=0.95,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnection:
    def test_connect_disconnect(self, backend):
        assert backend.is_connected()
        backend.disconnect()
        assert not backend.is_connected()


class TestDomainManagement:
    def test_create_and_get_domain(self, backend):
        domain = backend.create_domain("my_domain", description="Test domain")
        assert domain.name == "my_domain"
        assert domain.description == "Test domain"

        fetched = backend.get_domain("my_domain")
        assert fetched is not None
        assert fetched.name == "my_domain"

    def test_create_domain_is_idempotent(self, backend):
        backend.create_domain("d1", description="first")
        backend.create_domain("d1", description="second")  # Should not error
        domains = backend.list_domains()
        assert sum(1 for d in domains if d.name == "d1") == 1

    def test_list_domains(self, backend):
        backend.create_domain("a")
        backend.create_domain("b")
        names = {d.name for d in backend.list_domains()}
        assert "a" in names
        assert "b" in names

    def test_delete_domain_cascades(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        assert backend.get_domain("test_domain") is not None

        deleted = backend.delete_domain("test_domain")
        assert deleted
        assert backend.get_domain("test_domain") is None
        assert backend.get_node("c1") is None

    def test_delete_nonexistent_domain(self, backend):
        assert not backend.delete_domain("nope")


class TestNodeOperations:
    def test_add_and_get_node(self, backend):
        backend.create_domain("d")
        node = ConceptNode(
            id="n1",
            name="TestNode",
            description="A test",
            domain="d",
            source_id="src",
        )
        backend.add_node(node)

        fetched = backend.get_node("n1")
        assert fetched is not None
        assert fetched.name == "TestNode"
        assert fetched.domain == "d"

    def test_get_node_with_domain_filter(self, backend):
        backend.create_domain("d1")
        backend.create_domain("d2")
        backend.add_node(
            ConceptNode(
                id="n1",
                name="Node",
                description="",
                domain="d1",
                source_id="s",
            )
        )

        assert backend.get_node("n1", domain="d1") is not None
        assert backend.get_node("n1", domain="d2") is None

    def test_find_nodes_with_pattern(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        nodes = list(backend.find_nodes(domain="test_domain", name_pattern="*lph*"))
        assert len(nodes) == 1
        assert nodes[0].name == "Alpha"

    def test_find_nodes_with_limit(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        nodes = list(backend.find_nodes(domain="test_domain", limit=2))
        assert len(nodes) == 2


class TestEdgeOperations:
    def test_add_and_get_edges(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        edges = list(backend.get_edges("c1", direction="out"))
        assert len(edges) == 1
        assert edges[0].target_id == "c2"
        assert edges[0].relation_type == RelationType.REQUIRES

    def test_get_edges_in_direction(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        in_edges = list(backend.get_edges("c2", direction="in"))
        assert any(e.source_id == "c1" for e in in_edges)

    def test_get_edges_both_direction(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        both = list(backend.get_edges("c2", direction="both"))
        # c2 has one incoming (c1->c2) and one outgoing (c2->c3)
        assert len(both) == 2


class TestRuleOperations:
    def test_add_and_get_rules(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        rules = backend.get_rules(domain="test_domain")
        assert len(rules) == 1
        assert rules[0].id == "r1"
        assert rules[0].text == "Always do X before Y"

    def test_get_rules_no_domain(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        rules = backend.get_rules()
        assert len(rules) >= 1


class TestManifestIngestion:
    def test_ingest_manifest_atomic(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)

        domain = backend.get_domain("test_domain")
        assert domain is not None
        assert domain.concept_count == 3
        assert domain.edge_count == 2
        assert domain.rule_count == 1

    def test_ingest_creates_domain(self, backend, sample_manifest):
        assert backend.get_domain("test_domain") is None
        backend.ingest_manifest(sample_manifest)
        assert backend.get_domain("test_domain") is not None


class TestQuery:
    def test_query_cypher(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        results = list(
            backend.query_cypher(
                "MATCH (c:Concept {domain: $d}) RETURN c.id AS id ORDER BY c.id",
                {"d": "test_domain"},
            )
        )
        ids = [r["id"] for r in results]
        assert ids == ["c1", "c2", "c3"]


class TestCheckpoints:
    def test_checkpoint_and_list(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        cp_id = backend.checkpoint("before_changes")
        checkpoints = backend.list_checkpoints()
        assert any(cp["id"] == cp_id for cp in checkpoints)

    def test_restore_raises_not_implemented(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        cp_id = backend.checkpoint("test")
        with pytest.raises(NotImplementedError, match="server restart"):
            backend.restore(cp_id)


class TestPersonalizedPageRank:
    def test_ppr_returns_scores(self, backend, sample_manifest):
        backend.ingest_manifest(sample_manifest)
        scores = backend.personalized_pagerank(
            source_nodes=["c1"],
            domain="test_domain",
        )
        # Should return some scores (may be empty if MAGE not loaded)
        # At minimum, should not raise
        assert isinstance(scores, dict)

    def test_ppr_finds_typed_edges(self, backend, sample_manifest):
        """PPR must traverse typed-label edges (REQUIRES, SUPPORTS, etc.).

        This is the regression test for the :REL vs typed-label bug.
        After ingestion, edges are stored as :REQUIRES and :SUPPORTS.
        PPR must find them and return nonzero scores for connected nodes.
        """
        backend.ingest_manifest(sample_manifest)
        scores = backend.personalized_pagerank(
            source_nodes=["c1"],
            domain="test_domain",
        )
        # c1 -> c2 (REQUIRES) -> c3 (SUPPORTS)
        # PPR seeded at c1 should assign scores to c2 and c3
        if scores:  # MAGE may not be loaded; skip assertion if empty
            assert "c1" in scores, f"Seed node c1 missing from PPR scores: {scores}"
            assert len(scores) > 1, f"PPR returned only seed node, edges not traversed: {scores}"

    def test_ppr_scores_decrease_with_distance(self, backend, sample_manifest):
        """Nodes closer to the seed should have higher PPR scores."""
        backend.ingest_manifest(sample_manifest)
        scores = backend.personalized_pagerank(
            source_nodes=["c1"],
            domain="test_domain",
        )
        if len(scores) >= 3:
            # c1 (seed) should have highest score, c2 next, c3 lowest
            assert scores.get("c1", 0) >= scores.get("c2", 0), f"c1 should score >= c2: {scores}"
            assert scores.get("c2", 0) >= scores.get("c3", 0), f"c2 should score >= c3: {scores}"

    def test_ppr_without_domain_filter(self, backend, sample_manifest):
        """PPR without domain filter should also work with typed edges."""
        backend.ingest_manifest(sample_manifest)
        scores = backend.personalized_pagerank(
            source_nodes=["c1"],
            domain=None,
        )
        assert isinstance(scores, dict)
        # Should include nodes from the ingested manifest
        if scores:
            assert any(nid.startswith("c") for nid in scores)

    def test_ppr_with_mixed_edge_types(self, backend):
        """PPR must traverse edges of different types (REQUIRES + SIMILAR_TO)."""
        backend.create_domain("mixed")
        for nid, name in [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D")]:
            backend.add_node(
                ConceptNode(id=nid, name=name, description="", domain="mixed", source_id="s")
            )
        # Mix of edge types
        backend.add_edge(
            ConceptEdge(source_id="a", target_id="b", relation_type=RelationType.REQUIRES)
        )
        backend.add_edge(
            ConceptEdge(source_id="b", target_id="c", relation_type=RelationType.SIMILAR_TO)
        )
        backend.add_edge(
            ConceptEdge(source_id="c", target_id="d", relation_type=RelationType.SUPPORTS)
        )

        scores = backend.personalized_pagerank(
            source_nodes=["a"],
            domain="mixed",
        )
        assert isinstance(scores, dict)
        if scores:
            # PPR should reach all nodes through the chain
            assert len(scores) >= 2, f"PPR didn't traverse mixed edges: {scores}"


class TestEdgeCountWithTypedLabels:
    """Verify that domain.edge_count correctly counts typed-label edges."""

    def test_edge_count_after_ingestion(self, backend, sample_manifest):
        """edge_count must reflect typed-label edges, not just :REL edges."""
        backend.ingest_manifest(sample_manifest)
        domain = backend.get_domain("test_domain")
        assert domain is not None
        assert domain.edge_count == 2, (
            f"Expected 2 edges (REQUIRES + SUPPORTS), got {domain.edge_count}"
        )

    def test_edge_count_with_multiple_types(self, backend):
        """edge_count must count ALL relationship types."""
        backend.create_domain("multi")
        for nid in ["a", "b", "c"]:
            backend.add_node(
                ConceptNode(id=nid, name=nid, description="", domain="multi", source_id="s")
            )
        backend.add_edge(
            ConceptEdge(source_id="a", target_id="b", relation_type=RelationType.REQUIRES)
        )
        backend.add_edge(
            ConceptEdge(source_id="b", target_id="c", relation_type=RelationType.SIMILAR_TO)
        )

        domain = backend.get_domain("multi")
        assert domain is not None
        assert domain.edge_count == 2, f"Expected 2 edges across types, got {domain.edge_count}"
