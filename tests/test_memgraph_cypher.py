"""Unit tests for MemgraphBackend Cypher query construction.

These tests DO NOT require a running Memgraph instance. They mock _run()
to capture the generated Cypher and verify correctness, especially around
relationship pattern matching (the :REL vs typed-label bug).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qortex.core.models import (
    ConceptEdge,
    RelationType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend():
    """Create an unconnected MemgraphBackend with _run mocked."""
    from qortex.core.backend import MemgraphBackend

    b = MemgraphBackend(host="localhost", port=7687)
    b._driver = MagicMock()  # Pretend we're connected
    return b


def _capture_cypher(backend, method_name, *args, **kwargs):
    """Call a backend method and return all Cypher strings passed to _run."""
    captured: list[str] = []

    def spy(cypher, params=None):
        captured.append(cypher)
        # PPR needs nodes from the first query so the method reaches the edge fetch.
        if "RETURN n.id AS id" in cypher:
            return [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}]
        # Edge fetch for PPR power iteration
        if "RETURN a.id AS src" in cypher:
            return [{"src": "c1", "tgt": "c2", "weight": 1.0}]
        return []

    with patch.object(backend, "_run", side_effect=spy):
        try:
            result = getattr(backend, method_name)(*args, **kwargs)
            # Exhaust iterators (get_edges returns Iterator)
            if hasattr(result, "__iter__") and not isinstance(result, (dict, str)):
                list(result)
        except Exception:
            pass  # Some methods may fail; we only care about the Cypher

    return captured


# ---------------------------------------------------------------------------
# PPR Cypher Tests
# ---------------------------------------------------------------------------


class TestPPRCypherConstruction:
    """Verify that personalized_pagerank generates correct Cypher patterns.

    The implementation uses power iteration in Python, issuing two queries:
      1. Node fetch: MATCH (n:Concept) RETURN n.id AS id
      2. Edge fetch: MATCH (a:Concept)-[r]->(b:Concept) RETURN a.id AS src, ...
    """

    def test_ppr_no_domain_uses_wildcard_relationship(self):
        """PPR without domain filter must use wildcard relationship in edge fetch."""
        b = _make_backend()
        cyphers = _capture_cypher(
            b,
            "personalized_pagerank",
            source_nodes=["c1"],
            domain=None,
        )
        assert len(cyphers) == 2, f"Expected 2 queries (nodes + edges), got {len(cyphers)}"

        for cypher in cyphers:
            assert ":REL" not in cypher, f"PPR Cypher still contains :REL: {cypher}"
            assert ":Concept" in cypher

        # Edge fetch must use wildcard -[r]->
        edge_cypher = cyphers[1]
        assert "-[r]->" in edge_cypher, f"Edge fetch does not use wildcard -[r]->: {edge_cypher}"

    def test_ppr_with_domain_uses_wildcard_relationship(self):
        """PPR with domain filter must use wildcard relationships + domain param."""
        b = _make_backend()
        cyphers = _capture_cypher(
            b,
            "personalized_pagerank",
            source_nodes=["c1"],
            domain="test_domain",
        )
        assert len(cyphers) == 2

        for cypher in cyphers:
            assert ":REL" not in cypher, f"PPR Cypher still contains :REL: {cypher}"
            assert "$d" in cypher, f"Domain param missing: {cypher}"

        edge_cypher = cyphers[1]
        assert "-[r]->" in edge_cypher

    def test_ppr_uses_power_iteration_not_mage(self):
        """PPR must NOT call MAGE pagerank.get(); it uses Python power iteration."""
        b = _make_backend()
        cyphers = _capture_cypher(
            b,
            "personalized_pagerank",
            source_nodes=["c1"],
        )
        for cypher in cyphers:
            assert "pagerank" not in cypher.lower(), (
                f"PPR should use power iteration, not MAGE: {cypher}"
            )

    def test_ppr_returns_scores_for_valid_seeds(self):
        """PPR with valid seeds must return non-empty scores."""
        b = _make_backend()
        scores = {}
        with patch.object(b, "_run") as mock_run:
            mock_run.side_effect = [
                [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],  # node fetch
                [{"src": "c1", "tgt": "c2", "weight": 1.0}],  # edge fetch
            ]
            scores = b.personalized_pagerank(source_nodes=["c1"], damping_factor=0.85)
        assert len(scores) > 0, "PPR should return scores for valid seeds"
        assert "c1" in scores

    def test_ppr_respects_damping_factor(self):
        """Different damping factors must produce different score distributions."""
        b = _make_backend()
        node_data = [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}]
        edge_data = [
            {"src": "c1", "tgt": "c2", "weight": 1.0},
            {"src": "c2", "tgt": "c3", "weight": 1.0},
        ]

        with patch.object(b, "_run") as mock_run:
            mock_run.side_effect = [node_data, edge_data]
            scores_low = b.personalized_pagerank(source_nodes=["c1"], damping_factor=0.5)

        with patch.object(b, "_run") as mock_run:
            mock_run.side_effect = [node_data, edge_data]
            scores_high = b.personalized_pagerank(source_nodes=["c1"], damping_factor=0.99)

        # With low damping, seed node keeps more weight (more teleportation)
        assert scores_low["c1"] > scores_high["c1"], (
            "Lower damping should give seed node higher relative score"
        )


# ---------------------------------------------------------------------------
# Edge Count Cypher Tests
# ---------------------------------------------------------------------------


class TestEdgeCountCypher:
    """Verify that domain edge counting uses wildcard relationship pattern."""

    def test_edge_count_uses_wildcard_relationship(self):
        """Domain stats edge count must match ANY relationship type, not :REL."""
        b = _make_backend()

        # _record_to_domain calls _count which calls _run
        # We need to simulate get_domain which calls _record_to_domain
        captured: list[str] = []

        def spy(cypher, params=None):
            captured.append(cypher)
            # Return a domain record for the first call, then counts
            if "MATCH (d:Domain" in cypher and "RETURN" in cypher and "count" not in cypher.lower():
                return [
                    {"name": "test", "description": "test", "created_at": None, "updated_at": None}
                ]
            return [{"cnt": 0}]

        with patch.object(b, "_run", side_effect=spy):
            b.get_domain("test")

        # Find the edge count query (matches relationship r between Concept nodes)
        edge_queries = [c for c in captured if "count(r)" in c and "-[r]-" in c]
        assert len(edge_queries) == 1, (
            f"Expected 1 edge count query, got {len(edge_queries)}: {captured}"
        )
        edge_cypher = edge_queries[0]

        # Must NOT contain :REL
        assert ":REL" not in edge_cypher, f"Edge count still uses :REL: {edge_cypher}"

        # Must use -[r]- (wildcard)
        assert "-[r]-" in edge_cypher, f"Edge count does not use wildcard: {edge_cypher}"


# ---------------------------------------------------------------------------
# Edge creation verifies typed labels
# ---------------------------------------------------------------------------


class TestEdgeCreationUsesTypedLabels:
    """Verify that add_edge stores edges with typed relationship labels."""

    @pytest.mark.parametrize(
        "rel_type,expected_label",
        [
            (RelationType.REQUIRES, "REQUIRES"),
            (RelationType.SUPPORTS, "SUPPORTS"),
            (RelationType.SIMILAR_TO, "SIMILAR_TO"),
            (RelationType.CONTRADICTS, "CONTRADICTS"),
            (RelationType.PART_OF, "PART_OF"),
            (RelationType.USES, "USES"),
            (RelationType.REFINES, "REFINES"),
            (RelationType.IMPLEMENTS, "IMPLEMENTS"),
            (RelationType.ALTERNATIVE_TO, "ALTERNATIVE_TO"),
            (RelationType.CHALLENGES, "CHALLENGES"),
            (RelationType.BELONGS_TO, "BELONGS_TO"),
            (RelationType.INSTANCE_OF, "INSTANCE_OF"),
            (RelationType.CONTAINS, "CONTAINS"),
        ],
    )
    def test_add_edge_uses_typed_label(self, rel_type, expected_label):
        """add_edge must CREATE with the typed label, not :REL."""
        b = _make_backend()
        edge = ConceptEdge(
            source_id="a",
            target_id="b",
            relation_type=rel_type,
            confidence=0.9,
        )
        cyphers = _capture_cypher(b, "add_edge", edge)
        assert len(cyphers) == 1
        cypher = cyphers[0]

        assert f":{expected_label}" in cypher, (
            f"Expected :{expected_label} in CREATE, got: {cypher}"
        )
        # Must NOT use generic :REL
        assert ":REL " not in cypher


# ---------------------------------------------------------------------------
# get_edges pattern tests
# ---------------------------------------------------------------------------


class TestGetEdgesCypher:
    """Verify get_edges generates correct Cypher for different directions + types."""

    def test_get_edges_no_filter_uses_wildcard(self):
        """get_edges with no relation_type must use wildcard pattern."""
        b = _make_backend()
        cyphers = _capture_cypher(b, "get_edges", "c1", "both", None)
        assert len(cyphers) == 1
        cypher = cyphers[0]

        # No hardcoded :REL
        assert ":REL" not in cypher

        # Uses wildcard: -[r]- (no label after r)
        assert "[r]" in cypher or "[r " in cypher

    def test_get_edges_with_type_filter(self):
        """get_edges with relation_type must filter by that specific label."""
        b = _make_backend()
        cyphers = _capture_cypher(b, "get_edges", "c1", "out", "requires")
        assert len(cyphers) == 1
        cypher = cyphers[0]

        assert ":REQUIRES" in cypher
        assert ":REL" not in cypher

    def test_get_edges_out_direction(self):
        """get_edges with 'out' direction uses outgoing arrow."""
        b = _make_backend()
        cyphers = _capture_cypher(b, "get_edges", "c1", "out", None)
        cypher = cyphers[0]
        assert "]->" in cypher

    def test_get_edges_in_direction(self):
        """get_edges with 'in' direction uses incoming arrow."""
        b = _make_backend()
        cyphers = _capture_cypher(b, "get_edges", "c1", "in", None)
        cypher = cyphers[0]
        assert "]->" in cypher  # Pattern is (t)-[r]->(s {id: $nid})


# ---------------------------------------------------------------------------
# Regression: no :REL anywhere in generated Cypher
# ---------------------------------------------------------------------------


class TestNoRELRegression:
    """Exhaustive scan: no method should generate Cypher containing ':REL'."""

    def test_ppr_no_rel(self):
        b = _make_backend()
        for domain in [None, "test"]:
            cyphers = _capture_cypher(
                b, "personalized_pagerank", source_nodes=["c1"], domain=domain
            )
            for c in cyphers:
                assert ":REL" not in c, f":REL found in PPR Cypher (domain={domain}): {c}"

    def test_get_domain_no_rel(self):
        """get_domain -> _record_to_domain -> edge count must not use :REL."""
        b = _make_backend()
        captured: list[str] = []

        def spy(cypher, params=None):
            captured.append(cypher)
            if "count" in cypher.lower():
                return [{"cnt": 0}]
            return [{"name": "d", "description": None, "created_at": None, "updated_at": None}]

        with patch.object(b, "_run", side_effect=spy):
            b.get_domain("d")

        for c in captured:
            assert ":REL" not in c, f":REL found in get_domain Cypher: {c}"

    def test_add_edge_no_rel(self):
        b = _make_backend()
        edge = ConceptEdge(
            source_id="a",
            target_id="b",
            relation_type=RelationType.REQUIRES,
            confidence=0.9,
        )
        cyphers = _capture_cypher(b, "add_edge", edge)
        for c in cyphers:
            assert ":REL " not in c, f":REL found in add_edge Cypher: {c}"

    def test_get_edges_no_rel(self):
        b = _make_backend()
        for direction in ["in", "out", "both"]:
            cyphers = _capture_cypher(b, "get_edges", "c1", direction, None)
            for c in cyphers:
                assert ":REL" not in c, f":REL found in get_edges Cypher (dir={direction}): {c}"
