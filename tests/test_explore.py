"""Tests for graph exploration, rule collection, and rule surfacing.

Tests cover:
- collect_rules_for_concepts (core/rules.py)
- New result types: NodeItem, EdgeItem, RuleItem, ExploreResult, RulesResult
- QueryResult backward compatibility (rules field defaults to [])
- LocalQortexClient.explore() — node, edges, neighbors, rules, depth, missing node
- LocalQortexClient.query() — rules appearing in results
- LocalQortexClient.rules() — projection via FlatRuleSource
"""

from __future__ import annotations

import hashlib

from qortex.client import (
    EdgeItem,
    ExploreResult,
    LocalQortexClient,
    NodeItem,
    QueryResult,
    RuleItem,
    RulesResult,
)
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.core.rules import collect_rules_for_concepts
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMS = 32


class FakeEmbedding:
    """Deterministic hash-based embedding for testing."""

    @property
    def dimensions(self) -> int:
        return DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:DIMS]]
            norm = sum(v * v for v in vec) ** 0.5
            result.append([v / norm for v in vec])
        return result


def make_graph_with_rules():
    """Create a graph with concepts, typed edges, AND explicit rules.

    Graph topology:
        Auth --REQUIRES--> JWT
        Auth --USES------> RBAC
        JWT  --PART_OF---> Auth

    Rules:
        r1: "Always use OAuth2" (linked to Auth)
        r2: "Rotate JWT keys every 90 days" (linked to Auth, JWT)
        r3: "Define roles before permissions" (linked to RBAC)

    Returns (backend, vector_index, embedding, nodes_dict).
    """
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeEmbedding()

    backend.create_domain("security")

    nodes = {
        "Auth": ConceptNode(
            id="security:Auth",
            name="Auth",
            description="Authentication via OAuth2",
            domain="security",
            source_id="test",
        ),
        "JWT": ConceptNode(
            id="security:JWT",
            name="JWT",
            description="JSON Web Tokens",
            domain="security",
            source_id="test",
        ),
        "RBAC": ConceptNode(
            id="security:RBAC",
            name="RBAC",
            description="Role-based access control",
            domain="security",
            source_id="test",
        ),
    }

    for node in nodes.values():
        backend.add_node(node)

    texts = [f"{n.name}: {n.description}" for n in nodes.values()]
    embeddings = embedding.embed(texts)
    for node, emb in zip(nodes.values(), embeddings):
        backend.add_embedding(node.id, emb)

    # Edges
    backend.add_edge(
        ConceptEdge(
            source_id="security:Auth",
            target_id="security:JWT",
            relation_type=RelationType.REQUIRES,
        )
    )
    backend.add_edge(
        ConceptEdge(
            source_id="security:Auth",
            target_id="security:RBAC",
            relation_type=RelationType.USES,
        )
    )
    backend.add_edge(
        ConceptEdge(
            source_id="security:JWT",
            target_id="security:Auth",
            relation_type=RelationType.PART_OF,
        )
    )

    # Rules
    backend.add_rule(
        ExplicitRule(
            id="r1",
            text="Always use OAuth2 for authentication",
            domain="security",
            source_id="test",
            concept_ids=["security:Auth"],
            category="architectural",
        )
    )
    backend.add_rule(
        ExplicitRule(
            id="r2",
            text="Rotate JWT signing keys every 90 days",
            domain="security",
            source_id="test",
            concept_ids=["security:Auth", "security:JWT"],
            category="security",
        )
    )
    backend.add_rule(
        ExplicitRule(
            id="r3",
            text="Define roles before assigning permissions",
            domain="security",
            source_id="test",
            concept_ids=["security:RBAC"],
            category="architectural",
        )
    )

    return backend, vector_index, embedding, nodes


def make_client_with_rules(mode: str = "auto"):
    """Create a LocalQortexClient backed by graph with rules."""
    backend, vector_index, embedding, nodes = make_graph_with_rules()
    client = LocalQortexClient(
        vector_index=vector_index,
        backend=backend,
        embedding_model=embedding,
        mode=mode,
    )
    return client, backend, nodes


# ===========================================================================
# TestRuleCollection — unit tests for core/rules.py
# ===========================================================================


class TestRuleCollection:
    def test_no_concepts_returns_empty(self):
        backend, _, _, _ = make_graph_with_rules()
        result = collect_rules_for_concepts(backend, [])
        assert result == []

    def test_single_concept_returns_linked_rules(self):
        backend, _, _, _ = make_graph_with_rules()
        result = collect_rules_for_concepts(backend, ["security:Auth"])
        ids = [r.id for r in result]
        assert "r1" in ids
        assert "r2" in ids
        # r3 not linked to Auth
        assert "r3" not in ids

    def test_multiple_concepts_union(self):
        backend, _, _, _ = make_graph_with_rules()
        result = collect_rules_for_concepts(backend, ["security:Auth", "security:RBAC"])
        ids = [r.id for r in result]
        assert "r1" in ids
        assert "r2" in ids
        assert "r3" in ids

    def test_deduplication(self):
        """r2 is linked to both Auth and JWT — should appear once."""
        backend, _, _, _ = make_graph_with_rules()
        result = collect_rules_for_concepts(backend, ["security:Auth", "security:JWT"])
        ids = [r.id for r in result]
        assert ids.count("r2") == 1

    def test_relevance_scoring_with_scores(self):
        backend, _, _, _ = make_graph_with_rules()
        scores = {"security:Auth": 0.9, "security:JWT": 0.7}
        result = collect_rules_for_concepts(
            backend, ["security:Auth", "security:JWT"], scores=scores
        )

        r2 = next(r for r in result if r.id == "r2")
        # r2 linked to both Auth(0.9) and JWT(0.7), relevance = max = 0.9
        assert r2.relevance == 0.9

    def test_relevance_zero_without_scores(self):
        backend, _, _, _ = make_graph_with_rules()
        result = collect_rules_for_concepts(backend, ["security:Auth"])
        for r in result:
            assert r.relevance == 0.0

    def test_sorted_by_relevance_descending(self):
        backend, _, _, _ = make_graph_with_rules()
        scores = {"security:Auth": 0.5, "security:JWT": 0.9, "security:RBAC": 0.3}
        result = collect_rules_for_concepts(
            backend,
            ["security:Auth", "security:JWT", "security:RBAC"],
            scores=scores,
        )
        relevances = [r.relevance for r in result]
        assert relevances == sorted(relevances, reverse=True)

    def test_domain_filter(self):
        """Rules only from specified domains."""
        backend, _, _, _ = make_graph_with_rules()
        # Add a rule in a different domain
        backend.create_domain("other")
        backend.add_rule(
            ExplicitRule(
                id="r-other",
                text="Other rule",
                domain="other",
                source_id="test",
                concept_ids=["security:Auth"],  # overlaps concept but wrong domain
            )
        )
        result = collect_rules_for_concepts(backend, ["security:Auth"], domains=["security"])
        ids = [r.id for r in result]
        assert "r-other" not in ids

    def test_no_overlap_returns_empty(self):
        backend, _, _, _ = make_graph_with_rules()
        result = collect_rules_for_concepts(backend, ["nonexistent:id"])
        assert result == []

    def test_derivation_is_explicit(self):
        backend, _, _, _ = make_graph_with_rules()
        result = collect_rules_for_concepts(backend, ["security:Auth"])
        for r in result:
            assert r.derivation == "explicit"


# ===========================================================================
# TestExploreResultTypes — dataclass contracts
# ===========================================================================


class TestExploreResultTypes:
    def test_node_item_fields(self):
        node = NodeItem(id="n1", name="Test", description="Desc", domain="d")
        assert node.id == "n1"
        assert node.confidence == 1.0
        assert node.properties == {}

    def test_edge_item_fields(self):
        edge = EdgeItem(source_id="a", target_id="b", relation_type="requires")
        assert edge.confidence == 1.0
        assert edge.properties == {}

    def test_rule_item_fields(self):
        rule = RuleItem(id="r1", text="Do X", domain="d")
        assert rule.category is None
        assert rule.relevance == 0.0
        assert rule.derivation == "explicit"
        assert rule.source_concepts == []
        assert rule.metadata == {}

    def test_explore_result_defaults(self):
        node = NodeItem(id="n1", name="N", description="D", domain="d")
        result = ExploreResult(node=node)
        assert result.edges == []
        assert result.rules == []
        assert result.neighbors == []

    def test_rules_result_defaults(self):
        result = RulesResult(rules=[])
        assert result.domain_count == 0
        assert result.projection == "rules"

    def test_query_result_backward_compat(self):
        """QueryResult should still work with just items + query_id."""
        result = QueryResult(items=[], query_id="q1")
        assert result.rules == []

    def test_query_result_with_rules(self):
        rule = RuleItem(id="r1", text="Do X", domain="d")
        result = QueryResult(items=[], query_id="q1", rules=[rule])
        assert len(result.rules) == 1


# ===========================================================================
# TestExploreClient — explore() via LocalQortexClient
# ===========================================================================


class TestExploreClient:
    def test_explore_returns_node(self):
        client, _, nodes = make_client_with_rules()
        result = client.explore("security:Auth")
        assert result is not None
        assert result.node.id == "security:Auth"
        assert result.node.name == "Auth"

    def test_explore_returns_edges(self):
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth")
        assert len(result.edges) > 0
        relation_types = {e.relation_type for e in result.edges}
        assert "requires" in relation_types or "uses" in relation_types

    def test_explore_returns_neighbors(self):
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth")
        neighbor_ids = {n.id for n in result.neighbors}
        assert "security:JWT" in neighbor_ids
        assert "security:RBAC" in neighbor_ids

    def test_explore_returns_rules(self):
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth")
        rule_ids = {r.id for r in result.rules}
        assert "r1" in rule_ids
        assert "r2" in rule_ids

    def test_explore_missing_node_returns_none(self):
        client, _, _ = make_client_with_rules()
        result = client.explore("nonexistent:node")
        assert result is None

    def test_explore_depth_1_immediate_neighbors(self):
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth", depth=1)
        neighbor_ids = {n.id for n in result.neighbors}
        assert "security:JWT" in neighbor_ids
        assert "security:RBAC" in neighbor_ids

    def test_explore_depth_clamped_minimum(self):
        """Depth below 1 should be clamped to 1."""
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth", depth=0)
        assert result is not None
        assert len(result.neighbors) > 0

    def test_explore_depth_clamped_maximum(self):
        """Depth above 3 should be clamped to 3."""
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth", depth=10)
        assert result is not None

    def test_explore_depth_2_expands_neighbors(self):
        """Depth 2 should reach neighbors-of-neighbors."""
        client, _, _ = make_client_with_rules()
        # From JWT: JWT -> Auth (already visited), so depth 2 from JWT
        # should pick up Auth's other neighbor RBAC
        result = client.explore("security:JWT", depth=2)
        neighbor_ids = {n.id for n in result.neighbors}
        # JWT -> Auth -> RBAC path
        assert "security:Auth" in neighbor_ids
        assert "security:RBAC" in neighbor_ids

    def test_explore_edge_deduplication(self):
        """Same edge shouldn't appear twice."""
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth")
        edge_keys = [(e.source_id, e.target_id, e.relation_type) for e in result.edges]
        assert len(edge_keys) == len(set(edge_keys))

    def test_explore_node_item_has_properties(self):
        client, backend, _ = make_client_with_rules()
        # Add a node with properties
        backend.add_node(
            ConceptNode(
                id="security:Custom",
                name="Custom",
                description="With props",
                domain="security",
                source_id="test",
                properties={"key": "value"},
            )
        )
        result = client.explore("security:Custom")
        assert result.node.properties == {"key": "value"}

    def test_explore_edge_items_have_relation_type_as_string(self):
        """Edge relation_type should be string, not enum."""
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth")
        for edge in result.edges:
            assert isinstance(edge.relation_type, str)

    def test_explore_includes_rules_from_neighbors(self):
        """Rules linked to neighbor nodes should be included."""
        client, _, _ = make_client_with_rules()
        result = client.explore("security:Auth")
        rule_ids = {r.id for r in result.rules}
        # r3 is linked to RBAC (a neighbor of Auth)
        assert "r3" in rule_ids


# ===========================================================================
# TestRulesInQuery — rules appearing in query() results
# ===========================================================================


class TestRulesInQuery:
    def test_query_returns_rules_field(self):
        client, _, _ = make_client_with_rules()
        result = client.query("Authentication via OAuth2")
        assert hasattr(result, "rules")
        assert isinstance(result.rules, list)

    def test_query_with_linked_rules(self):
        client, _, _ = make_client_with_rules()
        result = client.query("Authentication via OAuth2")
        if result.items:
            # Auth concept should have rules r1 and r2
            rule_ids = {r.id for r in result.rules}
            # At least one rule should be present if Auth matched
            node_ids = {item.node_id for item in result.items}
            if "security:Auth" in node_ids:
                assert "r1" in rule_ids or "r2" in rule_ids

    def test_query_rules_have_relevance_scores(self):
        client, _, _ = make_client_with_rules()
        result = client.query("Authentication via OAuth2")
        for rule in result.rules:
            assert isinstance(rule.relevance, float)

    def test_query_no_concepts_no_rules(self):
        """Empty query results should have empty rules."""
        vector_index = NumpyVectorIndex(dimensions=DIMS)
        backend = InMemoryBackend(vector_index=vector_index)
        backend.connect()
        embedding = FakeEmbedding()
        client = LocalQortexClient(
            vector_index=vector_index,
            backend=backend,
            embedding_model=embedding,
        )
        result = client.query("nothing here")
        assert result.rules == []

    def test_query_rules_are_rule_items(self):
        client, _, _ = make_client_with_rules()
        result = client.query("Authentication via OAuth2")
        for rule in result.rules:
            assert isinstance(rule, RuleItem)


# ===========================================================================
# TestRulesProjection — rules() endpoint
# ===========================================================================


class TestRulesProjection:
    def test_rules_returns_rules_result(self):
        client, _, _ = make_client_with_rules()
        result = client.rules()
        assert isinstance(result, RulesResult)
        assert result.projection == "rules"

    def test_rules_returns_explicit_rules(self):
        client, _, _ = make_client_with_rules()
        result = client.rules()
        explicit = [r for r in result.rules if r.derivation == "explicit"]
        assert len(explicit) >= 3  # r1, r2, r3

    def test_rules_includes_derived(self):
        """With include_derived=True, should also get edge-derived rules."""
        client, _, _ = make_client_with_rules()
        result = client.rules(include_derived=True)
        derivations = {r.derivation for r in result.rules}
        # Should have both explicit and derived
        assert "explicit" in derivations

    def test_rules_without_derived(self):
        client, _, _ = make_client_with_rules()
        result = client.rules(include_derived=False)
        for r in result.rules:
            assert r.derivation == "explicit"

    def test_rules_domain_filter(self):
        client, _, _ = make_client_with_rules()
        result = client.rules(domains=["security"])
        for r in result.rules:
            assert r.domain == "security"

    def test_rules_concept_ids_filter(self):
        client, _, _ = make_client_with_rules()
        result = client.rules(concept_ids=["security:RBAC"])
        # Only r3 is linked to RBAC alone
        for r in result.rules:
            assert "security:RBAC" in r.source_concepts

    def test_rules_category_filter(self):
        client, _, _ = make_client_with_rules()
        result = client.rules(categories=["architectural"])
        for r in result.rules:
            assert r.category == "architectural"

    def test_rules_domain_count(self):
        client, _, _ = make_client_with_rules()
        result = client.rules()
        assert result.domain_count >= 1

    def test_rules_empty_graph(self):
        vector_index = NumpyVectorIndex(dimensions=DIMS)
        backend = InMemoryBackend(vector_index=vector_index)
        backend.connect()
        embedding = FakeEmbedding()
        client = LocalQortexClient(
            vector_index=vector_index,
            backend=backend,
            embedding_model=embedding,
        )
        result = client.rules()
        assert result.rules == []
        assert result.domain_count == 0

    def test_rules_min_confidence(self):
        client, backend, _ = make_client_with_rules()
        # Add a low-confidence rule
        backend.add_rule(
            ExplicitRule(
                id="r-low",
                text="Maybe do this",
                domain="security",
                source_id="test",
                concept_ids=["security:Auth"],
                confidence=0.3,
            )
        )
        result = client.rules(min_confidence=0.5)
        ids = [r.id for r in result.rules]
        assert "r-low" not in ids
