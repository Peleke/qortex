"""Tests for MCP tools: qortex_explore, qortex_rules, and rules in qortex_query.

Tests the _impl functions directly via create_server() with InMemoryBackend.
No MCP transport needed — we call the plain Python impl functions.
"""

from __future__ import annotations

import hashlib

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.mcp import server as mcp_server
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_server_state():
    """Reset server globals between tests."""
    mcp_server._backend = None
    mcp_server._vector_index = None
    mcp_server._adapter = None
    mcp_server._graph_adapter = None
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None
    mcp_server._interoception = None
    yield
    mcp_server._backend = None
    mcp_server._vector_index = None
    mcp_server._adapter = None
    mcp_server._graph_adapter = None
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None
    mcp_server._interoception = None


@pytest.fixture
def vector_index():
    return NumpyVectorIndex(dimensions=DIMS)


@pytest.fixture
def backend(vector_index) -> InMemoryBackend:
    b = InMemoryBackend(vector_index=vector_index)
    b.connect()
    return b


@pytest.fixture
def embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def configured_server(backend, embedding, vector_index):
    mcp_server.create_server(
        backend=backend,
        embedding_model=embedding,
        vector_index=vector_index,
    )
    return mcp_server


def _setup_graph_with_rules(backend, embedding):
    """Populate backend with test graph + rules."""
    backend.create_domain("security")

    nodes = [
        ConceptNode(
            id="security:Auth",
            name="Auth",
            description="Authentication via OAuth2",
            domain="security",
            source_id="test",
        ),
        ConceptNode(
            id="security:JWT",
            name="JWT",
            description="JSON Web Tokens",
            domain="security",
            source_id="test",
        ),
        ConceptNode(
            id="security:RBAC",
            name="RBAC",
            description="Role-based access control",
            domain="security",
            source_id="test",
        ),
    ]

    for node in nodes:
        backend.add_node(node)

    texts = [f"{n.name}: {n.description}" for n in nodes]
    embeddings = embedding.embed(texts)
    for node, emb in zip(nodes, embeddings):
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

    # Rules
    backend.add_rule(
        ExplicitRule(
            id="r1",
            text="Always use OAuth2",
            domain="security",
            source_id="test",
            concept_ids=["security:Auth"],
            category="architectural",
        )
    )
    backend.add_rule(
        ExplicitRule(
            id="r2",
            text="Rotate JWT keys every 90 days",
            domain="security",
            source_id="test",
            concept_ids=["security:Auth", "security:JWT"],
            category="security",
        )
    )
    backend.add_rule(
        ExplicitRule(
            id="r3",
            text="Define roles before permissions",
            domain="security",
            source_id="test",
            concept_ids=["security:RBAC"],
            category="architectural",
        )
    )

    return nodes


# ===========================================================================
# TestQortexExploreMCP
# ===========================================================================


class TestQortexExploreMCP:
    def test_explore_returns_node(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth")
        assert result is not None
        assert result["node"]["id"] == "security:Auth"
        assert result["node"]["name"] == "Auth"

    def test_explore_returns_edges(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth")
        assert len(result["edges"]) > 0
        edge_types = {e["relation_type"] for e in result["edges"]}
        assert "requires" in edge_types or "uses" in edge_types

    def test_explore_returns_neighbors(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth")
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        assert "security:JWT" in neighbor_ids
        assert "security:RBAC" in neighbor_ids

    def test_explore_returns_rules(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth")
        rule_ids = {r["id"] for r in result["rules"]}
        assert "r1" in rule_ids
        assert "r2" in rule_ids

    def test_explore_missing_node(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("nonexistent:node")
        assert result is None

    def test_explore_depth_2(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:JWT", depth=2)
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        # JWT -> Auth -> RBAC path
        assert "security:Auth" in neighbor_ids
        assert "security:RBAC" in neighbor_ids

    def test_explore_depth_clamped(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth", depth=0)
        assert result is not None
        assert len(result["neighbors"]) > 0

    def test_explore_edge_deduplication(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth")
        edge_keys = [(e["source_id"], e["target_id"], e["relation_type"]) for e in result["edges"]]
        assert len(edge_keys) == len(set(edge_keys))

    def test_explore_edge_relation_type_is_string(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth")
        for edge in result["edges"]:
            assert isinstance(edge["relation_type"], str)

    def test_explore_rules_include_neighbor_rules(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._explore_impl("security:Auth")
        rule_ids = {r["id"] for r in result["rules"]}
        # r3 linked to RBAC (neighbor of Auth)
        assert "r3" in rule_ids


# ===========================================================================
# TestQortexRulesMCP
# ===========================================================================


class TestQortexRulesMCP:
    def test_rules_returns_dict(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._rules_impl()
        assert isinstance(result, dict)
        assert "rules" in result
        assert "domain_count" in result
        assert result["projection"] == "rules"

    def test_rules_returns_all_explicit(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._rules_impl()
        explicit = [r for r in result["rules"] if r["derivation"] == "explicit"]
        assert len(explicit) >= 3

    def test_rules_domain_filter(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._rules_impl(domains=["security"])
        for r in result["rules"]:
            assert r["domain"] == "security"

    def test_rules_concept_ids_filter(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._rules_impl(concept_ids=["security:RBAC"])
        for r in result["rules"]:
            assert "security:RBAC" in r["source_concepts"]

    def test_rules_category_filter(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._rules_impl(categories=["architectural"])
        for r in result["rules"]:
            assert r["category"] == "architectural"

    def test_rules_include_derived_false(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._rules_impl(include_derived=False)
        for r in result["rules"]:
            assert r["derivation"] == "explicit"

    def test_rules_min_confidence(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        backend.add_rule(
            ExplicitRule(
                id="r-low",
                text="Maybe",
                domain="security",
                source_id="test",
                concept_ids=["security:Auth"],
                confidence=0.2,
            )
        )
        result = mcp_server._rules_impl(min_confidence=0.5)
        ids = [r["id"] for r in result["rules"]]
        assert "r-low" not in ids

    def test_rules_empty_backend(self, configured_server):
        result = mcp_server._rules_impl()
        assert result["rules"] == []
        assert result["domain_count"] == 0

    def test_rules_domain_count(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._rules_impl()
        assert result["domain_count"] >= 1


# ===========================================================================
# TestQueryWithRulesMCP
# ===========================================================================


class TestQueryWithRulesMCP:
    def test_query_includes_rules_key(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._query_impl(context="Authentication via OAuth2")
        assert "rules" in result

    def test_query_rules_is_list(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._query_impl(context="Authentication via OAuth2")
        assert isinstance(result["rules"], list)

    def test_query_rules_linked_to_results(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._query_impl(context="Authentication via OAuth2")
        if result["items"]:
            node_ids = {item["node_id"] for item in result["items"]}
            if "security:Auth" in node_ids:
                rule_ids = {r["id"] for r in result["rules"]}
                assert "r1" in rule_ids or "r2" in rule_ids

    def test_query_rules_empty_when_no_results(self, configured_server):
        result = mcp_server._query_impl(context="nothing")
        assert result["rules"] == []

    def test_query_rules_have_all_fields(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._query_impl(context="Authentication via OAuth2")
        for rule in result["rules"]:
            assert "id" in rule
            assert "text" in rule
            assert "domain" in rule
            assert "category" in rule
            assert "confidence" in rule
            assert "relevance" in rule
            assert "derivation" in rule
            assert "source_concepts" in rule

    def test_query_rules_have_relevance_from_scores(self, configured_server, backend, embedding):
        _setup_graph_with_rules(backend, embedding)
        result = mcp_server._query_impl(context="Authentication via OAuth2")
        for rule in result["rules"]:
            assert isinstance(rule["relevance"], float)
