"""E2E dogfood: Mastra MCP client using graph-enhanced qortex.

Proves that any MCP client (Mastra TS, Claude Desktop, etc.) can use the
full graph-enhanced pipeline via MCP tools:

1. qortex_query  -> items WITH rules auto-surfaced
2. qortex_explore -> navigate typed edges from any query result
3. qortex_rules  -> get projected rules by domain/concept/category
4. qortex_feedback -> close the learning loop
5. Full loop: query -> explore -> rules -> feedback -> re-query

Uses _impl functions directly (same code path as JSON-RPC tool calls).
No MCP transport needed. FakeEmbedding for deterministic results.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.mcp import server as mcp_server
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Test infrastructure
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


def setup_graph_server():
    """Boot MCP server with a graph that has concepts, edges, and rules.

    This is the realistic setup: not just vectors, but typed relationships
    and explicit rules. This is what differentiates qortex from Pinecone/Chroma.
    """
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeEmbedding()

    backend.create_domain("security")

    nodes = [
        ConceptNode(
            id="sec:oauth",
            name="OAuth2",
            description="OAuth2 authorization framework for delegated API access",
            domain="security",
            source_id="handbook",
        ),
        ConceptNode(
            id="sec:jwt",
            name="JWT",
            description="JSON Web Tokens for stateless authentication",
            domain="security",
            source_id="handbook",
        ),
        ConceptNode(
            id="sec:rbac",
            name="RBAC",
            description="Role-based access control for permission management",
            domain="security",
            source_id="handbook",
        ),
        ConceptNode(
            id="sec:mfa",
            name="MFA",
            description="Multi-factor authentication with multiple verification factors",
            domain="security",
            source_id="handbook",
        ),
    ]

    for node in nodes:
        backend.add_node(node)

    texts = [f"{n.name}: {n.description}" for n in nodes]
    embeddings = embedding.embed(texts)
    for node, emb in zip(nodes, embeddings):
        backend.add_embedding(node.id, emb)

    ids = [n.id for n in nodes]
    vector_index.add(ids, embeddings)

    # Typed edges
    backend.add_edge(
        ConceptEdge(
            source_id="sec:oauth",
            target_id="sec:jwt",
            relation_type=RelationType.REQUIRES,
        )
    )
    backend.add_edge(
        ConceptEdge(
            source_id="sec:oauth",
            target_id="sec:rbac",
            relation_type=RelationType.USES,
        )
    )
    backend.add_edge(
        ConceptEdge(
            source_id="sec:mfa",
            target_id="sec:oauth",
            relation_type=RelationType.SUPPORTS,
        )
    )

    # Explicit rules
    backend.add_rule(
        ExplicitRule(
            id="rule:use-oauth",
            text="Always use OAuth2 for third-party API access",
            domain="security",
            source_id="handbook",
            concept_ids=["sec:oauth"],
            category="security",
        )
    )
    backend.add_rule(
        ExplicitRule(
            id="rule:rotate-jwt",
            text="Rotate JWT signing keys every 90 days",
            domain="security",
            source_id="handbook",
            concept_ids=["sec:oauth", "sec:jwt"],
            category="operations",
        )
    )
    backend.add_rule(
        ExplicitRule(
            id="rule:rbac-first",
            text="Define RBAC roles before writing authorization code",
            domain="security",
            source_id="handbook",
            concept_ids=["sec:rbac"],
            category="architectural",
        )
    )

    mcp_server.create_server(
        backend=backend,
        embedding_model=embedding,
        vector_index=vector_index,
    )

    return backend, vector_index, embedding


# ===========================================================================
# MCP 1: qortex_query returns rules
# ===========================================================================


class TestMCPQueryWithRules:
    """qortex_query now auto-surfaces linked rules in the response."""

    def test_query_includes_rules_key(self):
        setup_graph_server()
        result = mcp_server._query_impl(context="OAuth2 authorization")
        assert "rules" in result

    def test_query_rules_linked_to_results(self):
        setup_graph_server()
        result = mcp_server._query_impl(context="OAuth2 authorization")
        if result["items"]:
            node_ids = {item["node_id"] for item in result["items"]}
            if "sec:oauth" in node_ids:
                rule_ids = {r["id"] for r in result["rules"]}
                assert "rule:use-oauth" in rule_ids or "rule:rotate-jwt" in rule_ids

    def test_query_rules_have_all_fields(self):
        setup_graph_server()
        result = mcp_server._query_impl(context="OAuth2 authorization")
        for rule in result["rules"]:
            assert "id" in rule
            assert "text" in rule
            assert "domain" in rule
            assert "category" in rule
            assert "relevance" in rule
            assert "source_concepts" in rule

    def test_query_rules_json_serializable(self):
        setup_graph_server()
        result = mcp_server._query_impl(context="OAuth2 authorization")
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["rules"] == result["rules"]


# ===========================================================================
# MCP 2: qortex_explore from query results
# ===========================================================================


class TestMCPExploreFromQuery:
    """Query via MCP, then explore the graph from a result node_id."""

    def test_query_then_explore(self):
        """Mastra TS client: query -> take node_id -> explore neighborhood."""
        setup_graph_server()

        # Step 1: Query (MCP tool call)
        query_result = mcp_server._query_impl(context="OAuth2 authorization")
        assert len(query_result["items"]) > 0

        # Step 2: Explore (MCP tool call)
        node_id = query_result["items"][0]["node_id"]
        explore_result = mcp_server._explore_impl(node_id)
        assert explore_result is not None
        assert explore_result["node"]["id"] == node_id

    def test_explore_returns_typed_edges(self):
        setup_graph_server()
        result = mcp_server._explore_impl("sec:oauth")
        assert len(result["edges"]) > 0
        for edge in result["edges"]:
            assert "source_id" in edge
            assert "target_id" in edge
            assert "relation_type" in edge
            assert isinstance(edge["relation_type"], str)

    def test_explore_returns_neighbors(self):
        setup_graph_server()
        result = mcp_server._explore_impl("sec:oauth")
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        assert "sec:jwt" in neighbor_ids
        assert "sec:rbac" in neighbor_ids

    def test_explore_returns_linked_rules(self):
        setup_graph_server()
        result = mcp_server._explore_impl("sec:oauth")
        rule_ids = {r["id"] for r in result["rules"]}
        assert "rule:use-oauth" in rule_ids

    def test_explore_depth_2_reaches_farther(self):
        setup_graph_server()
        result = mcp_server._explore_impl("sec:jwt", depth=2)
        neighbor_ids = {n["id"] for n in result["neighbors"]}
        # JWT -> OAuth -> RBAC path
        assert "sec:oauth" in neighbor_ids
        assert "sec:rbac" in neighbor_ids

    def test_explore_nonexistent_returns_none(self):
        setup_graph_server()
        result = mcp_server._explore_impl("nonexistent:node")
        assert result is None

    def test_explore_json_serializable(self):
        setup_graph_server()
        result = mcp_server._explore_impl("sec:oauth")
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["node"]["id"] == "sec:oauth"


# ===========================================================================
# MCP 3: qortex_rules projection
# ===========================================================================


class TestMCPRulesProjection:
    """qortex_rules: get projected rules from the knowledge graph."""

    def test_rules_returns_all(self):
        setup_graph_server()
        result = mcp_server._rules_impl()
        assert len(result["rules"]) >= 3
        assert result["projection"] == "rules"

    def test_rules_filter_by_domain(self):
        setup_graph_server()
        result = mcp_server._rules_impl(domains=["security"])
        for r in result["rules"]:
            assert r["domain"] == "security"

    def test_rules_filter_by_concept_ids(self):
        setup_graph_server()
        result = mcp_server._rules_impl(concept_ids=["sec:rbac"])
        for r in result["rules"]:
            assert "sec:rbac" in r["source_concepts"]

    def test_rules_filter_by_category(self):
        setup_graph_server()
        result = mcp_server._rules_impl(categories=["operations"])
        for r in result["rules"]:
            assert r["category"] == "operations"

    def test_rules_json_serializable(self):
        setup_graph_server()
        result = mcp_server._rules_impl()
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert len(deserialized["rules"]) == len(result["rules"])


# ===========================================================================
# MCP 4: Full Mastra consumer loop
# ===========================================================================


class TestMCPFullMastraLoop:
    """The full enchilada: what a Mastra TS client does over MCP.

    1. Status check
    2. Query with rules auto-surfaced
    3. Explore graph from top result
    4. Get rules for activated concepts
    5. Feedback
    6. Re-query
    """

    def test_complete_mcp_workflow(self):
        setup_graph_server()

        # 1. Status (health check)
        status = mcp_server._status_impl()
        assert status["status"] == "ok"
        assert status["vector_search"] is True

        # 2. Query (Mastra: store.query())
        query_result = mcp_server._query_impl(
            context="How to implement authentication?",
            domains=["security"],
            top_k=4,
        )
        assert len(query_result["items"]) > 0
        assert "rules" in query_result
        query_id = query_result["query_id"]

        # 3. Explore top result (Mastra: qortex_explore tool call)
        top_node_id = query_result["items"][0]["node_id"]
        explore_result = mcp_server._explore_impl(top_node_id)
        assert explore_result is not None
        assert explore_result["node"]["id"] == top_node_id

        # 4. Get rules for all activated concepts
        activated_ids = [item["node_id"] for item in query_result["items"]]
        rules_result = mcp_server._rules_impl(concept_ids=activated_ids)
        assert isinstance(rules_result["rules"], list)

        # 5. Feedback (the thing Mastra can't do natively)
        feedback_result = mcp_server._feedback_impl(
            query_id=query_id,
            outcomes={query_result["items"][0]["id"]: "accepted"},
            source="mastra-mcp-dogfood",
        )
        assert feedback_result["status"] == "recorded"

        # 6. Re-query (in Level 2, this improves from feedback)
        query_result_2 = mcp_server._query_impl(
            context="How to implement authentication?",
            domains=["security"],
            top_k=4,
        )
        assert len(query_result_2["items"]) > 0
        assert query_result_2["query_id"] != query_id

    def test_mastra_shape_mapping(self):
        """Map MCP responses to Mastra QueryResult shape end-to-end."""
        setup_graph_server()

        raw = mcp_server._query_impl(
            context="OAuth2 authorization",
            domains=["security"],
            top_k=3,
        )

        # Map to Mastra QueryResult: {id, score, metadata, document}
        mastra_results = [
            {
                "id": item["id"],
                "score": item["score"],
                "metadata": {
                    **item["metadata"],
                    "domain": item["domain"],
                    "node_id": item["node_id"],
                },
                "document": item["content"],
            }
            for item in raw["items"]
        ]

        expected_keys = {"id", "score", "metadata", "document"}
        for r in mastra_results:
            assert set(r.keys()) == expected_keys
            assert isinstance(r["score"], float)
            assert isinstance(r["document"], str)

        # NEW: rules are also available for the Mastra client
        assert "rules" in raw
        for rule in raw["rules"]:
            assert "id" in rule
            assert "text" in rule

    def test_all_mcp_responses_json_serializable(self):
        """Every MCP response must survive JSON roundtrip (stdio transport)."""
        setup_graph_server()

        # Status
        status = mcp_server._status_impl()
        assert json.loads(json.dumps(status)) == status

        # Domains
        domains = mcp_server._domains_impl()
        assert json.loads(json.dumps(domains)) == domains

        # Query
        query = mcp_server._query_impl("OAuth2", domains=["security"], top_k=2)
        assert json.loads(json.dumps(query)) == query

        # Explore
        if query["items"]:
            explore = mcp_server._explore_impl(query["items"][0]["node_id"])
            assert json.loads(json.dumps(explore)) == explore

        # Rules
        rules = mcp_server._rules_impl()
        assert json.loads(json.dumps(rules)) == rules

        # Feedback
        if query["items"]:
            fb = mcp_server._feedback_impl(
                query["query_id"],
                {query["items"][0]["id"]: "accepted"},
            )
            assert json.loads(json.dumps(fb)) == fb
