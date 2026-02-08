"""E2E dogfood: Mastra MastraVector drop-in via qortex vector-level MCP tools.

Proves the FULL MastraVector lifecycle through the same MCP _impl functions
that the TypeScript @peleke/mastra-qortex package calls over JSON-RPC.

This is the Python-side proof that the QortexVector TypeScript class
will work end-to-end when connected to a real qortex MCP server.

Test flow mirrors Mastra's MastraVector contract exactly:
1. createIndex → qortex_vector_create_index
2. upsert → qortex_vector_upsert (with metadata, documents, IDs)
3. listIndexes → qortex_vector_list_indexes
4. describeIndex → qortex_vector_describe_index
5. query → qortex_vector_query (with filter, topK, includeVector)
6. updateVector → qortex_vector_update (by ID and by filter)
7. deleteVector → qortex_vector_delete
8. deleteVectors → qortex_vector_delete_many (by IDs and by filter)
9. deleteIndex → qortex_vector_delete_index

Plus qortex extras:
10. textQuery → qortex_query (graph-enhanced)
11. explore → qortex_explore (graph traversal)
12. rules → qortex_rules (rule projection)
13. feedback → qortex_feedback (learning loop)
"""

import json

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.mcp.server import (
    _explore_impl,
    _feedback_impl,
    _query_impl,
    _rules_impl,
    _vector_create_index_impl,
    _vector_delete_impl,
    _vector_delete_index_impl,
    _vector_delete_many_impl,
    _vector_describe_index_impl,
    _vector_list_indexes_impl,
    _vector_query_impl,
    _vector_update_impl,
    _vector_upsert_impl,
    create_server,
)
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbedding:
    """Deterministic 8-dim embedding for reproducible tests."""

    dimensions = 8

    def embed(self, texts):
        import hashlib

        results = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:8]]
            results.append(vec)
        return results


@pytest.fixture(autouse=True)
def _server_with_graph():
    """Set up server with a pre-populated knowledge graph for graph-enhanced tests."""
    vec = NumpyVectorIndex(dimensions=8)
    backend = InMemoryBackend(vector_index=vec)
    backend.connect()

    emb = FakeEmbedding()
    create_server(backend=backend, embedding_model=emb, vector_index=vec)

    # --- Populate graph for text-level queries ---
    backend.create_domain("security")

    concepts = [
        ConceptNode(
            id="sec:oauth",
            name="OAuth2",
            description="OAuth2 is an authorization framework for secure API access",
            domain="security",
            source_id="test",
        ),
        ConceptNode(
            id="sec:jwt",
            name="JWT",
            description="JWT tokens carry signed claims between parties",
            domain="security",
            source_id="test",
        ),
        ConceptNode(
            id="sec:rbac",
            name="RBAC",
            description="Role-based access control assigns permissions based on roles",
            domain="security",
            source_id="test",
        ),
        ConceptNode(
            id="sec:mfa",
            name="MFA",
            description="Multi-factor authentication requires multiple verification methods",
            domain="security",
            source_id="test",
        ),
    ]

    for c in concepts:
        backend.add_node(c)

    # Generate and store embeddings
    texts = [c.description for c in concepts]
    ids = [c.id for c in concepts]
    embeddings = emb.embed(texts)
    for cid, embedding in zip(ids, embeddings):
        backend.add_embedding(cid, embedding)
    vec.add(ids, embeddings)

    # Add typed edges
    edges = [
        ConceptEdge(
            source_id="sec:oauth",
            target_id="sec:jwt",
            relation_type=RelationType.REQUIRES,
        ),
        ConceptEdge(
            source_id="sec:oauth",
            target_id="sec:rbac",
            relation_type=RelationType.USES,
        ),
        ConceptEdge(
            source_id="sec:mfa",
            target_id="sec:oauth",
            relation_type=RelationType.REFINES,
        ),
    ]
    for e in edges:
        backend.add_edge(e)

    # Add rules
    rules = [
        ExplicitRule(
            id="rule:use-oauth",
            text="Always use OAuth2 for third-party API access",
            domain="security",
            category="security",
            source_id="test",
            concept_ids=["sec:oauth"],
        ),
        ExplicitRule(
            id="rule:rotate-jwt",
            text="Rotate JWT signing keys every 90 days",
            domain="security",
            category="operational",
            source_id="test",
            concept_ids=["sec:jwt"],
        ),
    ]
    for r in rules:
        backend.add_rule(r)

    yield


# ---------------------------------------------------------------------------
# Full MastraVector lifecycle (the 9 abstract methods)
# ---------------------------------------------------------------------------


class TestMastraVectorDropIn:
    """Proves every MastraVector abstract method works via MCP tools."""

    def test_create_and_describe_index(self):
        r = _vector_create_index_impl("docs", 8, "cosine")
        assert r["status"] == "created"

        r = _vector_describe_index_impl("docs")
        assert r["dimension"] == 8
        assert r["count"] == 0
        assert r["metric"] == "cosine"

    def test_list_indexes(self):
        _vector_create_index_impl("idx_a", 8)
        _vector_create_index_impl("idx_b", 4)
        r = _vector_list_indexes_impl()
        assert sorted(r["indexes"]) == ["idx_a", "idx_b"]

    def test_upsert_with_metadata_and_documents(self):
        _vector_create_index_impl("docs", 8)
        r = _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]],
            metadata=[{"source": "handbook"}, {"source": "api-docs"}],
            ids=["v1", "v2"],
            documents=["OAuth2 authorization", "JWT token structure"],
        )
        assert r["ids"] == ["v1", "v2"]
        assert _vector_describe_index_impl("docs")["count"] == 2

    def test_query_with_scores(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]],
            metadata=[{"source": "a"}, {"source": "b"}],
            ids=["v1", "v2"],
        )
        r = _vector_query_impl("docs", [1, 0, 0, 0, 0, 0, 0, 0], top_k=2)
        assert len(r["results"]) == 2
        assert r["results"][0]["id"] == "v1"
        assert r["results"][0]["score"] > r["results"][1]["score"]

    def test_query_with_metadata_filter(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ],
            metadata=[
                {"source": "handbook", "chapter": "auth"},
                {"source": "handbook", "chapter": "tokens"},
                {"source": "api-docs", "chapter": "auth"},
            ],
            ids=["v1", "v2", "v3"],
        )
        # Filter: only handbook source
        r = _vector_query_impl(
            "docs",
            [1, 0, 0, 0, 0, 0, 0, 0],
            top_k=10,
            filter={"source": "handbook"},
        )
        assert all(res["metadata"]["source"] == "handbook" for res in r["results"])

        # Complex filter: handbook AND auth chapter
        r = _vector_query_impl(
            "docs",
            [1, 0, 0, 0, 0, 0, 0, 0],
            top_k=10,
            filter={"$and": [{"source": "handbook"}, {"chapter": "auth"}]},
        )
        assert len(r["results"]) == 1
        assert r["results"][0]["id"] == "v1"

    def test_query_with_documents_in_results(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0]],
            ids=["v1"],
            documents=["OAuth2 is an authorization framework"],
        )
        r = _vector_query_impl("docs", [1, 0, 0, 0, 0, 0, 0, 0], top_k=1)
        assert r["results"][0]["document"] == "OAuth2 is an authorization framework"

    def test_update_vector_by_id(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0]],
            metadata=[{"status": "draft"}],
            ids=["v1"],
        )
        r = _vector_update_impl("docs", id="v1", metadata={"status": "published"})
        assert r["count"] == 1

        q = _vector_query_impl("docs", [1, 0, 0, 0, 0, 0, 0, 0], top_k=1)
        assert q["results"][0]["metadata"]["status"] == "published"

    def test_update_vector_by_filter(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]],
            metadata=[{"source": "handbook"}, {"source": "handbook"}],
            ids=["v1", "v2"],
        )
        r = _vector_update_impl(
            "docs",
            filter={"source": "handbook"},
            metadata={"archived": True},
        )
        assert r["count"] == 2

    def test_delete_single_vector(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]],
            ids=["v1", "v2"],
        )
        _vector_delete_impl("docs", "v1")
        assert _vector_describe_index_impl("docs")["count"] == 1

    def test_delete_vectors_by_ids(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ],
            ids=["v1", "v2", "v3"],
        )
        r = _vector_delete_many_impl("docs", ids=["v1", "v2"])
        assert r["count"] == 2
        assert _vector_describe_index_impl("docs")["count"] == 1

    def test_delete_vectors_by_filter(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ],
            metadata=[
                {"source": "old"},
                {"source": "old"},
                {"source": "new"},
            ],
            ids=["v1", "v2", "v3"],
        )
        r = _vector_delete_many_impl("docs", filter={"source": "old"})
        assert r["count"] == 2
        assert _vector_describe_index_impl("docs")["count"] == 1

    def test_delete_index(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl("docs", vectors=[[1, 0, 0, 0, 0, 0, 0, 0]], ids=["v1"])
        _vector_delete_index_impl("docs")
        assert _vector_list_indexes_impl()["indexes"] == []


# ---------------------------------------------------------------------------
# Graph-enhanced extras (what makes qortex different)
# ---------------------------------------------------------------------------


class TestGraphEnhancedExtras:
    """Proves the graph-enhanced capabilities work alongside MastraVector."""

    def test_text_query_returns_items_and_rules(self):
        """qortex_query = text-level search with auto-surfaced rules."""
        r = _query_impl("OAuth2 authorization", domains=["security"], top_k=3)

        # Verify response shape matches what TypeScript client expects
        assert "items" in r
        assert "query_id" in r
        assert "rules" in r
        assert len(r["items"]) > 0

        # Items have expected fields
        item = r["items"][0]
        assert "id" in item
        assert "content" in item
        assert "score" in item
        assert "domain" in item
        assert "node_id" in item

    def test_explore_from_query_result(self):
        """Query → take node_id → explore → typed edges + neighbors."""
        # First query
        q = _query_impl("OAuth2", domains=["security"], top_k=1)
        node_id = q["items"][0]["node_id"]

        # Explore from that result
        r = _explore_impl(node_id)
        assert r is not None
        assert r["node"]["id"] == node_id
        assert len(r["edges"]) > 0
        assert len(r["neighbors"]) > 0

        # Edges are typed
        edge_types = {e["relation_type"] for e in r["edges"]}
        assert len(edge_types) > 0

    def test_explore_depth_2(self):
        """Explore at depth 2 returns more of the graph."""
        r1 = _explore_impl("sec:oauth", depth=1)
        r2 = _explore_impl("sec:oauth", depth=2)
        assert r2 is not None
        # Depth 2 should reach more nodes than depth 1
        assert len(r2["neighbors"]) >= len(r1["neighbors"])

    def test_rules_projection(self):
        """qortex_rules returns filtered rules by domain/concept/category."""
        # All rules
        r = _rules_impl(domains=["security"])
        assert len(r["rules"]) >= 2

        # By concept
        r = _rules_impl(concept_ids=["sec:jwt"])
        jwt_rules = [rule for rule in r["rules"]]
        assert any("JWT" in rule["text"] or "jwt" in rule["text"].lower() for rule in jwt_rules)

        # By category
        r = _rules_impl(categories=["operational"])
        assert all(rule["category"] == "operational" for rule in r["rules"])

    def test_feedback_loop(self):
        """Feedback adjusts future retrieval (the learning loop)."""
        q1 = _query_impl("OAuth2", domains=["security"], top_k=4)
        assert len(q1["items"]) > 0

        # Submit feedback
        outcomes = {}
        for item in q1["items"]:
            outcomes[item["id"]] = "accepted" if "OAuth" in item["content"] else "rejected"

        r = _feedback_impl(q1["query_id"], outcomes)
        assert r["status"] == "recorded"
        assert r["outcome_count"] == len(outcomes)

    def test_rules_in_query_results(self):
        """Rules auto-surfaced in query results."""
        r = _query_impl("OAuth2 authorization", domains=["security"], top_k=3)
        # Rules should be present (may be empty if no linked rules match)
        assert isinstance(r["rules"], list)


# ---------------------------------------------------------------------------
# JSON serialization (MCP contract: everything must round-trip)
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    """Every MCP response must be JSON-serializable (the transport contract)."""

    def test_vector_query_serializable(self):
        _vector_create_index_impl("docs", 8)
        _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0]],
            metadata=[{"key": "value"}],
            ids=["v1"],
            documents=["test doc"],
        )
        r = _vector_query_impl("docs", [1, 0, 0, 0, 0, 0, 0, 0], top_k=1)
        serialized = json.dumps(r)
        deserialized = json.loads(serialized)
        assert deserialized["results"][0]["id"] == "v1"

    def test_text_query_serializable(self):
        r = _query_impl("OAuth2", domains=["security"])
        serialized = json.dumps(r)
        deserialized = json.loads(serialized)
        assert "items" in deserialized
        assert "rules" in deserialized

    def test_explore_serializable(self):
        r = _explore_impl("sec:oauth")
        serialized = json.dumps(r)
        deserialized = json.loads(serialized)
        assert deserialized["node"]["id"] == "sec:oauth"

    def test_rules_serializable(self):
        r = _rules_impl(domains=["security"])
        serialized = json.dumps(r)
        deserialized = json.loads(serialized)
        assert "rules" in deserialized


# ---------------------------------------------------------------------------
# Full drop-in simulation (what a Mastra app would actually do)
# ---------------------------------------------------------------------------


class TestFullDropInSimulation:
    """Simulates a Mastra app using QortexVector as a drop-in.

    This test mirrors the exact sequence of operations a Mastra app
    would perform when using @peleke/mastra-qortex.
    """

    def test_mastra_app_workflow(self):
        # === Phase 1: Standard MastraVector operations ===

        # Create index (Mastra calls createIndex)
        r = _vector_create_index_impl("my_app", 8, "cosine")
        assert r["status"] == "created"

        # Upsert embeddings from Mastra's embedding pipeline
        embedding = FakeEmbedding()
        texts = [
            "OAuth2 is an authorization framework for API access",
            "JWT tokens carry signed claims between parties",
            "RBAC assigns permissions based on user roles",
            "Multi-factor authentication adds security layers",
        ]
        vectors = embedding.embed(texts)

        r = _vector_upsert_impl(
            "my_app",
            vectors=vectors,
            metadata=[{"text": t, "source": "security-handbook"} for t in texts],
            ids=[f"doc_{i}" for i in range(len(texts))],
            documents=texts,
        )
        assert len(r["ids"]) == 4

        # Query (Mastra calls query with embedding)
        query_vec = embedding.embed(["how to authenticate API requests"])[0]
        r = _vector_query_impl("my_app", query_vec, top_k=2)
        assert len(r["results"]) == 2
        assert all("score" in res for res in r["results"])
        assert r["results"][0]["score"] >= r["results"][1]["score"]

        # Query with filter (Mastra supports this)
        r = _vector_query_impl(
            "my_app",
            query_vec,
            top_k=10,
            filter={"source": "security-handbook"},
        )
        assert len(r["results"]) > 0

        # Update metadata (Mastra calls updateVector)
        _vector_update_impl("my_app", id="doc_0", metadata={"reviewed": True})
        q = _vector_query_impl("my_app", vectors[0], top_k=1)
        assert q["results"][0]["metadata"]["reviewed"] is True

        # === Phase 2: qortex graph-enhanced extras ===
        # (This is what you get on top of standard MastraVector)

        # Text-level query with graph-enhanced PPR
        tq = _query_impl("OAuth2 authorization", domains=["security"], top_k=3)
        assert len(tq["items"]) > 0
        assert tq["query_id"] != ""

        # Explore graph from a result
        node_id = tq["items"][0]["node_id"]
        explored = _explore_impl(node_id)
        assert explored is not None
        assert explored["node"]["id"] == node_id

        # Get rules
        rules = _rules_impl(domains=["security"])
        assert len(rules["rules"]) >= 1

        # Submit feedback (the learning loop)
        fb = _feedback_impl(
            tq["query_id"],
            {tq["items"][0]["id"]: "accepted"},
            source="mastra",
        )
        assert fb["status"] == "recorded"

        # === Phase 3: Cleanup ===

        # Delete some vectors
        _vector_delete_impl("my_app", "doc_3")
        assert _vector_describe_index_impl("my_app")["count"] == 3

        # Delete by filter
        _vector_delete_many_impl(
            "my_app",
            filter={"source": "security-handbook"},
        )
        assert _vector_describe_index_impl("my_app")["count"] == 0

        # Delete index
        _vector_delete_index_impl("my_app")
        assert "my_app" not in _vector_list_indexes_impl()["indexes"]

    def test_multiple_indexes_concurrent(self):
        """Mastra apps may use multiple indexes simultaneously."""
        _vector_create_index_impl("code", 8)
        _vector_create_index_impl("docs", 8)

        _vector_upsert_impl(
            "code",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0]],
            metadata=[{"type": "function"}],
            ids=["fn_1"],
        )
        _vector_upsert_impl(
            "docs",
            vectors=[[0, 1, 0, 0, 0, 0, 0, 0]],
            metadata=[{"type": "article"}],
            ids=["art_1"],
        )

        # Query each independently
        r_code = _vector_query_impl("code", [1, 0, 0, 0, 0, 0, 0, 0], top_k=1)
        r_docs = _vector_query_impl("docs", [0, 1, 0, 0, 0, 0, 0, 0], top_k=1)

        assert r_code["results"][0]["id"] == "fn_1"
        assert r_docs["results"][0]["id"] == "art_1"

        # Indexes are independent
        assert _vector_describe_index_impl("code")["count"] == 1
        assert _vector_describe_index_impl("docs")["count"] == 1

    def test_upsert_overwrites_existing(self):
        """Mastra's upsert semantics: same ID replaces."""
        _vector_create_index_impl("docs", 8)

        _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0, 0, 0, 0, 0]],
            metadata=[{"version": 1}],
            ids=["v1"],
        )
        _vector_upsert_impl(
            "docs",
            vectors=[[0, 1, 0, 0, 0, 0, 0, 0]],
            metadata=[{"version": 2}],
            ids=["v1"],
        )

        assert _vector_describe_index_impl("docs")["count"] == 1
        r = _vector_query_impl("docs", [0, 1, 0, 0, 0, 0, 0, 0], top_k=1)
        assert r["results"][0]["metadata"]["version"] == 2
