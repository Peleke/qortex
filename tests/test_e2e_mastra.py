"""E2E: qortex MCP server as a Mastra vector store backend.

This is not a unit test. This boots the real MCP server with a real
embedding model (sentence-transformers/all-MiniLM-L6-v2), ingests a real
document, queries it with real embeddings, and verifies the responses
match what Mastra's MastraVector interface expects.

The MCP server is the bridge to TypeScript. A Mastra TS client would
call these same tools over stdio JSON-RPC. This test proves the server
returns correct shapes with real data — not mocks, not fakes.

Requires: sentence-transformers (pip install sentence-transformers)
"""

from __future__ import annotations

import json
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Skip if sentence-transformers not installed
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

pytestmark = pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")


# ---------------------------------------------------------------------------
# Fixtures: real MCP server with real embedding model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcp_server():
    """Boot the real MCP server with real everything.

    - Real embedding model (all-MiniLM-L6-v2, 384 dims)
    - Real VectorIndex (NumpyVectorIndex)
    - Real GraphBackend (InMemoryBackend)
    - StubLLMBackend for extraction (no API key needed)

    scope=module so we load the model once, not per test.
    """
    from qortex.core.memory import InMemoryBackend
    from qortex.mcp.server import create_server, set_llm_backend
    from qortex.vec.embeddings import SentenceTransformerEmbedding
    from qortex.vec.index import NumpyVectorIndex
    from qortex_ingest.base import StubLLMBackend

    embedding = SentenceTransformerEmbedding()
    vector_index = NumpyVectorIndex(dimensions=embedding.dimensions)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()

    server = create_server(
        backend=backend,
        embedding_model=embedding,
        vector_index=vector_index,
    )

    set_llm_backend(StubLLMBackend(concepts=[
        {
            "name": "OAuth2",
            "description": "OAuth2 is an open standard for access delegation, "
                           "commonly used for token-based authentication and authorization",
            "confidence": 1.0,
        },
        {
            "name": "JWT",
            "description": "JSON Web Tokens are compact, URL-safe tokens for "
                           "transmitting claims between parties in web applications",
            "confidence": 0.95,
        },
        {
            "name": "RBAC",
            "description": "Role-based access control restricts system access to "
                           "authorized users based on their assigned organizational roles",
            "confidence": 0.9,
        },
        {
            "name": "MFA",
            "description": "Multi-factor authentication requires users to provide "
                           "two or more verification factors to gain access to a resource",
            "confidence": 0.85,
        },
    ]))

    return server


@pytest.fixture(scope="module")
def ingested_domain(mcp_server, tmp_path_factory):
    """Ingest a real document into the MCP server."""
    from qortex.mcp.server import _ingest_impl

    doc_path = tmp_path_factory.mktemp("docs") / "auth_guide.txt"
    doc_path.write_text(textwrap.dedent("""\
        Authentication and Authorization Best Practices

        OAuth2 provides delegated authorization for web applications.
        It uses access tokens and refresh tokens to manage sessions securely.
        JWT (JSON Web Tokens) are commonly used as the token format in OAuth2 flows.

        Role-based access control (RBAC) should be implemented at the API gateway level.
        Each endpoint should declare required roles and permissions.
        Multi-factor authentication (MFA) adds an additional security layer beyond passwords.

        When implementing authentication, always use HTTPS, validate all tokens server-side,
        and implement proper token rotation and revocation mechanisms.
    """))

    result = _ingest_impl(str(doc_path), "security")
    assert "error" not in result, f"Ingest failed: {result}"
    assert result["concepts"] >= 1
    return result


# ===========================================================================
# Mastra MastraVector.query() equivalent
# ===========================================================================


class TestMastraQuery:
    """MastraVector.query(indexName, queryVector, topK, filter, includeVector)

    In qortex MCP: qortex_query(context, domains, top_k, min_confidence)
    The Mastra TS client would call qortex_query and map the response.
    """

    def test_query_returns_results_with_real_embeddings(self, mcp_server, ingested_domain):
        from qortex.mcp.server import _query_impl

        result = _query_impl(
            context="How does OAuth2 authentication work?",
            domains=["security"],
            top_k=3,
        )

        assert "items" in result
        assert "query_id" in result
        assert len(result["items"]) > 0
        assert result["query_id"] != ""

    def test_query_result_maps_to_mastra_query_result(self, mcp_server, ingested_domain):
        """Verify the MCP response maps cleanly to Mastra's QueryResult shape."""
        from qortex.mcp.server import _query_impl

        result = _query_impl(
            context="What is role-based access control?",
            domains=["security"],
            top_k=5,
        )

        # Map to Mastra shape — this is what the TS @qortex/mastra package does
        mastra_results = []
        for item in result["items"]:
            mastra_result = {
                "id": item["id"],
                "score": item["score"],
                "metadata": {
                    **item["metadata"],
                    "domain": item["domain"],
                    "node_id": item["node_id"],
                },
                "document": item["content"],
            }
            mastra_results.append(mastra_result)

        # Verify Mastra QueryResult shape: {id, score, metadata, document}
        expected_keys = {"id", "score", "metadata", "document"}
        for r in mastra_results:
            assert set(r.keys()) == expected_keys
            assert isinstance(r["id"], str)
            assert isinstance(r["score"], float)
            assert isinstance(r["metadata"], dict)
            assert isinstance(r["document"], str)
            assert len(r["document"]) > 0

    def test_semantic_relevance_with_real_embeddings(self, mcp_server, ingested_domain):
        """Real embeddings should rank semantically relevant results higher."""
        from qortex.mcp.server import _query_impl

        result = _query_impl(
            context="OAuth2 token-based authentication",
            domains=["security"],
            top_k=4,
        )

        assert len(result["items"]) > 0

        # With real embeddings, "OAuth2" concept should score higher than
        # unrelated concepts for an OAuth2 query
        top_result = result["items"][0]
        assert top_result["score"] > 0.0
        # The top result should be semantically related to OAuth2
        assert "oauth" in top_result["content"].lower() or "auth" in top_result["content"].lower()

    def test_domain_filtering(self, mcp_server, ingested_domain):
        """Querying a non-existent domain returns empty results."""
        from qortex.mcp.server import _query_impl

        result = _query_impl(
            context="OAuth2",
            domains=["nonexistent_domain"],
            top_k=5,
        )

        assert result["items"] == []

    def test_top_k_limits_results(self, mcp_server, ingested_domain):
        from qortex.mcp.server import _query_impl

        result = _query_impl(
            context="authentication",
            domains=["security"],
            top_k=1,
        )

        assert len(result["items"]) <= 1


# ===========================================================================
# Mastra MastraVector.listIndexes() equivalent
# ===========================================================================


class TestMastraListIndexes:
    """MastraVector.listIndexes() → IndexStats[]

    In qortex MCP: qortex_domains()
    The TS client maps domains to Mastra's IndexStats shape.
    """

    def test_domains_map_to_index_stats(self, mcp_server, ingested_domain):
        from qortex.mcp.server import _domains_impl, _embedding_model

        result = _domains_impl()

        # Map to Mastra IndexStats shape
        mastra_indexes = []
        for d in result["domains"]:
            mastra_indexes.append({
                "name": d["name"],
                "dimension": _embedding_model.dimensions,
                "metric": "cosine",
                "count": d["concept_count"],
            })

        expected_keys = {"name", "dimension", "metric", "count"}
        assert len(mastra_indexes) > 0
        for idx in mastra_indexes:
            assert set(idx.keys()) == expected_keys
            assert isinstance(idx["name"], str)
            assert isinstance(idx["dimension"], int)
            assert idx["dimension"] == 384  # all-MiniLM-L6-v2
            assert idx["metric"] == "cosine"
            assert isinstance(idx["count"], int)
            assert idx["count"] > 0


# ===========================================================================
# Mastra MastraVector.describeIndex() equivalent
# ===========================================================================


class TestMastraDescribeIndex:
    def test_describe_security_domain(self, mcp_server, ingested_domain):
        from qortex.mcp.server import _domains_impl, _embedding_model

        result = _domains_impl()
        security = next((d for d in result["domains"] if d["name"] == "security"), None)
        assert security is not None

        # Mastra describeIndex shape
        index_info = {
            "name": security["name"],
            "dimension": _embedding_model.dimensions,
            "metric": "cosine",
            "count": security["concept_count"],
        }

        assert index_info["name"] == "security"
        assert index_info["dimension"] == 384
        assert index_info["count"] == 4  # 4 concepts from StubLLMBackend


# ===========================================================================
# Feedback: the thing Mastra can't do
# ===========================================================================


class TestMastraFeedbackUpgrade:
    """This is the qortex differentiator. Mastra has no feedback loop.

    A Mastra TS client using qortex gets this for free via MCP.
    """

    def test_full_query_feedback_cycle(self, mcp_server, ingested_domain):
        from qortex.mcp.server import _feedback_impl, _query_impl

        # 1. Query (same as Mastra would)
        query_result = _query_impl(
            context="authentication methods and best practices",
            domains=["security"],
            top_k=3,
        )
        assert len(query_result["items"]) > 0
        query_id = query_result["query_id"]

        # 2. Feedback (Mastra can't do this)
        feedback_result = _feedback_impl(
            query_id=query_id,
            outcomes={
                query_result["items"][0]["id"]: "accepted",
            },
            source="mastra",
        )

        assert feedback_result["status"] == "recorded"
        assert feedback_result["query_id"] == query_id
        assert feedback_result["outcome_count"] == 1
        assert feedback_result["source"] == "mastra"


# ===========================================================================
# Full Mastra consumer simulation
# ===========================================================================


class TestMastraE2ESimulation:
    """Simulate what a Mastra TS consumer would do end-to-end.

    1. Check server status
    2. List available indexes
    3. Query with real text
    4. Map results to Mastra shapes
    5. Give feedback
    6. Query again

    Every step uses the actual MCP server _impl functions — the same
    code paths that JSON-RPC tool calls hit.
    """

    def test_full_consumer_workflow(self, mcp_server, ingested_domain):
        from qortex.mcp.server import (
            _domains_impl,
            _embedding_model,
            _feedback_impl,
            _query_impl,
            _status_impl,
        )

        # --- Step 1: Check server (Mastra would do this on init) ---
        status = _status_impl()
        assert status["status"] == "ok"
        assert status["vector_search"] is True
        assert status["embedding_model"] == "all-MiniLM-L6-v2"

        # --- Step 2: listIndexes() ---
        domains = _domains_impl()
        indexes = [
            {
                "name": d["name"],
                "dimension": _embedding_model.dimensions,
                "metric": "cosine",
                "count": d["concept_count"],
            }
            for d in domains["domains"]
        ]
        assert any(idx["name"] == "security" for idx in indexes)

        # --- Step 3: query() ---
        raw = _query_impl(
            context="How should I implement token-based authentication?",
            domains=["security"],
            top_k=4,
        )

        # --- Step 4: Map to Mastra QueryResult[] ---
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

        assert len(mastra_results) > 0

        # Verify all results are valid Mastra QueryResult shapes
        for r in mastra_results:
            assert set(r.keys()) == {"id", "score", "metadata", "document"}
            assert r["score"] > 0.0
            assert len(r["document"]) > 0

        # With real embeddings, results should be semantically relevant
        all_content = " ".join(r["document"].lower() for r in mastra_results)
        assert any(
            term in all_content
            for term in ["auth", "token", "oauth", "jwt", "access"]
        ), f"Results not semantically relevant: {all_content[:200]}"

        # --- Step 5: feedback() (the upgrade Mastra doesn't have) ---
        fb = _feedback_impl(
            query_id=raw["query_id"],
            outcomes={mastra_results[0]["id"]: "accepted"},
            source="mastra-e2e",
        )
        assert fb["status"] == "recorded"

        # --- Step 6: query() again (in Level 2, this would improve) ---
        raw2 = _query_impl(
            context="What roles and permissions should API endpoints require?",
            domains=["security"],
            top_k=3,
        )
        assert len(raw2["items"]) > 0
        assert raw2["query_id"] != raw["query_id"]  # different query IDs

    def test_json_serializable(self, mcp_server, ingested_domain):
        """Every MCP response must be JSON-serializable (goes over stdio)."""
        from qortex.mcp.server import (
            _domains_impl,
            _feedback_impl,
            _query_impl,
            _status_impl,
        )

        # All responses must round-trip through JSON
        status = _status_impl()
        assert json.loads(json.dumps(status)) == status

        domains = _domains_impl()
        assert json.loads(json.dumps(domains)) == domains

        query = _query_impl("test query", domains=["security"], top_k=2)
        assert json.loads(json.dumps(query)) == query

        if query["items"]:
            fb = _feedback_impl(query["query_id"], {query["items"][0]["id"]: "accepted"})
            assert json.loads(json.dumps(fb)) == fb


# ===========================================================================
# Python adapter (QortexVectorStore) with real embeddings
# ===========================================================================


class TestMastraAdapterRealEmbeddings:
    """Same tests but through the Python QortexVectorStore adapter.

    This is the Python-side drop-in. It wraps LocalQortexClient
    which uses the same real embedding model.
    """

    @pytest.fixture
    def store(self, mcp_server, ingested_domain):
        """Create a QortexVectorStore backed by real embeddings."""
        from qortex.adapters.mastra import QortexVectorStore
        from qortex.client import LocalQortexClient
        from qortex.mcp.server import _backend, _embedding_model, _vector_index

        client = LocalQortexClient(
            vector_index=_vector_index,
            backend=_backend,
            embedding_model=_embedding_model,
        )
        return QortexVectorStore(client=client)

    def test_query_with_real_embeddings(self, store):
        results = store.query(
            index_name="security",
            query_text="How does OAuth2 work?",
            top_k=3,
        )

        assert len(results) > 0
        assert all(set(r.keys()) == {"id", "score", "metadata", "document"} for r in results)
        assert results[0]["score"] > 0.0

        # Semantic relevance check
        top_doc = results[0]["document"].lower()
        assert any(term in top_doc for term in ["oauth", "auth", "token", "access"])

    def test_list_indexes_with_real_data(self, store):
        indexes = store.list_indexes()

        assert any(idx["name"] == "security" for idx in indexes)
        security = next(idx for idx in indexes if idx["name"] == "security")
        assert security["count"] == 4
        assert security["dimension"] == 384
        assert security["metric"] == "cosine"

    def test_feedback_with_real_query(self, store):
        results = store.query(
            index_name="security",
            query_text="multi-factor authentication",
            top_k=2,
        )
        assert len(results) > 0
        assert store.last_query_id is not None

        # Feedback — no error
        store.feedback({results[0]["id"]: "accepted"})
