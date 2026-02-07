"""Tests for qortex MCP server tools.

Tests the _impl functions directly via create_server() with InMemoryBackend.
No MCP transport needed — we call the plain Python impl functions.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptNode
from qortex.mcp import server as mcp_server


# ---------------------------------------------------------------------------
# Test embedding model (deterministic, no external deps)
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
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None
    yield
    mcp_server._backend = None
    mcp_server._vector_index = None
    mcp_server._adapter = None
    mcp_server._embedding_model = None
    mcp_server._llm_backend = None


@pytest.fixture
def vector_index():
    """In-memory vector index for testing."""
    from qortex.vec.index import NumpyVectorIndex

    return NumpyVectorIndex(dimensions=DIMS)


@pytest.fixture
def backend(vector_index) -> InMemoryBackend:
    """Backend wired to vector_index for dual-write on add_embedding."""
    b = InMemoryBackend(vector_index=vector_index)
    b.connect()
    return b


@pytest.fixture
def embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def configured_server(backend, embedding, vector_index):
    """Server with independent vec + graph layers configured."""
    mcp_server.create_server(
        backend=backend,
        embedding_model=embedding,
        vector_index=vector_index,
    )
    return mcp_server


def _make_node(node_id: str, name: str, desc: str, domain: str = "test") -> ConceptNode:
    return ConceptNode(
        id=node_id,
        name=name,
        description=desc,
        domain=domain,
        source_id="test-source",
    )


def _add_nodes_with_embeddings(backend, embedding, nodes: list[ConceptNode]) -> None:
    """Add nodes to backend and index their embeddings.

    InMemoryBackend.add_embedding() dual-writes to both internal storage
    and the vector_index (if configured), so VecOnlyAdapter can find them.
    """
    for node in nodes:
        backend.add_node(node)

    texts = [f"{n.name}: {n.description}" for n in nodes]
    embeddings = embedding.embed(texts)
    for node, emb in zip(nodes, embeddings):
        backend.add_embedding(node.id, emb)


# ===========================================================================
# qortex_status
# ===========================================================================


class TestQortexStatus:
    def test_status_returns_ok(self, configured_server):
        result = mcp_server._status_impl()
        assert result["status"] == "ok"

    def test_status_reports_backend_type(self, configured_server):
        result = mcp_server._status_impl()
        assert result["backend"] == "InMemoryBackend"

    def test_status_reports_vector_search_available(self, configured_server):
        result = mcp_server._status_impl()
        assert result["vector_search"] is True

    def test_status_reports_no_mage(self, configured_server):
        result = mcp_server._status_impl()
        assert result["graph_algorithms"] is False

    def test_status_reports_embedding_model(self, configured_server):
        result = mcp_server._status_impl()
        assert result["embedding_model"] == "FakeEmbedding"


# ===========================================================================
# qortex_domains
# ===========================================================================


class TestQortexDomains:
    def test_empty_domains(self, configured_server):
        result = mcp_server._domains_impl()
        assert result["domains"] == []

    def test_domains_after_adding_nodes(self, configured_server, backend, embedding):
        nodes = [_make_node("n1", "Auth", "Authentication system", domain="security")]
        _add_nodes_with_embeddings(backend, embedding, nodes)
        backend.create_domain("security", description="Security domain")

        result = mcp_server._domains_impl()
        assert len(result["domains"]) == 1
        assert result["domains"][0]["name"] == "security"

    def test_domains_reports_counts(self, configured_server, backend, embedding):
        backend.create_domain("test", description="Test domain")
        nodes = [
            _make_node("n1", "A", "desc A"),
            _make_node("n2", "B", "desc B"),
        ]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._domains_impl()
        domain = result["domains"][0]
        assert domain["concept_count"] == 2


# ===========================================================================
# qortex_query
# ===========================================================================


class TestQortexQuery:
    def test_query_empty_backend(self, configured_server):
        result = mcp_server._query_impl(context="test query")
        assert result["items"] == []
        assert result["query_id"] != ""

    def test_query_returns_results(self, configured_server, backend, embedding):
        nodes = [
            _make_node("n1", "Auth", "Authentication and login system"),
            _make_node("n2", "Database", "PostgreSQL database layer"),
            _make_node("n3", "API", "REST API endpoints"),
        ]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._query_impl(context="Authentication and login system")
        assert len(result["items"]) > 0
        assert result["query_id"] != ""

    def test_query_items_have_required_fields(self, configured_server, backend, embedding):
        nodes = [_make_node("n1", "Auth", "Authentication system")]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._query_impl(context="auth")
        item = result["items"][0]
        assert "id" in item
        assert "content" in item
        assert "score" in item
        assert "domain" in item
        assert "node_id" in item
        assert "metadata" in item

    def test_query_respects_top_k(self, configured_server, backend, embedding):
        nodes = [_make_node(f"n{i}", f"Node{i}", f"Description {i}") for i in range(10)]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._query_impl(context="description", top_k=3)
        assert len(result["items"]) <= 3

    def test_query_with_domain_filter(self, configured_server, backend, embedding):
        nodes = [
            _make_node("n1", "Auth", "Authentication", domain="security"),
            _make_node("n2", "DB", "Database", domain="infra"),
        ]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._query_impl(
            context="Authentication",
            domains=["security"],
        )
        for item in result["items"]:
            assert item["domain"] == "security"

    def test_query_scores_are_rounded(self, configured_server, backend, embedding):
        nodes = [_make_node("n1", "Auth", "Authentication")]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._query_impl(context="auth")
        if result["items"]:
            score = result["items"][0]["score"]
            # Score should be rounded to 4 decimal places
            assert score == round(score, 4)

    def test_query_clamps_top_k(self, configured_server, backend, embedding):
        """top_k <= 0 should be clamped to 1."""
        nodes = [_make_node("n1", "Auth", "Authentication")]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._query_impl(context="auth", top_k=0)
        assert len(result["items"]) <= 1

    def test_query_clamps_min_confidence(self, configured_server, backend, embedding):
        """min_confidence > 1 should be clamped to 1.0 (no results pass)."""
        nodes = [_make_node("n1", "Auth", "Authentication")]
        _add_nodes_with_embeddings(backend, embedding, nodes)

        result = mcp_server._query_impl(context="auth", min_confidence=2.0)
        assert result["items"] == []

    def test_query_without_adapter_returns_error(self):
        """When no embedding model is available, return error."""
        backend = InMemoryBackend()
        backend.connect()
        mcp_server.create_server(backend=backend, embedding_model=None)

        result = mcp_server._query_impl(context="test")
        assert "error" in result
        assert result["items"] == []


# ===========================================================================
# qortex_feedback
# ===========================================================================


class TestQortexFeedback:
    def test_feedback_returns_recorded(self, configured_server):
        result = mcp_server._feedback_impl(
            query_id="test-qid",
            outcomes={"item1": "accepted", "item2": "rejected"},
            source="test",
        )
        assert result["status"] == "recorded"
        assert result["outcome_count"] == 2
        assert result["source"] == "test"

    def test_feedback_with_default_source(self, configured_server):
        result = mcp_server._feedback_impl(
            query_id="test-qid",
            outcomes={"item1": "accepted"},
        )
        assert result["source"] == "unknown"

    def test_feedback_empty_outcomes(self, configured_server):
        result = mcp_server._feedback_impl(
            query_id="test-qid",
            outcomes={},
        )
        assert result["outcome_count"] == 0


# ===========================================================================
# qortex_ingest
# ===========================================================================


class TestQortexIngest:
    def test_ingest_nonexistent_file(self, configured_server):
        result = mcp_server._ingest_impl(
            source_path="/nonexistent/file.txt",
            domain="test",
        )
        assert "error" in result

    def test_ingest_text_file(self, configured_server):
        """Ingest a text file using StubLLMBackend."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is test content about authentication and security.")
            f.flush()
            path = f.name

        result = mcp_server._ingest_impl(source_path=path, domain="test")
        assert "error" not in result
        assert result["domain"] == "test"
        assert result["source"] == Path(path).name

        Path(path).unlink()

    def test_ingest_markdown_file(self, configured_server):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Test\n\nSome markdown content about APIs.\n")
            f.flush()
            path = f.name

        result = mcp_server._ingest_impl(source_path=path, domain="docs")
        assert "error" not in result
        assert result["domain"] == "docs"

        Path(path).unlink()

    def test_ingest_auto_detects_type(self, configured_server):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Markdown\n\nContent.\n")
            f.flush()
            path = f.name

        # source_type=None should auto-detect to "markdown"
        result = mcp_server._ingest_impl(source_path=path, domain="test")
        assert "error" not in result

        Path(path).unlink()

    def test_ingest_type_override(self, configured_server):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Plain text in unknown extension.\n")
            f.flush()
            path = f.name

        result = mcp_server._ingest_impl(
            source_path=path, domain="test", source_type="text"
        )
        assert "error" not in result

        Path(path).unlink()

    def test_ingest_pdf_returns_error(self, configured_server):
        """PDF ingest raises NotImplementedError, which should surface as error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("fake pdf content")
            f.flush()
            path = f.name

        with pytest.raises(NotImplementedError):
            mcp_server._ingest_impl(source_path=path, domain="test")

        Path(path).unlink()

    def test_ingest_invalid_source_type(self, configured_server):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            f.flush()
            path = f.name

        result = mcp_server._ingest_impl(
            source_path=path, domain="test", source_type="invalid"
        )
        assert "error" in result
        assert "Invalid source_type" in result["error"]

        Path(path).unlink()

    def test_ingest_directory_returns_error(self, configured_server):
        """Directories should not be accepted."""
        result = mcp_server._ingest_impl(source_path="/tmp", domain="test")
        assert "error" in result
        assert "Not a file" in result["error"]

    def test_ingest_with_custom_llm(self, configured_server):
        """StubLLMBackend with injected concepts shows up in results."""
        from qortex_ingest.base import StubLLMBackend

        llm = StubLLMBackend(
            concepts=[
                {"name": "TestConcept", "description": "A test concept", "confidence": 0.9}
            ],
            rules=[{"text": "Always test your code", "confidence": 1.0}],
        )
        mcp_server.set_llm_backend(llm)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content about testing.\n")
            f.flush()
            path = f.name

        result = mcp_server._ingest_impl(source_path=path, domain="testing")
        assert result["concepts"] == 1
        assert result["rules"] == 1

        Path(path).unlink()


# ===========================================================================
# End-to-end: ingest → query roundtrip
# ===========================================================================


class TestIngestQueryRoundtrip:
    def test_ingest_then_query_returns_results(self, configured_server, backend, embedding):
        """Full roundtrip: ingest a file, then query for its content."""
        from qortex_ingest.base import StubLLMBackend

        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "Authentication",
                    "description": "User authentication via OAuth2",
                    "confidence": 1.0,
                },
                {
                    "name": "Authorization",
                    "description": "Role-based access control",
                    "confidence": 0.9,
                },
            ],
        )
        mcp_server.set_llm_backend(llm)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("OAuth2 authentication and role-based access control.\n")
            f.flush()
            path = f.name

        # Ingest
        ingest_result = mcp_server._ingest_impl(source_path=path, domain="security")
        assert ingest_result["concepts"] == 2

        # Query
        query_result = mcp_server._query_impl(
            context="User authentication via OAuth2",
            domains=["security"],
        )
        assert len(query_result["items"]) > 0
        # The best match should be the auth concept
        ids = [item["id"] for item in query_result["items"]]
        assert "security:Authentication" in ids

        Path(path).unlink()

    def test_ingest_then_query_with_feedback(self, configured_server, backend, embedding):
        """Full loop: ingest → query → feedback."""
        from qortex_ingest.base import StubLLMBackend

        llm = StubLLMBackend(
            concepts=[{"name": "TestNode", "description": "Test description"}],
        )
        mcp_server.set_llm_backend(llm)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.\n")
            f.flush()
            path = f.name

        mcp_server._ingest_impl(source_path=path, domain="test")
        query_result = mcp_server._query_impl(context="Test description")
        query_id = query_result["query_id"]

        feedback_result = mcp_server._feedback_impl(
            query_id=query_id,
            outcomes={"test:TestNode": "accepted"},
            source="test-harness",
        )
        assert feedback_result["status"] == "recorded"

        Path(path).unlink()


# ===========================================================================
# create_server + set_llm_backend
# ===========================================================================


class TestServerConfiguration:
    def test_create_server_returns_mcp_instance(self, backend, embedding, vector_index):
        result = mcp_server.create_server(
            backend=backend, embedding_model=embedding, vector_index=vector_index
        )
        assert result is mcp_server.mcp

    def test_create_server_without_embedding(self, backend):
        mcp_server.create_server(backend=backend, embedding_model=None)
        assert mcp_server._adapter is None
        assert mcp_server._vector_index is None

    def test_create_server_auto_creates_vector_index(self, backend, embedding):
        """When no vector_index is provided but embedding_model is, auto-create one."""
        mcp_server.create_server(backend=backend, embedding_model=embedding)
        assert mcp_server._vector_index is not None
        assert mcp_server._adapter is not None

    def test_create_server_with_explicit_vector_index(self, backend, embedding, vector_index):
        """Explicit vector_index is used instead of auto-created one."""
        mcp_server.create_server(
            backend=backend, embedding_model=embedding, vector_index=vector_index
        )
        assert mcp_server._vector_index is vector_index

    def test_set_llm_backend(self, configured_server):
        from qortex_ingest.base import StubLLMBackend

        llm = StubLLMBackend()
        mcp_server.set_llm_backend(llm)
        assert mcp_server._llm_backend is llm
