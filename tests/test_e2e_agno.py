"""E2E: qortex as an Agno knowledge source with real embeddings.

This is not a unit test. This boots real infrastructure:
- Real embedding model (sentence-transformers/all-MiniLM-L6-v2, 384 dims)
- Real VectorIndex (NumpyVectorIndex)
- Real GraphBackend (InMemoryBackend)
- Real document ingestion via MCP server _impl functions
- Real semantic search with relevance verification

The E2E proves two things:
1. Shallow: QortexKnowledge satisfies Agno's KnowledgeProtocol (retrieve, build_context)
2. Deep preview: qortex results map cleanly to Agno's Document dataclass

The deep integration (QortexVectorDb implementing Agno's VectorDb ABC) is tracked
separately. This test proves the retrieval path works end-to-end with real data.

Requires: sentence-transformers (pip install sentence-transformers)
"""

from __future__ import annotations

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
def real_backend():
    """Boot real everything — same pattern as test_e2e_mastra.py.

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

    # Boot MCP server (sets module-level state for _impl functions)
    create_server(
        backend=backend,
        embedding_model=embedding,
        vector_index=vector_index,
    )

    set_llm_backend(
        StubLLMBackend(
            concepts=[
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
            ]
        )
    )

    return {
        "backend": backend,
        "embedding": embedding,
        "vector_index": vector_index,
    }


@pytest.fixture(scope="module")
def ingested_domain(real_backend, tmp_path_factory):
    """Ingest a real document into the backend."""
    from qortex.mcp.server import _ingest_impl

    doc_path = tmp_path_factory.mktemp("docs") / "auth_guide.txt"
    doc_path.write_text(
        textwrap.dedent("""\
        Authentication and Authorization Best Practices

        OAuth2 provides delegated authorization for web applications.
        It uses access tokens and refresh tokens to manage sessions securely.
        JWT (JSON Web Tokens) are commonly used as the token format in OAuth2 flows.

        Role-based access control (RBAC) should be implemented at the API gateway level.
        Each endpoint should declare required roles and permissions.
        Multi-factor authentication (MFA) adds an additional security layer beyond passwords.

        When implementing authentication, always use HTTPS, validate all tokens server-side,
        and implement proper token rotation and revocation mechanisms.
    """)
    )

    result = _ingest_impl(str(doc_path), "security")
    assert "error" not in result, f"Ingest failed: {result}"
    assert result["concepts"] >= 1
    return result


@pytest.fixture(scope="module")
def agno_knowledge(real_backend, ingested_domain):
    """Create QortexKnowledge backed by real embeddings."""
    from qortex.adapters.agno import QortexKnowledge
    from qortex.client import LocalQortexClient

    client = LocalQortexClient(
        vector_index=real_backend["vector_index"],
        backend=real_backend["backend"],
        embedding_model=real_backend["embedding"],
    )
    return QortexKnowledge(client=client, domains=["security"], top_k=5)


# ===========================================================================
# Agno KnowledgeProtocol: retrieve()
# ===========================================================================


class TestAgnoRetrieve:
    """QortexKnowledge.retrieve(query) → list of Agno Document-shaped results.

    This is the core method Agno agents call via search_knowledge_base tool.
    """

    def test_retrieve_returns_results_with_real_embeddings(self, agno_knowledge):
        docs = agno_knowledge.retrieve("How does OAuth2 authentication work?")

        assert len(docs) > 0

    def test_retrieve_returns_agno_document_shape(self, agno_knowledge):
        """Every result must have Agno's Document fields."""
        docs = agno_knowledge.retrieve("What is role-based access control?")

        assert len(docs) > 0
        required_fields = {"content", "id", "name", "meta_data", "reranking_score"}

        for doc in docs:
            if isinstance(doc, dict):
                assert required_fields.issubset(set(doc.keys())), (
                    f"Missing fields: {required_fields - set(doc.keys())}"
                )
                assert isinstance(doc["content"], str)
                assert len(doc["content"]) > 0
                assert isinstance(doc["meta_data"], dict)
                assert isinstance(doc["reranking_score"], float)
            else:
                # If agno is installed, we get real Document instances
                assert hasattr(doc, "content")
                assert hasattr(doc, "id")
                assert hasattr(doc, "reranking_score")
                assert len(doc.content) > 0

    def test_semantic_relevance_with_real_embeddings(self, agno_knowledge):
        """Real embeddings should return semantically relevant results."""
        docs = agno_knowledge.retrieve("OAuth2 token-based authentication")

        assert len(docs) > 0

        # Top result should be semantically related to OAuth2
        top = docs[0]
        content = top["content"].lower() if isinstance(top, dict) else top.content.lower()
        assert any(term in content for term in ["oauth", "auth", "token", "access"]), (
            f"Top result not relevant: {content[:200]}"
        )

    def test_retrieve_respects_domain_filter(self, agno_knowledge):
        """QortexKnowledge is initialized with domains=["security"].
        Results should only come from that domain."""
        docs = agno_knowledge.retrieve("authentication")

        for doc in docs:
            meta = doc["meta_data"] if isinstance(doc, dict) else doc.meta_data
            assert meta.get("domain") == "security"

    def test_scores_are_between_zero_and_one(self, agno_knowledge):
        docs = agno_knowledge.retrieve("multi-factor authentication")

        for doc in docs:
            score = doc["reranking_score"] if isinstance(doc, dict) else doc.reranking_score
            assert 0.0 <= score <= 1.0


# ===========================================================================
# Agno KnowledgeProtocol: build_context()
# ===========================================================================


class TestAgnoBuildContext:
    """build_context() returns a string for prompt injection.

    Agno uses this to inject knowledge into the agent's system prompt
    when add_knowledge_to_context=True.
    """

    def test_build_context_returns_string(self, agno_knowledge):
        """build_context() now returns instructions (KnowledgeProtocol)."""
        context = agno_knowledge.build_context()

        assert isinstance(context, str)
        assert len(context) > 0
        assert "search_knowledge_base" in context

    def test_build_context_includes_tool_instructions(self, agno_knowledge):
        """build_context includes instructions for all three tools."""
        context = agno_knowledge.build_context()
        context_lower = context.lower()
        assert "search" in context_lower
        assert "explore" in context_lower
        assert "feedback" in context_lower

    def test_build_context_includes_domain(self):
        """build_context mentions configured domains."""
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(
            client=self._make_client(),
            domains=["security", "auth"],
        )
        context = knowledge.build_context()
        assert "security" in context
        assert "auth" in context

    def _make_client(self):
        """Helper to create a test client."""
        from qortex.client import LocalQortexClient
        from qortex.core.memory import InMemoryBackend
        from qortex.vec.embeddings import SentenceTransformerEmbedding
        from qortex.vec.index import NumpyVectorIndex

        vi = NumpyVectorIndex(dimensions=384)
        backend = InMemoryBackend(vector_index=vi)
        backend.connect()
        emb = SentenceTransformerEmbedding()
        return LocalQortexClient(vector_index=vi, backend=backend, embedding_model=emb)


# ===========================================================================
# Agno Document construction from qortex QueryItem
# ===========================================================================


class TestAgnoDocumentMapping:
    """Verify QueryItem.to_agno_document() produces valid Agno Documents.

    If agno is installed, this should return real Document instances.
    If not, it returns dicts with the same shape.
    """

    def test_query_item_to_agno_document(self):
        """Direct conversion test — no retrieval needed."""
        from qortex.client import QueryItem

        item = QueryItem(
            id="security:oauth2",
            content="OAuth2 is an authorization framework for HTTP services",
            score=0.95,
            domain="security",
            node_id="security:oauth2",
            metadata={"source_id": "docs/auth.md"},
        )

        doc = item.to_agno_document()

        if isinstance(doc, dict):
            assert doc["content"] == item.content
            assert doc["id"] == item.id
            assert doc["name"] == item.node_id
            assert doc["reranking_score"] == item.score
            assert doc["meta_data"]["domain"] == "security"
            assert doc["meta_data"]["source_id"] == "docs/auth.md"
        else:
            # Real agno Document
            assert doc.content == item.content
            assert doc.id == item.id
            assert doc.reranking_score == item.score

    def test_agno_document_can_construct_real_instance(self, agno_knowledge):
        """If agno source is available, verify our dicts construct real Documents."""
        import importlib.util
        import sys
        import types
        from pathlib import Path

        agno_src = Path("/Users/peleke/Documents/Projects/agno/libs/agno")
        doc_path = agno_src / "agno" / "knowledge" / "document" / "base.py"
        if not doc_path.exists():
            pytest.skip("agno source not found")

        # Load Document class from source (same as test_framework_compat)
        stubbed: list[str] = []
        for mod_name in ("agno", "agno.knowledge", "agno.knowledge.embedder"):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
                stubbed.append(mod_name)
        sys.modules["agno.knowledge.embedder"].Embedder = None  # type: ignore[attr-defined]

        try:
            spec = importlib.util.spec_from_file_location("agno_doc", doc_path)
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            AgnoDocument = mod.Document

            # Retrieve and construct real Documents
            docs = agno_knowledge.retrieve("OAuth2 authentication")
            assert len(docs) > 0

            for doc_dict in docs:
                if isinstance(doc_dict, dict):
                    agno_doc = AgnoDocument(
                        content=doc_dict["content"],
                        id=doc_dict.get("id"),
                        name=doc_dict.get("name"),
                        meta_data=doc_dict.get("meta_data", {}),
                        reranking_score=doc_dict.get("reranking_score"),
                    )
                    assert agno_doc.content == doc_dict["content"]
                    assert agno_doc.id == doc_dict["id"]
                    assert agno_doc.reranking_score == doc_dict["reranking_score"]
        finally:
            for mod_name in stubbed:
                sys.modules.pop(mod_name, None)


# ===========================================================================
# Feedback: the thing Agno deferred to Phase 2
# ===========================================================================


class TestAgnoFeedbackUpgrade:
    """Agno's FeedbackConfig and SelfImprovementConfig are Phase 2/3 stubs.

    qortex has this NOW. A qortex-backed Agno agent gets feedback-driven
    learning for free — before Agno ships their own implementation.
    """

    def test_full_retrieve_feedback_cycle(self, agno_knowledge):
        # 1. Retrieve (same as Agno agent would via search_knowledge_base)
        docs = agno_knowledge.retrieve("authentication best practices")
        assert len(docs) > 0
        assert agno_knowledge.last_query_id is not None

        # 2. Feedback (Agno can't do this yet — FeedbackConfig is a stub)
        first_id = docs[0]["id"] if isinstance(docs[0], dict) else docs[0].id
        agno_knowledge.feedback({first_id: "accepted"})
        # No error. Feedback recorded. Teleportation factors update.

    def test_feedback_without_prior_retrieve_is_safe(self, agno_knowledge):
        """If no retrieve happened, feedback is a no-op."""
        from qortex.adapters.agno import QortexKnowledge
        from qortex.client import LocalQortexClient

        # Fresh knowledge with no prior query
        client = LocalQortexClient(
            vector_index=agno_knowledge._client._vector_index,
            backend=agno_knowledge._client._backend,
            embedding_model=agno_knowledge._client._embedding_model,
        )
        fresh = QortexKnowledge(client=client, domains=["security"])
        assert fresh.last_query_id is None

        # feedback with no prior query — safe no-op
        fresh.feedback({"some:id": "accepted"})

    def test_multiple_feedback_cycles(self, agno_knowledge):
        """Multiple retrieve → feedback cycles with different queries."""
        # Cycle 1
        docs1 = agno_knowledge.retrieve("OAuth2 tokens")
        qid1 = agno_knowledge.last_query_id
        agno_knowledge.feedback(
            {(docs1[0]["id"] if isinstance(docs1[0], dict) else docs1[0].id): "accepted"}
        )

        # Cycle 2
        docs2 = agno_knowledge.retrieve("role-based access control")
        qid2 = agno_knowledge.last_query_id
        agno_knowledge.feedback(
            {(docs2[0]["id"] if isinstance(docs2[0], dict) else docs2[0].id): "accepted"}
        )

        # Different queries → different query IDs
        assert qid1 != qid2


# ===========================================================================
# Full Agno consumer simulation
# ===========================================================================


class TestAgnoE2ESimulation:
    """Simulate what an Agno agent would do end-to-end.

    1. Initialize knowledge source
    2. build_context() for system prompt
    3. retrieve() for search_knowledge_base tool
    4. Verify Agno Document shapes
    5. Feedback (the upgrade Agno doesn't have yet)

    Every step uses real embeddings and real data.
    """

    def test_full_agent_workflow(self, real_backend, ingested_domain):
        from qortex.adapters.agno import QortexKnowledge
        from qortex.client import LocalQortexClient

        # --- Step 1: Agent initialization ---
        client = LocalQortexClient(
            vector_index=real_backend["vector_index"],
            backend=real_backend["backend"],
            embedding_model=real_backend["embedding"],
        )
        knowledge = QortexKnowledge(client=client, domains=["security"], top_k=4)

        # --- Step 2: build_context (returns instructions per KnowledgeProtocol) ---
        context = knowledge.build_context()
        assert isinstance(context, str)
        assert "search_knowledge_base" in context

        # --- Step 3: retrieve (search_knowledge_base tool) ---
        docs = knowledge.retrieve(
            "How should I implement token-based authentication?",
        )
        assert len(docs) > 0

        # --- Step 4: Verify Agno Document shape ---
        for doc in docs:
            if isinstance(doc, dict):
                assert "content" in doc
                assert "id" in doc
                assert "reranking_score" in doc
                assert isinstance(doc["content"], str)
                assert doc["reranking_score"] > 0.0
            else:
                assert hasattr(doc, "content")
                assert doc.reranking_score > 0.0

        # Semantic relevance with real embeddings
        all_content = " ".join(
            (d["content"] if isinstance(d, dict) else d.content).lower() for d in docs
        )
        assert any(term in all_content for term in ["auth", "token", "oauth", "jwt", "access"]), (
            f"Results not semantically relevant: {all_content[:200]}"
        )

        # --- Step 5: Feedback (Agno Phase 2 — we have it NOW) ---
        assert knowledge.last_query_id is not None
        first_id = docs[0]["id"] if isinstance(docs[0], dict) else docs[0].id
        knowledge.feedback({first_id: "accepted"})

        # --- Step 6: Another retrieve (in Level 2, this improves) ---
        docs2 = knowledge.retrieve(
            "What roles and permissions should API endpoints require?",
        )
        assert len(docs2) > 0
        assert knowledge.last_query_id is not None

    def test_cross_adapter_consistency(self, real_backend, ingested_domain):
        """Same backend, Agno + Mastra adapters return same results."""
        from qortex.adapters.agno import QortexKnowledge
        from qortex.adapters.mastra import QortexVectorStore
        from qortex.client import LocalQortexClient

        client = LocalQortexClient(
            vector_index=real_backend["vector_index"],
            backend=real_backend["backend"],
            embedding_model=real_backend["embedding"],
        )

        query = "OAuth2 token authentication"

        # Agno
        agno_docs = QortexKnowledge(
            client=client,
            domains=["security"],
            top_k=3,
        ).retrieve(query)

        # Mastra
        mastra_results = QortexVectorStore(client=client).query(
            index_name="security",
            query_text=query,
            top_k=3,
        )

        # Same IDs, same order
        agno_ids = [d["id"] if isinstance(d, dict) else d.id for d in agno_docs]
        mastra_ids = [r["id"] for r in mastra_results]
        assert agno_ids == mastra_ids

        # Same scores
        agno_scores = [
            d["reranking_score"] if isinstance(d, dict) else d.reranking_score for d in agno_docs
        ]
        mastra_scores = [r["score"] for r in mastra_results]
        assert agno_scores == mastra_scores
