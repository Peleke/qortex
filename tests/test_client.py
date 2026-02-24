"""Exhaustive tests for QortexClient + framework adapters.

Tests cover:
- QortexClient protocol conformance
- LocalQortexClient: query, feedback, ingest, domains, status
- QueryItem conversion helpers (langchain, crewai, agno, mastra)
- Result type dataclass contracts
- QortexKnowledgeStorage (crewai adapter)
- QortexKnowledge (agno adapter)
- QortexVectorStore (mastra adapter)
- QortexRetriever (langchain adapter, conditional on langchain-core)
- Full roundtrip: ingest → query → feedback through each adapter
"""

from __future__ import annotations

import hashlib

import pytest

from qortex.client import (
    DomainInfo,
    FeedbackResult,
    IngestResult,
    LocalQortexClient,
    QortexClient,
    QueryItem,
    QueryResult,
    StatusResult,
)
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptNode
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


def make_client(with_data: bool = False) -> LocalQortexClient:
    """Create a LocalQortexClient with optional pre-populated data."""
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeEmbedding()

    client = LocalQortexClient(
        vector_index=vector_index,
        backend=backend,
        embedding_model=embedding,
    )

    if with_data:
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
                id="security:RBAC",
                name="RBAC",
                description="Role-based access control",
                domain="security",
                source_id="test",
            ),
            ConceptNode(
                id="security:JWT",
                name="JWT",
                description="JSON Web Tokens for session management",
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

    return client


# ===========================================================================
# Result type contracts
# ===========================================================================


class TestQueryItem:
    def test_fields(self):
        item = QueryItem(
            id="test:foo",
            content="Foo concept",
            score=0.95,
            domain="test",
            node_id="test:foo",
            metadata={"key": "val"},
        )
        assert item.id == "test:foo"
        assert item.content == "Foo concept"
        assert item.score == 0.95
        assert item.domain == "test"
        assert item.node_id == "test:foo"
        assert item.metadata == {"key": "val"}

    def test_default_metadata(self):
        item = QueryItem(id="x", content="y", score=0.5, domain="d", node_id="x")
        assert item.metadata == {}

    def test_to_crewai_result(self):
        item = QueryItem(
            id="test:foo",
            content="Foo",
            score=0.9,
            domain="test",
            node_id="test:foo",
            metadata={"extra": 1},
        )
        result = item.to_crewai_result()
        assert result["id"] == "test:foo"
        assert result["content"] == "Foo"
        assert result["score"] == 0.9
        assert result["metadata"]["domain"] == "test"
        assert result["metadata"]["node_id"] == "test:foo"
        assert result["metadata"]["extra"] == 1

    def test_to_mastra_result(self):
        item = QueryItem(
            id="test:foo",
            content="Foo",
            score=0.9,
            domain="test",
            node_id="test:foo",
        )
        result = item.to_mastra_result()
        assert result["id"] == "test:foo"
        assert result["score"] == 0.9
        assert result["document"] == "Foo"
        assert result["metadata"]["domain"] == "test"

    def test_to_agno_document(self):
        """Returns agno Document (real or dict fallback) with correct fields."""
        item = QueryItem(
            id="test:foo",
            content="Foo",
            score=0.9,
            domain="test",
            node_id="test:foo",
        )
        result = item.to_agno_document()
        # Works with both real agno.Document and dict fallback
        if isinstance(result, dict):
            assert result["content"] == "Foo"
            assert result["id"] == "test:foo"
            assert result["name"] == "test:foo"
            assert result["reranking_score"] == 0.9
            assert result["meta_data"]["domain"] == "test"
        else:
            assert result.content == "Foo"
            assert result.id == "test:foo"
            assert result.name == "test:foo"
            assert result.reranking_score == 0.9
            assert result.meta_data["domain"] == "test"

    def test_to_langchain_document_fallback(self):
        """Without langchain installed, returns a dict with Document shape."""
        item = QueryItem(
            id="test:foo",
            content="Foo",
            score=0.9,
            domain="test",
            node_id="test:foo",
        )
        result = item.to_langchain_document()
        # Might be a real Document or a dict depending on langchain availability
        if isinstance(result, dict):
            assert result["page_content"] == "Foo"
            assert result["id"] == "test:foo"
            assert result["metadata"]["score"] == 0.9
        else:
            # langchain is installed
            assert result.page_content == "Foo"
            assert result.id == "test:foo"
            assert result.metadata["score"] == 0.9


class TestQueryResult:
    def test_fields(self):
        result = QueryResult(items=[], query_id="qid-123")
        assert result.items == []
        assert result.query_id == "qid-123"

    def test_with_items(self):
        items = [QueryItem(id="a", content="A", score=0.9, domain="d", node_id="a")]
        result = QueryResult(items=items, query_id="q1")
        assert len(result.items) == 1
        assert result.items[0].id == "a"


class TestFeedbackResult:
    def test_fields(self):
        r = FeedbackResult(status="recorded", query_id="q1", outcome_count=2, source="test")
        assert r.status == "recorded"
        assert r.outcome_count == 2


class TestIngestResult:
    def test_fields(self):
        r = IngestResult(domain="test", source="file.txt", concepts=3, edges=1, rules=0)
        assert r.domain == "test"
        assert r.concepts == 3

    def test_default_warnings(self):
        r = IngestResult(domain="test", source="f.txt", concepts=0, edges=0, rules=0)
        assert r.warnings == []


class TestDomainInfo:
    def test_fields(self):
        d = DomainInfo(name="security", description="Auth domain", concept_count=5)
        assert d.name == "security"
        assert d.concept_count == 5

    def test_defaults(self):
        d = DomainInfo(name="x")
        assert d.concept_count == 0
        assert d.edge_count == 0
        assert d.rule_count == 0


class TestStatusResult:
    def test_fields(self):
        s = StatusResult(status="ok", backend="InMemoryBackend")
        assert s.status == "ok"
        assert s.vector_search is False


# ===========================================================================
# QortexClient protocol conformance
# ===========================================================================


class TestProtocolConformance:
    def test_local_client_implements_protocol(self):
        client = make_client()
        assert isinstance(client, QortexClient)

    def test_protocol_methods_exist(self):
        client = make_client()
        assert callable(getattr(client, "query", None))
        assert callable(getattr(client, "feedback", None))
        assert callable(getattr(client, "ingest", None))
        assert callable(getattr(client, "domains", None))
        assert callable(getattr(client, "status", None))


# ===========================================================================
# LocalQortexClient
# ===========================================================================


class TestLocalQortexClientQuery:
    def test_query_empty(self):
        client = make_client()
        result = client.query("anything")
        assert result.items == []
        assert result.query_id != ""

    def test_query_returns_results(self):
        client = make_client(with_data=True)
        result = client.query("Authentication via OAuth2")
        assert len(result.items) > 0
        assert result.query_id != ""

    def test_query_items_have_correct_type(self):
        client = make_client(with_data=True)
        result = client.query("auth")
        for item in result.items:
            assert isinstance(item, QueryItem)

    def test_query_scores_are_rounded(self):
        client = make_client(with_data=True)
        result = client.query("auth")
        for item in result.items:
            assert item.score == round(item.score, 4)

    def test_query_respects_top_k(self):
        client = make_client(with_data=True)
        result = client.query("security", top_k=1)
        assert len(result.items) <= 1

    def test_query_respects_domain_filter(self):
        client = make_client(with_data=True)
        result = client.query("auth", domains=["security"])
        for item in result.items:
            assert item.domain == "security"

    def test_query_clamps_top_k_zero(self):
        client = make_client(with_data=True)
        result = client.query("auth", top_k=0)
        assert len(result.items) <= 1  # clamped to 1

    def test_query_clamps_min_confidence_over_one(self):
        client = make_client(with_data=True)
        result = client.query("auth", min_confidence=2.0)
        assert result.items == []  # clamped to 1.0, nothing passes

    def test_query_without_adapter(self):
        backend = InMemoryBackend()
        backend.connect()
        client = LocalQortexClient(vector_index=None, backend=backend)
        result = client.query("test")
        assert result.items == []
        assert result.query_id != ""


class TestLocalQortexClientFeedback:
    def test_feedback_returns_result(self):
        client = make_client()
        result = client.feedback("q1", {"item1": "accepted"}, source="test")
        assert result.status == "recorded"
        assert result.outcome_count == 1
        assert result.source == "test"

    def test_feedback_default_source(self):
        client = make_client()
        result = client.feedback("q1", {})
        assert result.source == "unknown"


class TestLocalQortexClientIngest:
    def test_ingest_text_file(self, tmp_path):
        client = make_client()
        p = tmp_path / "test.txt"
        p.write_text("Authentication via OAuth2 protocol.\n")

        result = client.ingest(str(p), domain="security")
        assert result.domain == "security"
        assert result.source == "test.txt"

    def test_ingest_nonexistent_file(self):
        client = make_client()
        with pytest.raises(FileNotFoundError):
            client.ingest("/nonexistent/file.txt", domain="test")

    def test_ingest_directory_raises(self):
        client = make_client()
        with pytest.raises(ValueError, match="Not a file"):
            client.ingest("/tmp", domain="test")

    def test_ingest_invalid_source_type(self, tmp_path):
        client = make_client()
        p = tmp_path / "test.txt"
        p.write_text("test")

        with pytest.raises(ValueError, match="Invalid source_type"):
            client.ingest(str(p), domain="test", source_type="invalid")


class TestLocalQortexClientDomains:
    def test_empty_domains(self):
        client = make_client()
        domains = client.domains()
        assert domains == []

    def test_domains_after_data(self):
        client = make_client(with_data=True)
        domains = client.domains()
        assert len(domains) >= 1
        assert any(d.name == "security" for d in domains)

    def test_domain_info_type(self):
        client = make_client(with_data=True)
        domains = client.domains()
        for d in domains:
            assert isinstance(d, DomainInfo)


class TestLocalQortexClientStatus:
    def test_status_ok(self):
        client = make_client()
        status = client.status()
        assert status.status == "ok"
        assert status.backend == "InMemoryBackend"
        assert status.vector_search is True
        assert status.vector_index == "NumpyVectorIndex"

    def test_status_without_vec(self):
        backend = InMemoryBackend()
        backend.connect()
        client = LocalQortexClient(vector_index=None, backend=backend)
        status = client.status()
        assert status.vector_search is False
        assert status.vector_index is None


# ===========================================================================
# Full roundtrip: ingest → query → feedback
# ===========================================================================


class TestLocalQortexClientRoundtrip:
    def test_ingest_then_query(self, tmp_path):
        from qortex.ingest.base import StubLLMBackend

        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "OAuth2",
                    "description": "OAuth2 authentication protocol",
                    "confidence": 1.0,
                },
                {"name": "RBAC", "description": "Role-based access control", "confidence": 0.9},
            ],
        )

        vector_index = NumpyVectorIndex(dimensions=DIMS)
        backend = InMemoryBackend(vector_index=vector_index)
        backend.connect()
        embedding = FakeEmbedding()

        client = LocalQortexClient(
            vector_index=vector_index,
            backend=backend,
            embedding_model=embedding,
            llm_backend=llm,
        )

        p = tmp_path / "test.txt"
        p.write_text("OAuth2 and role-based access control.\n")

        ingest_result = client.ingest(str(p), domain="security")
        assert ingest_result.concepts == 2

        query_result = client.query("OAuth2 authentication", domains=["security"])
        assert len(query_result.items) > 0

        feedback_result = client.feedback(
            query_result.query_id,
            {query_result.items[0].id: "accepted"},
            source="test",
        )
        assert feedback_result.status == "recorded"


# ===========================================================================
# CrewAI adapter
# ===========================================================================


class TestCrewAIAdapter:
    def test_search_returns_crewai_shape(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        client = make_client(with_data=True)
        storage = QortexKnowledgeStorage(client=client, domains=["security"])

        results = storage.search("authentication", limit=5)
        assert isinstance(results, list)
        for r in results:
            assert "id" in r
            assert "content" in r
            assert "score" in r
            assert "metadata" in r
            assert isinstance(r["metadata"], dict)
            assert "domain" in r["metadata"]

    def test_search_empty(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        client = make_client()
        storage = QortexKnowledgeStorage(client=client)
        results = storage.search("anything")
        assert results == []

    def test_search_respects_limit(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        client = make_client(with_data=True)
        storage = QortexKnowledgeStorage(client=client, domains=["security"])
        results = storage.search("security", limit=1)
        assert len(results) <= 1

    def test_search_respects_score_threshold(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        client = make_client(with_data=True)
        storage = QortexKnowledgeStorage(client=client, domains=["security"])
        results = storage.search("auth", score_threshold=0.99)
        # Very high threshold — may return 0 or only very close matches
        for r in results:
            assert r["score"] >= 0.99

    def test_save_is_noop(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=make_client())
        storage.save(["doc1", "doc2"])  # should not raise

    def test_reset_is_noop(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=make_client())
        storage.reset()  # should not raise

    def test_feedback_loop(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        client = make_client(with_data=True)
        storage = QortexKnowledgeStorage(client=client, domains=["security"])

        results = storage.search("authentication")
        assert len(results) > 0
        assert storage.last_query_id is not None

        storage.feedback({results[0]["id"]: "accepted"})  # should not raise

    def test_feedback_without_query_is_noop(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=make_client())
        storage.feedback({"item1": "accepted"})  # should not raise


# ===========================================================================
# Agno adapter
# ===========================================================================


class TestAgnoAdapter:
    def test_retrieve_returns_agno_shape(self):
        from qortex.adapters.agno import QortexKnowledge

        client = make_client(with_data=True)
        knowledge = QortexKnowledge(client=client, domains=["security"])

        docs = knowledge.retrieve("authentication")
        assert isinstance(docs, list)
        for doc in docs:
            # Works with dicts (no agno installed) or actual Documents
            if isinstance(doc, dict):
                assert "content" in doc
                assert "id" in doc
                assert "name" in doc
                assert "meta_data" in doc
                assert "reranking_score" in doc
            else:
                assert hasattr(doc, "content")

    def test_retrieve_empty(self):
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(client=make_client())
        docs = knowledge.retrieve("anything")
        assert docs == []

    def test_build_context(self):
        from qortex.adapters.agno import QortexKnowledge

        client = make_client(with_data=True)
        knowledge = QortexKnowledge(client=client, domains=["security"])

        context = knowledge.build_context()
        assert isinstance(context, str)
        assert "search_knowledge_base" in context
        assert "security" in context  # domain should appear

    def test_build_context_no_domains(self):
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(client=make_client())
        context = knowledge.build_context()
        assert isinstance(context, str)
        assert "search_knowledge_base" in context

    def test_get_tools(self):
        from qortex.adapters.agno import QortexKnowledge

        client = make_client(with_data=True)
        knowledge = QortexKnowledge(client=client, domains=["security"])

        tools = knowledge.get_tools()
        assert len(tools) == 3
        names = [t.__name__ for t in tools]
        assert "search_knowledge_base" in names
        assert "explore_knowledge_graph" in names
        assert "report_knowledge_feedback" in names

    def test_get_tools_filtered(self):
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(
            client=make_client(),
            enable_explore=False,
            enable_feedback=False,
        )
        tools = knowledge.get_tools()
        assert len(tools) == 1
        assert tools[0].__name__ == "search_knowledge_base"

    def test_feedback_loop(self):
        from qortex.adapters.agno import QortexKnowledge

        client = make_client(with_data=True)
        knowledge = QortexKnowledge(client=client, domains=["security"])

        docs = knowledge.retrieve("auth")
        assert knowledge.last_query_id is not None
        doc = docs[0]
        item_id = doc.id if hasattr(doc, "id") else doc.get("id", "")
        knowledge.feedback({item_id: "accepted"})  # should not raise

    def test_retrieve_with_kwargs(self):
        from qortex.adapters.agno import QortexKnowledge

        client = make_client(with_data=True)
        knowledge = QortexKnowledge(client=client, domains=["security"])

        docs = knowledge.retrieve("auth", top_k=1)
        assert len(docs) <= 1


# ===========================================================================
# Mastra adapter
# ===========================================================================


class TestMastraAdapter:
    def test_query_returns_mastra_shape(self):
        from qortex.adapters.mastra import QortexVectorStore

        client = make_client(with_data=True)
        store = QortexVectorStore(client=client)

        results = store.query(index_name="security", query_text="authentication")
        assert isinstance(results, list)
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "metadata" in r
            assert "document" in r
            assert isinstance(r["metadata"], dict)

    def test_query_empty(self):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=make_client())
        results = store.query(index_name="test", query_text="anything")
        assert results == []

    def test_query_requires_text_or_vector(self):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=make_client())
        with pytest.raises(ValueError, match="query_text or query_vector"):
            store.query(index_name="test")

    def test_query_vector_only_raises(self):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=make_client())
        with pytest.raises(NotImplementedError, match="query_vector without query_text"):
            store.query(index_name="test", query_vector=[1.0, 2.0])

    def test_list_indexes(self):
        from qortex.adapters.mastra import QortexVectorStore

        client = make_client(with_data=True)
        store = QortexVectorStore(client=client)

        indexes = store.list_indexes()
        assert isinstance(indexes, list)
        for idx in indexes:
            assert "name" in idx
            assert "dimension" in idx
            assert "metric" in idx
            assert "count" in idx
            assert idx["metric"] == "cosine"

    def test_describe_index(self):
        from qortex.adapters.mastra import QortexVectorStore

        client = make_client(with_data=True)
        store = QortexVectorStore(client=client)

        info = store.describe_index("security")
        assert info["name"] == "security"
        assert info["count"] >= 0

    def test_describe_index_not_found(self):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=make_client())
        with pytest.raises(ValueError, match="not found"):
            store.describe_index("nonexistent")

    def test_upsert_raises(self):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=make_client())
        with pytest.raises(NotImplementedError, match="file-based ingestion"):
            store.upsert("test", [{"id": "x", "vector": [1.0]}])

    def test_delete_index_raises(self):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=make_client())
        with pytest.raises(NotImplementedError):
            store.delete_index("test")

    def test_create_index_is_noop(self):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=make_client())
        store.create_index("test", dimension=384)  # should not raise

    def test_feedback_loop(self):
        from qortex.adapters.mastra import QortexVectorStore

        client = make_client(with_data=True)
        store = QortexVectorStore(client=client)

        results = store.query(index_name="security", query_text="auth")
        assert store.last_query_id is not None
        store.feedback({results[0]["id"]: "accepted"})  # should not raise


# ===========================================================================
# LangChain adapter (conditional)
# ===========================================================================


try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever as LCBaseRetriever

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
class TestLangChainAdapter:
    def test_is_base_retriever(self):
        from qortex.adapters.langchain import QortexRetriever

        client = make_client()
        retriever = QortexRetriever(client=client)
        assert isinstance(retriever, LCBaseRetriever)

    def test_invoke_returns_documents(self):
        from qortex.adapters.langchain import QortexRetriever

        client = make_client(with_data=True)
        retriever = QortexRetriever(client=client, domains=["security"])

        docs = retriever.invoke("authentication")
        assert isinstance(docs, list)
        for doc in docs:
            assert isinstance(doc, LCDocument)
            assert doc.page_content
            assert "score" in doc.metadata
            assert "domain" in doc.metadata

    def test_invoke_empty(self):
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=make_client())
        docs = retriever.invoke("anything")
        assert docs == []

    def test_invoke_respects_top_k(self):
        from qortex.adapters.langchain import QortexRetriever

        client = make_client(with_data=True)
        retriever = QortexRetriever(client=client, domains=["security"], top_k=1)
        docs = retriever.invoke("security")
        assert len(docs) <= 1

    def test_feedback_loop(self):
        from qortex.adapters.langchain import QortexRetriever

        client = make_client(with_data=True)
        retriever = QortexRetriever(client=client, domains=["security"])

        docs = retriever.invoke("auth")
        assert len(docs) > 0
        # Feedback with document IDs
        retriever.feedback({docs[0].id: "accepted"})

    def test_langchain_document_conversion(self):
        """QueryItem.to_langchain_document() returns actual Document."""
        item = QueryItem(
            id="test:foo",
            content="Foo content",
            score=0.9,
            domain="test",
            node_id="test:foo",
            metadata={"extra": 1},
        )
        doc = item.to_langchain_document()
        assert isinstance(doc, LCDocument)
        assert doc.page_content == "Foo content"
        assert doc.id == "test:foo"
        assert doc.metadata["score"] == 0.9
        assert doc.metadata["domain"] == "test"
        assert doc.metadata["extra"] == 1


# ===========================================================================
# Cross-adapter consistency
# ===========================================================================


class TestCrossAdapterConsistency:
    """All adapters should return consistent data for the same query."""

    def test_same_query_same_results(self):
        from qortex.adapters.agno import QortexKnowledge
        from qortex.adapters.crewai import QortexKnowledgeStorage
        from qortex.adapters.mastra import QortexVectorStore

        client = make_client(with_data=True)
        query = "Authentication via OAuth2"

        # Query through each adapter
        crewai_results = QortexKnowledgeStorage(client=client, domains=["security"]).search(query)
        agno_results = QortexKnowledge(client=client, domains=["security"]).retrieve(query)
        mastra_results = QortexVectorStore(client=client).query(
            index_name="security", query_text=query
        )

        # All should return the same number of results
        assert len(crewai_results) == len(agno_results) == len(mastra_results)

        # All should return the same IDs (in same order)
        crewai_ids = [r["id"] for r in crewai_results]
        agno_ids = [r["id"] if isinstance(r, dict) else r.id for r in agno_results]
        mastra_ids = [r["id"] for r in mastra_results]

        assert crewai_ids == agno_ids == mastra_ids

    def test_scores_consistent_across_adapters(self):
        from qortex.adapters.crewai import QortexKnowledgeStorage
        from qortex.adapters.mastra import QortexVectorStore

        client = make_client(with_data=True)
        query = "role-based access"

        crewai_results = QortexKnowledgeStorage(client=client, domains=["security"]).search(query)
        mastra_results = QortexVectorStore(client=client).query(
            index_name="security", query_text=query
        )

        if crewai_results and mastra_results:
            # Scores should be identical (same underlying query)
            assert crewai_results[0]["score"] == mastra_results[0]["score"]
