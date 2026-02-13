"""Framework compatibility: verify qortex adapters against ACTUAL framework source code.

This imports the real types from the local framework repos and verifies
qortex adapter output is structurally compatible. NOT mocks — these are
the actual TypedDict/dataclass definitions from crewAI, agno, and langchain.

Framework source repos:
    crewAI:    /Users/peleke/Documents/Projects/crewAI/
    agno:      /Users/peleke/Documents/Projects/agno/
    langchain: /Users/peleke/Documents/Projects/langchain/

If a framework repo is missing, tests for that framework are skipped.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from qortex.client import LocalQortexClient, QueryItem
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptNode
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Framework source paths
# ---------------------------------------------------------------------------

CREWAI_SRC = Path("/Users/peleke/Documents/Projects/crewAI/lib/crewai/src")
AGNO_SRC = Path("/Users/peleke/Documents/Projects/agno/libs/agno")
LANGCHAIN_SRC = Path("/Users/peleke/Documents/Projects/langchain/libs/core")

# ---------------------------------------------------------------------------
# Helpers to import framework types without full dependency chains
# ---------------------------------------------------------------------------


def _load_module_from_file(name: str, filepath: Path):
    """Import a single module by file path, bypassing __init__.py chains."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_crewai_search_result():
    """Load crewai's SearchResult TypedDict from source."""
    types_path = CREWAI_SRC / "crewai" / "rag" / "types.py"
    if not types_path.exists():
        return None
    mod = _load_module_from_file("crewai_rag_types", types_path)
    return getattr(mod, "SearchResult", None)


def _load_agno_document():
    """Load agno's Document dataclass from source.

    Temporarily stubs agno.knowledge.embedder in sys.modules so the Document
    module can import without the full agno dependency chain. Cleans up after.
    """
    doc_path = AGNO_SRC / "agno" / "knowledge" / "document" / "base.py"
    if not doc_path.exists():
        return None
    # Track which modules we stub so we can clean up
    stubbed: list[str] = []
    for mod_name in ("agno", "agno.knowledge", "agno.knowledge.embedder"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
            stubbed.append(mod_name)
    sys.modules["agno.knowledge.embedder"].Embedder = None  # type: ignore[attr-defined]
    try:
        mod = _load_module_from_file("agno_document", doc_path)
        return getattr(mod, "Document", None)
    finally:
        # Remove stubs we added — don't pollute sys.modules for other tests
        for mod_name in stubbed:
            sys.modules.pop(mod_name, None)


# Attempt to load framework types
CrewAISearchResult = _load_crewai_search_result()
AgnoDocument = _load_agno_document()

try:
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIMS = 32


class FakeEmbedding:
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


@pytest.fixture
def client():
    """Client with realistic data."""
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeEmbedding()

    backend.create_domain("security")
    nodes = [
        ConceptNode(
            id="security:oauth2",
            name="OAuth2",
            description="OAuth2 authorization framework for HTTP services",
            domain="security",
            source_id="docs/auth.md",
        ),
        ConceptNode(
            id="security:jwt",
            name="JWT",
            description="JSON Web Tokens for compact claims transfer",
            domain="security",
            source_id="docs/auth.md",
        ),
        ConceptNode(
            id="security:rbac",
            name="RBAC",
            description="Role-based access control for authorization",
            domain="security",
            source_id="docs/auth.md",
        ),
    ]
    for node in nodes:
        backend.add_node(node)
    texts = [f"{n.name}: {n.description}" for n in nodes]
    embeddings = embedding.embed(texts)
    for node, emb in zip(nodes, embeddings):
        backend.add_embedding(node.id, emb)

    return LocalQortexClient(
        vector_index=vector_index,
        backend=backend,
        embedding_model=embedding,
    )


# ===========================================================================
# CrewAI: Verify against actual SearchResult TypedDict
# ===========================================================================


@pytest.mark.skipif(
    CrewAISearchResult is None,
    reason=f"crewAI source not found at {CREWAI_SRC}",
)
class TestCrewAICompat:
    """Verify qortex crewai adapter output against the REAL SearchResult TypedDict."""

    def test_output_matches_search_result_keys(self, client):
        """Our output has exactly the keys crewai's SearchResult TypedDict requires."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=client, domains=["security"])
        results = storage.search(["authentication OAuth2"], limit=3)

        expected_keys = set(CrewAISearchResult.__annotations__.keys())
        # Expected: {id, content, metadata, score}

        assert len(results) > 0
        for r in results:
            assert set(r.keys()) == expected_keys, (
                f"Key mismatch: got {set(r.keys())}, expected {expected_keys}"
            )

    def test_output_matches_search_result_types(self, client):
        """Each field has the correct type per crewai's TypedDict."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=client, domains=["security"])
        results = storage.search(["auth"], limit=3)

        for r in results:
            assert isinstance(r["id"], str), f"id should be str, got {type(r['id'])}"
            assert isinstance(r["content"], str), f"content should be str, got {type(r['content'])}"
            assert isinstance(r["metadata"], dict), (
                f"metadata should be dict, got {type(r['metadata'])}"
            )
            assert isinstance(r["score"], float), f"score should be float, got {type(r['score'])}"

    def test_accepts_list_str_query(self, client):
        """crewai passes query as list[str]. Our adapter MUST accept this."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=client, domains=["security"])

        # Single-element list (most common)
        results_single = storage.search(["OAuth2 authentication"])
        assert isinstance(results_single, list)

        # Multi-element list (crewai joins with space)
        results_multi = storage.search(["OAuth2", "authentication", "protocol"])
        assert isinstance(results_multi, list)

    def test_accepts_str_query_for_convenience(self, client):
        """We also accept plain str for flexibility (not required by crewai)."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=client, domains=["security"])
        results = storage.search("OAuth2 authentication")
        assert isinstance(results, list)

    def test_score_threshold_matches_crewai_semantics(self, client):
        """crewai's score_threshold filters results below threshold."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=client, domains=["security"])

        # Default threshold is 0.6 in crewai
        results = storage.search(["auth"], score_threshold=0.6)
        for r in results:
            assert r["score"] >= 0.6

    def test_save_and_reset_dont_crash(self, client):
        """crewai expects save() and reset() to exist."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=client)
        storage.save(["document text 1", "document text 2"])
        storage.reset()


# ===========================================================================
# Agno: Verify against actual Document dataclass
# ===========================================================================


@pytest.mark.skipif(
    AgnoDocument is None,
    reason=f"agno source not found at {AGNO_SRC}",
)
class TestAgnoCompat:
    """Verify qortex agno adapter output against the REAL Document dataclass."""

    def test_dict_output_has_document_fields(self, client):
        """When agno is not installed, we return dicts. They must have Document's core fields."""
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(client=client, domains=["security"])
        docs = knowledge.retrieve("authentication")

        # Core fields agno's Document expects
        required_fields = {"content", "id", "name", "meta_data", "reranking_score"}

        assert len(docs) > 0
        for doc in docs:
            if isinstance(doc, dict):
                assert required_fields.issubset(set(doc.keys())), (
                    f"Missing fields: {required_fields - set(doc.keys())}"
                )
            else:
                for field in required_fields:
                    assert hasattr(doc, field), f"Missing attribute: {field}"

    def test_dict_output_types_match_document(self, client):
        """Field types match agno Document dataclass."""
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(client=client, domains=["security"])
        docs = knowledge.retrieve("auth")

        for doc in docs:
            if isinstance(doc, dict):
                assert isinstance(doc["content"], str)
                assert doc["id"] is None or isinstance(doc["id"], str)
                assert doc["name"] is None or isinstance(doc["name"], str)
                assert isinstance(doc["meta_data"], dict)
                assert doc["reranking_score"] is None or isinstance(
                    doc["reranking_score"], (int, float)
                )

    def test_can_construct_real_agno_document_from_output(self, client):
        """Our dict output can construct an actual agno Document instance."""
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(client=client, domains=["security"])
        docs = knowledge.retrieve("auth")

        assert len(docs) > 0
        for doc_dict in docs:
            if isinstance(doc_dict, dict):
                # Construct a real agno Document from our output
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

    def test_build_context_returns_instructions(self, client):
        """build_context returns tool instructions per KnowledgeProtocol."""
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(client=client, domains=["security"])
        context = knowledge.build_context()
        assert isinstance(context, str)
        assert "search_knowledge_base" in context
        assert "security" in context


# ===========================================================================
# LangChain: Verify against actual BaseRetriever + Document
# ===========================================================================


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
class TestLangChainCompat:
    """Verify qortex langchain adapter against the REAL BaseRetriever + Document."""

    def test_is_actual_base_retriever_subclass(self, client):
        """QortexRetriever IS a BaseRetriever — not a duck type, the real thing."""
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=client)
        assert isinstance(retriever, LangChainBaseRetriever)
        assert issubclass(type(retriever), LangChainBaseRetriever)

    def test_invoke_returns_real_documents(self, client):
        """invoke() returns actual langchain Document instances."""
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=client, domains=["security"])
        docs = retriever.invoke("OAuth2 authentication")

        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, LangChainDocument)
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0
            assert isinstance(doc.metadata, dict)
            assert doc.id is not None

    def test_works_in_lcel_chain(self, client):
        """QortexRetriever works in a real LCEL (LangChain Expression Language) chain.

        This simulates: retriever | format_docs — the first half of a RAG pipeline.
        The second half (| prompt | llm) would need an LLM, but the retrieval
        part is what we're proving.
        """
        from langchain_core.runnables import RunnableLambda

        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=client, domains=["security"], top_k=3)

        # This is REAL langchain LCEL — not a mock
        def format_docs(docs):
            return "\n---\n".join(
                f"[{doc.metadata.get('domain', '?')}] {doc.page_content}" for doc in docs
            )

        chain = retriever | RunnableLambda(format_docs)

        # Execute the chain exactly as langchain would
        result = chain.invoke("authentication protocol")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "---" in result or "security" in result.lower()

    def test_works_with_retriever_pipe_operator(self, client):
        """The | operator composes retrievers in langchain. Verify it works."""
        from langchain_core.runnables import RunnablePassthrough

        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=client, domains=["security"], top_k=2)

        # Build the context dict that a RAG prompt would consume
        chain = {
            "context": retriever,
            "question": RunnablePassthrough(),
        }

        # This is how langchain RAG chains work
        # (we can't complete the chain without an LLM, but the retrieval step works)
        from langchain_core.runnables import RunnableParallel

        parallel = RunnableParallel(chain)
        result = parallel.invoke("What is RBAC?")

        assert "context" in result
        assert "question" in result
        assert isinstance(result["context"], list)
        assert all(isinstance(doc, LangChainDocument) for doc in result["context"])
        assert result["question"] == "What is RBAC?"

    def test_document_metadata_has_score(self, client):
        """langchain convention: score goes in metadata (not a top-level field)."""
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=client, domains=["security"])
        docs = retriever.invoke("auth")

        for doc in docs:
            assert "score" in doc.metadata
            assert isinstance(doc.metadata["score"], float)
            assert 0.0 <= doc.metadata["score"] <= 1.0

    def test_query_item_to_langchain_document_roundtrip(self):
        """QueryItem.to_langchain_document() returns a real Document."""
        item = QueryItem(
            id="test:1",
            content="OAuth2 protocol",
            score=0.95,
            domain="security",
            node_id="test:1",
            metadata={"source": "docs"},
        )
        doc = item.to_langchain_document()

        assert isinstance(doc, LangChainDocument)
        assert doc.page_content == "OAuth2 protocol"
        assert doc.id == "test:1"
        assert doc.metadata["score"] == 0.95
        assert doc.metadata["domain"] == "security"
        assert doc.metadata["source"] == "docs"


# ===========================================================================
# Mastra: Verify shape against documented MastraVector interface
# ===========================================================================


class TestMastraCompat:
    """Verify qortex mastra adapter against Mastra's MastraVector interface.

    Mastra is TypeScript — we can't import it. But we verify against the
    documented interface shapes from the Mastra source code analysis.

    MastraVector.query() → QueryResult[]: {id, score, metadata, document}
    MastraVector.listIndexes() → IndexStats[]: {name, dimension, metric, count}
    """

    QUERY_RESULT_KEYS = {"id", "score", "metadata", "document"}
    INDEX_STATS_KEYS = {"name", "dimension", "metric", "count"}

    def test_query_returns_mastra_query_result_shape(self, client):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=client)
        results = store.query(index_name="security", query_text="auth", top_k=3)

        assert len(results) > 0
        for r in results:
            assert set(r.keys()) == self.QUERY_RESULT_KEYS, (
                f"Key mismatch: got {set(r.keys())}, expected {self.QUERY_RESULT_KEYS}"
            )
            assert isinstance(r["id"], str)
            assert isinstance(r["score"], (int, float))
            assert isinstance(r["metadata"], dict)
            assert isinstance(r["document"], str)  # Mastra calls content "document"

    def test_list_indexes_returns_index_stats_shape(self, client):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=client)
        indexes = store.list_indexes()

        assert len(indexes) > 0
        for idx in indexes:
            assert set(idx.keys()) == self.INDEX_STATS_KEYS, (
                f"Key mismatch: got {set(idx.keys())}, expected {self.INDEX_STATS_KEYS}"
            )
            assert isinstance(idx["name"], str)
            assert isinstance(idx["dimension"], int)
            assert isinstance(idx["metric"], str)
            assert isinstance(idx["count"], int)

    def test_describe_index_returns_index_stats_shape(self, client):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=client)
        info = store.describe_index("security")

        assert set(info.keys()) == self.INDEX_STATS_KEYS
        assert info["name"] == "security"
        assert info["count"] == 3  # 3 security concepts

    def test_feedback_method_exists(self, client):
        """Mastra has NO feedback. This is qortex's differentiator."""
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=client)
        results = store.query(index_name="security", query_text="auth")
        assert store.last_query_id is not None

        # This call doesn't exist in Mastra. It's the upgrade.
        store.feedback({results[0]["id"]: "accepted"})


# ===========================================================================
# Cross-framework: same backend, all frameworks, verified against real types
# ===========================================================================


class TestCrossFrameworkCompat:
    """Same query through all adapters, verified against real framework types."""

    def test_all_adapters_same_ids(self, client):
        """All adapters return the same IDs for the same query."""
        from qortex.adapters.agno import QortexKnowledge
        from qortex.adapters.crewai import QortexKnowledgeStorage
        from qortex.adapters.mastra import QortexVectorStore

        query = "OAuth2 authentication"

        crewai_ids = [
            r["id"]
            for r in QortexKnowledgeStorage(client=client, domains=["security"]).search(
                [query], limit=3
            )
        ]
        agno_ids = [
            (r["id"] if isinstance(r, dict) else r.id)
            for r in QortexKnowledge(client=client, domains=["security"], top_k=3).retrieve(query)
        ]
        mastra_ids = [
            r["id"]
            for r in QortexVectorStore(client=client).query(
                index_name="security", query_text=query, top_k=3
            )
        ]

        assert crewai_ids == agno_ids == mastra_ids

    @pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_langchain_ids_match(self, client):
        from qortex.adapters.crewai import QortexKnowledgeStorage
        from qortex.adapters.langchain import QortexRetriever

        query = "role-based access"

        lc_ids = [
            doc.id
            for doc in QortexRetriever(client=client, domains=["security"], top_k=3).invoke(query)
        ]
        crewai_ids = [
            r["id"]
            for r in QortexKnowledgeStorage(client=client, domains=["security"]).search(
                [query], limit=3
            )
        ]

        assert lc_ids == crewai_ids
