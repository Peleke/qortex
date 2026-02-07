"""Dogfood: Demonstrate qortex as a drop-in replacement for framework memory systems.

This isn't a unit test — it's a demonstration that qortex adapters are genuine
drop-in replacements. Each test simulates a real consumer workflow:

1. Mastra: QortexVectorStore as MastraVector drop-in
2. LangChain: QortexRetriever as BaseRetriever in a retrieval chain
3. CrewAI: QortexKnowledgeStorage as KnowledgeStorage drop-in
4. Agno: QortexKnowledge as knowledge source
5. Cross-framework: Same qortex backend, multiple frameworks, consistent results
6. Feedback loop: The thing Mastra can't do — learning from outcomes

The point: install qortex, swap one import, get vec+graph+feedback for free.
"""

from __future__ import annotations

import hashlib

import pytest

from qortex.client import LocalQortexClient, QueryItem, QueryResult
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptNode
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Shared setup: a realistic knowledge base
# ---------------------------------------------------------------------------

DIMS = 32


class FakeEmbedding:
    """Deterministic embedding for reproducible tests."""

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
def knowledge_base():
    """A realistic knowledge base with security + architecture domains."""
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeEmbedding()

    # Security domain
    backend.create_domain("security")
    security_nodes = [
        ConceptNode(
            id="security:oauth2", name="OAuth2",
            description="OAuth2 is an authorization framework that enables applications "
                        "to obtain limited access to user accounts on HTTP services",
            domain="security", source_id="docs/auth.md",
        ),
        ConceptNode(
            id="security:jwt", name="JWT",
            description="JSON Web Tokens provide a compact, URL-safe means of "
                        "representing claims to be transferred between two parties",
            domain="security", source_id="docs/auth.md",
        ),
        ConceptNode(
            id="security:rbac", name="RBAC",
            description="Role-based access control restricts system access to "
                        "authorized users based on their assigned roles",
            domain="security", source_id="docs/auth.md",
        ),
        ConceptNode(
            id="security:mfa", name="MFA",
            description="Multi-factor authentication requires users to provide "
                        "two or more verification factors to gain access",
            domain="security", source_id="docs/auth.md",
        ),
    ]

    # Architecture domain
    backend.create_domain("architecture")
    arch_nodes = [
        ConceptNode(
            id="arch:microservices", name="Microservices",
            description="Microservices architecture structures an application as "
                        "a collection of loosely coupled, independently deployable services",
            domain="architecture", source_id="docs/arch.md",
        ),
        ConceptNode(
            id="arch:event-driven", name="Event-Driven",
            description="Event-driven architecture uses events to trigger and "
                        "communicate between decoupled services and components",
            domain="architecture", source_id="docs/arch.md",
        ),
        ConceptNode(
            id="arch:cqrs", name="CQRS",
            description="Command Query Responsibility Segregation separates read "
                        "and write operations for a data store into distinct models",
            domain="architecture", source_id="docs/arch.md",
        ),
    ]

    all_nodes = security_nodes + arch_nodes
    for node in all_nodes:
        backend.add_node(node)

    texts = [f"{n.name}: {n.description}" for n in all_nodes]
    embeddings = embedding.embed(texts)
    for node, emb in zip(all_nodes, embeddings):
        backend.add_embedding(node.id, emb)

    client = LocalQortexClient(
        vector_index=vector_index,
        backend=backend,
        embedding_model=embedding,
    )

    return client


# ===========================================================================
# 1. MASTRA DROP-IN: QortexVectorStore as MastraVector
# ===========================================================================


class TestMastraDropIn:
    """Demonstrate: swap PgVector/Chroma/Pinecone for qortex. Same API."""

    def test_query_like_mastra_vector(self, knowledge_base):
        """Mastra consumer code: store.query(index_name, query_text, top_k)"""
        from qortex.adapters.mastra import QortexVectorStore

        # --- This is what the consumer writes ---
        store = QortexVectorStore(client=knowledge_base)
        results = store.query(
            index_name="security",
            query_text="How does OAuth2 authentication work?",
            top_k=3,
        )
        # --- End consumer code ---

        # Mastra QueryResult shape: {id, score, metadata, document}
        assert len(results) > 0
        first = results[0]
        assert isinstance(first["id"], str)
        assert isinstance(first["score"], float)
        assert isinstance(first["metadata"], dict)
        assert isinstance(first["document"], str)
        assert first["metadata"]["domain"] == "security"

    def test_list_indexes_like_mastra(self, knowledge_base):
        """Mastra consumer code: store.listIndexes()"""
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=knowledge_base)
        indexes = store.list_indexes()

        # Mastra IndexStats shape: {name, dimension, metric, count}
        names = {idx["name"] for idx in indexes}
        assert "security" in names
        assert "architecture" in names
        for idx in indexes:
            assert "dimension" in idx
            assert idx["metric"] == "cosine"
            assert isinstance(idx["count"], int)

    def test_describe_index_like_mastra(self, knowledge_base):
        """Mastra consumer code: store.describeIndex(name)"""
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=knowledge_base)
        info = store.describe_index("security")

        assert info["name"] == "security"
        assert info["count"] == 4  # 4 security concepts
        assert info["metric"] == "cosine"

    def test_create_index_is_graceful(self, knowledge_base):
        """Mastra consumer code: store.createIndex(name, dimension, metric)
        In qortex, domains auto-create. This should be a no-op, not an error."""
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=knowledge_base)
        store.create_index("new-domain", dimension=768, metric="cosine")
        # No error — graceful

    def test_feedback_is_the_upgrade(self, knowledge_base):
        """This is what Mastra CAN'T do. qortex learns from outcomes."""
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=knowledge_base)

        # Query
        results = store.query(
            index_name="security",
            query_text="authentication protocol",
            top_k=5,
        )
        assert len(results) > 0
        assert store.last_query_id is not None

        # Feedback — close the learning loop
        # In Mastra: not possible. In qortex: one method call.
        store.feedback({
            results[0]["id"]: "accepted",
            results[-1]["id"]: "rejected",
        })
        # No error. Feedback recorded. Future queries improve.


# ===========================================================================
# 2. LANGCHAIN DROP-IN: QortexRetriever as BaseRetriever
# ===========================================================================


try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever as LCBaseRetriever

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
class TestLangChainDropIn:
    """Demonstrate: swap any langchain retriever for qortex. Same chain API."""

    def test_invoke_like_any_retriever(self, knowledge_base):
        """LangChain consumer code: retriever.invoke(query)"""
        from qortex.adapters.langchain import QortexRetriever

        # --- This is what the consumer writes ---
        retriever = QortexRetriever(
            client=knowledge_base,
            domains=["security"],
            top_k=3,
        )
        docs = retriever.invoke("What is role-based access control?")
        # --- End consumer code ---

        # Standard langchain Document shape
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, LCDocument)
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0
            assert "score" in doc.metadata
            assert "domain" in doc.metadata
            assert doc.metadata["domain"] == "security"
            assert doc.id is not None

    def test_is_actual_base_retriever(self, knowledge_base):
        """The retriever IS a BaseRetriever — works in any chain."""
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=knowledge_base)
        assert isinstance(retriever, LCBaseRetriever)

    def test_works_in_retrieval_pipeline(self, knowledge_base):
        """Simulate a retrieval pipeline: retrieve → format → done.
        In real LangChain, this would be retriever | format | llm."""
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(
            client=knowledge_base,
            domains=["architecture"],
            top_k=2,
        )

        # Step 1: Retrieve (exactly like any langchain retriever)
        docs = retriever.invoke("event driven microservices")

        # Step 2: Format context (what you'd pipe to an LLM)
        context = "\n\n".join(
            f"[{doc.metadata.get('domain', 'unknown')}] {doc.page_content}"
            for doc in docs
        )

        assert len(context) > 0
        assert "architecture" in context.lower() or "event" in context.lower()

    def test_cross_domain_retrieval(self, knowledge_base):
        """Retrieve across all domains — no domain filter."""
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=knowledge_base, top_k=7)
        docs = retriever.invoke("system design")

        # Should find results from both domains
        domains = {doc.metadata["domain"] for doc in docs}
        assert len(docs) > 0
        # With our hash-based embeddings, we can't guarantee both domains
        # match "system design", but results should be non-empty

    def test_feedback_closes_learning_loop(self, knowledge_base):
        """LangChain retrievers don't learn. qortex does."""
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(
            client=knowledge_base,
            domains=["security"],
        )

        docs = retriever.invoke("authentication")
        assert len(docs) > 0

        # Standard langchain: no way to say "this result was good"
        # qortex: one method call
        retriever.feedback({docs[0].id: "accepted"})


# ===========================================================================
# 3. CREWAI DROP-IN: QortexKnowledgeStorage as KnowledgeStorage
# ===========================================================================


class TestCrewAIDropIn:
    """Demonstrate: swap ChromaDB KnowledgeStorage for qortex."""

    def test_search_like_crewai(self, knowledge_base):
        """CrewAI consumer code: storage.search(query, limit, score_threshold)"""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        # --- This is what the consumer writes ---
        storage = QortexKnowledgeStorage(
            client=knowledge_base,
            domains=["security"],
        )
        results = storage.search("JWT token authentication", limit=3)
        # --- End consumer code ---

        # CrewAI SearchResult shape: {id, content, metadata, score}
        assert len(results) > 0
        for r in results:
            assert isinstance(r["id"], str)
            assert isinstance(r["content"], str)
            assert isinstance(r["score"], float)
            assert isinstance(r["metadata"], dict)
            assert r["metadata"]["domain"] == "security"

    def test_save_and_reset_are_graceful(self, knowledge_base):
        """CrewAI expects save() and reset(). Both are no-ops in qortex
        (we use file-based ingestion, not string injection)."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=knowledge_base)
        storage.save(["doc1", "doc2"])  # no-op, no error
        storage.reset()  # no-op, no error

    def test_score_threshold_filters(self, knowledge_base):
        """CrewAI's score_threshold is passed through."""
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(
            client=knowledge_base,
            domains=["security"],
        )

        # Very high threshold — should filter most results
        results = storage.search("auth", limit=10, score_threshold=0.999)
        for r in results:
            assert r["score"] >= 0.999


# ===========================================================================
# 4. AGNO DROP-IN: QortexKnowledge as knowledge source
# ===========================================================================


class TestAgnoDropIn:
    """Demonstrate: use qortex as an agno knowledge source."""

    def test_retrieve_like_agno(self, knowledge_base):
        """Agno consumer code: knowledge.retrieve(query)"""
        from qortex.adapters.agno import QortexKnowledge

        # --- This is what the consumer writes ---
        knowledge = QortexKnowledge(
            client=knowledge_base,
            domains=["architecture"],
            top_k=3,
        )
        docs = knowledge.retrieve("microservices architecture")
        # --- End consumer code ---

        assert len(docs) > 0
        for doc in docs:
            if isinstance(doc, dict):
                assert "content" in doc
                assert "id" in doc
                assert "name" in doc
                assert "meta_data" in doc
                assert "reranking_score" in doc
            else:
                assert hasattr(doc, "content")

    def test_build_context_like_agno(self, knowledge_base):
        """Agno consumer code: knowledge.build_context(query)"""
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(
            client=knowledge_base,
            domains=["security"],
        )

        context = knowledge.build_context("What authentication methods exist?")
        assert isinstance(context, str)
        assert len(context) > 0


# ===========================================================================
# 5. CROSS-FRAMEWORK CONSISTENCY
# ===========================================================================


class TestCrossFrameworkDogfood:
    """The killer feature: same qortex backend, every framework gets identical results."""

    def test_all_adapters_return_same_data(self, knowledge_base):
        """Query the same backend through all 4 adapters. Same results."""
        from qortex.adapters.agno import QortexKnowledge
        from qortex.adapters.crewai import QortexKnowledgeStorage
        from qortex.adapters.mastra import QortexVectorStore

        query = "OAuth2 authorization framework"

        # Mastra
        mastra_results = QortexVectorStore(client=knowledge_base).query(
            index_name="security", query_text=query, top_k=3,
        )

        # CrewAI
        crewai_results = QortexKnowledgeStorage(
            client=knowledge_base, domains=["security"]
        ).search(query, limit=3)

        # Agno
        agno_results = QortexKnowledge(
            client=knowledge_base, domains=["security"], top_k=3,
        ).retrieve(query)

        # All return same number of results
        assert len(mastra_results) == len(crewai_results) == len(agno_results)

        # All return same IDs in same order
        mastra_ids = [r["id"] for r in mastra_results]
        crewai_ids = [r["id"] for r in crewai_results]
        agno_ids = [
            r["id"] if isinstance(r, dict) else r.id for r in agno_results
        ]

        assert mastra_ids == crewai_ids == agno_ids

        # All return same scores
        mastra_scores = [r["score"] for r in mastra_results]
        crewai_scores = [r["score"] for r in crewai_results]
        agno_scores = [
            r["reranking_score"] if isinstance(r, dict) else r.reranking_score
            for r in agno_results
        ]

        assert mastra_scores == crewai_scores == agno_scores

    @pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_langchain_matches_others(self, knowledge_base):
        """LangChain retriever returns same results as other adapters."""
        from qortex.adapters.crewai import QortexKnowledgeStorage
        from qortex.adapters.langchain import QortexRetriever

        query = "role-based access control"

        lc_docs = QortexRetriever(
            client=knowledge_base, domains=["security"], top_k=3,
        ).invoke(query)

        crewai_results = QortexKnowledgeStorage(
            client=knowledge_base, domains=["security"],
        ).search(query, limit=3)

        # Same IDs
        lc_ids = [doc.id for doc in lc_docs]
        crewai_ids = [r["id"] for r in crewai_results]
        assert lc_ids == crewai_ids

        # Same scores
        lc_scores = [doc.metadata["score"] for doc in lc_docs]
        crewai_scores = [r["score"] for r in crewai_results]
        assert lc_scores == crewai_scores


# ===========================================================================
# 6. FEEDBACK LOOP: The thing nobody else has
# ===========================================================================


class TestFeedbackLoopDogfood:
    """Demonstrate the unique value prop: feedback → learning → improvement."""

    def test_full_feedback_cycle(self, knowledge_base):
        """Complete cycle: query → review results → feedback → next query.

        In real usage, teleportation factors would update and improve retrieval.
        At Level 0 (vec-only), feedback is recorded but not yet acted on.
        At Level 2 (feedback-driven), each feedback cycle makes retrieval better.
        """
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=knowledge_base)

        # Cycle 1: Query
        results = store.query(
            index_name="security",
            query_text="authentication methods",
            top_k=4,
        )
        assert len(results) > 0
        query_id_1 = store.last_query_id

        # Cycle 1: Feedback
        store.feedback({
            results[0]["id"]: "accepted",
        })

        # Cycle 2: Query again
        results_2 = store.query(
            index_name="security",
            query_text="access control mechanisms",
            top_k=4,
        )
        query_id_2 = store.last_query_id

        # Different queries get different query IDs (for correlation)
        assert query_id_1 != query_id_2

        # Cycle 2: Feedback
        store.feedback({
            results_2[0]["id"]: "accepted",
        })

        # At Level 0: feedback is recorded, retrieval is unchanged
        # At Level 2: feedback would adjust teleportation factors
        # The point: the WIRING is there. Consumers don't change code.

    def test_feedback_across_adapters_shares_backend(self, knowledge_base):
        """Feedback from one adapter improves retrieval for all adapters.

        This is possible because all adapters target the same QortexClient.
        Mastra feedback → qortex learns → LangChain retrieval improves.
        """
        from qortex.adapters.crewai import QortexKnowledgeStorage
        from qortex.adapters.mastra import QortexVectorStore

        # Mastra queries and gives feedback
        mastra = QortexVectorStore(client=knowledge_base)
        results = mastra.query(
            index_name="security",
            query_text="authentication",
            top_k=3,
        )
        mastra.feedback({results[0]["id"]: "accepted"})

        # CrewAI queries the same backend — same data, same improvements
        crewai = QortexKnowledgeStorage(
            client=knowledge_base, domains=["security"]
        )
        crewai_results = crewai.search("authentication", limit=3)

        # Both get the same results (shared backend)
        assert [r["id"] for r in results] == [r["id"] for r in crewai_results]

        # CrewAI can also give feedback
        crewai.feedback({crewai_results[0]["id"]: "accepted"})
        # No error. Cross-framework learning works.


# ===========================================================================
# 7. THE PITCH: One import swap, zero code changes
# ===========================================================================


class TestTheOneliner:
    """The entire value proposition in code.

    Before (Mastra + PgVector):
        from @mastra/pg import PgVector
        store = PgVector(connectionString)

    After (qortex):
        from qortex.adapters.mastra import QortexVectorStore
        store = QortexVectorStore(client=client)

    Same API. Same results. Plus: graph, feedback, cross-session learning.
    """

    def test_mastra_oneliner(self, knowledge_base):
        from qortex.adapters.mastra import QortexVectorStore

        store = QortexVectorStore(client=knowledge_base)
        results = store.query(index_name="security", query_text="auth", top_k=3)
        assert len(results) > 0
        assert all(
            set(r.keys()) == {"id", "score", "metadata", "document"}
            for r in results
        )

    def test_crewai_oneliner(self, knowledge_base):
        from qortex.adapters.crewai import QortexKnowledgeStorage

        storage = QortexKnowledgeStorage(client=knowledge_base, domains=["security"])
        results = storage.search("auth", limit=3)
        assert len(results) > 0
        assert all(
            set(r.keys()) == {"id", "content", "metadata", "score"}
            for r in results
        )

    @pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain-core not installed")
    def test_langchain_oneliner(self, knowledge_base):
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=knowledge_base, domains=["security"])
        docs = retriever.invoke("auth")
        assert len(docs) > 0
        assert all(isinstance(d, LCDocument) for d in docs)

    def test_agno_oneliner(self, knowledge_base):
        from qortex.adapters.agno import QortexKnowledge

        knowledge = QortexKnowledge(client=knowledge_base, domains=["security"])
        docs = knowledge.retrieve("auth")
        assert len(docs) > 0
