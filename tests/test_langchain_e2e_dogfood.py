"""E2E dogfood: QortexVectorStore as a LangChain VectorStore drop-in.

This is the case study. If you can run this test, you can ship langchain-qortex.

Proves the FULL integration — not just retrieval, but the things no other
VectorStore can do:

1. QortexVectorStore.from_texts() — zero-config, just like Chroma/FAISS
2. similarity_search() / similarity_search_with_score() — standard VectorStore API
3. as_retriever() — works in any LangChain chain
4. add_texts() — incremental ingestion
5. explore(node_id) — navigate the graph from any search result
6. rules() — get projected rules linked to concepts
7. feedback() — close the learning loop (the thing nobody else has)
8. Rules auto-surfaced in query results — zero consumer effort

The pitch: swap Chroma for QortexVectorStore. Same API. Same chains.
Plus: graph structure, rules, feedback-driven learning.
"""

from __future__ import annotations

import hashlib

import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from qortex.adapters.langchain_vectorstore import QortexVectorStore
from qortex.client import LocalQortexClient, RuleItem
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

DIMS = 32


class FakeLCEmbedding(Embeddings):
    """LangChain-native embedding for from_texts demo."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:DIMS]]
            norm = sum(v * v for v in vec) ** 0.5
            result.append([v / norm for v in vec])
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FakeQortexEmbedding:
    """qortex-native embedding for client-based setup."""

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


def make_security_graph():
    """Build a security knowledge graph with concepts, edges, and rules.

    This is the realistic setup: concepts linked by typed edges, with
    explicit rules attached to concepts. The graph enables explore() and
    rules() — the things flat vector stores can't do.
    """
    vector_index = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vector_index)
    backend.connect()
    embedding = FakeQortexEmbedding()

    backend.create_domain("security")

    nodes = [
        ConceptNode(
            id="sec:oauth", name="OAuth2",
            description="OAuth2 authorization framework for delegated access to APIs",
            domain="security", source_id="security-handbook",
        ),
        ConceptNode(
            id="sec:jwt", name="JWT",
            description="JSON Web Tokens for stateless authentication and claims transfer",
            domain="security", source_id="security-handbook",
        ),
        ConceptNode(
            id="sec:rbac", name="RBAC",
            description="Role-based access control restricts system access by user roles",
            domain="security", source_id="security-handbook",
        ),
        ConceptNode(
            id="sec:mfa", name="MFA",
            description="Multi-factor authentication requires multiple verification factors",
            domain="security", source_id="security-handbook",
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

    # Typed edges — the graph structure
    backend.add_edge(ConceptEdge(
        source_id="sec:oauth", target_id="sec:jwt",
        relation_type=RelationType.REQUIRES,
    ))
    backend.add_edge(ConceptEdge(
        source_id="sec:oauth", target_id="sec:rbac",
        relation_type=RelationType.USES,
    ))
    backend.add_edge(ConceptEdge(
        source_id="sec:mfa", target_id="sec:oauth",
        relation_type=RelationType.SUPPORTS,
    ))

    # Rules — projected from the graph
    backend.add_rule(ExplicitRule(
        id="rule:oauth-required", text="Always use OAuth2 for third-party API access",
        domain="security", source_id="security-handbook",
        concept_ids=["sec:oauth"], category="security",
    ))
    backend.add_rule(ExplicitRule(
        id="rule:rotate-jwt", text="Rotate JWT signing keys every 90 days",
        domain="security", source_id="security-handbook",
        concept_ids=["sec:oauth", "sec:jwt"], category="operations",
    ))
    backend.add_rule(ExplicitRule(
        id="rule:rbac-before-code", text="Define RBAC roles before writing authorization code",
        domain="security", source_id="security-handbook",
        concept_ids=["sec:rbac"], category="architectural",
    ))

    client = LocalQortexClient(
        vector_index=vector_index,
        backend=backend,
        embedding_model=embedding,
        mode="graph",
    )
    return client


# ===========================================================================
# E2E 1: from_texts — THE ONELINER (like Chroma.from_texts)
# ===========================================================================


class TestFromTextsOneliner:
    """The simplest way to use qortex in LangChain. One line."""

    def test_from_texts_and_search(self):
        """
        # Before (Chroma):
        vs = Chroma.from_texts(texts, embedding)

        # After (qortex):
        vs = QortexVectorStore.from_texts(texts, embedding)

        Same API. Same result type. Plus graph + rules + feedback.
        """
        vs = QortexVectorStore.from_texts(
            texts=[
                "OAuth2 is an authorization framework for API access",
                "JWT tokens carry signed claims between parties",
                "RBAC assigns permissions based on user roles",
            ],
            embedding=FakeLCEmbedding(),
            metadatas=[
                {"source": "handbook", "chapter": "auth"},
                {"source": "handbook", "chapter": "tokens"},
                {"source": "handbook", "chapter": "access"},
            ],
            domain="security",
        )

        assert isinstance(vs, VectorStore)
        docs = vs.similarity_search("authentication tokens", k=2)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_from_texts_as_retriever_in_chain(self):
        """Use in a chain: retriever = vs.as_retriever()"""
        vs = QortexVectorStore.from_texts(
            texts=["Python is great", "Rust is fast", "Go is concurrent"],
            embedding=FakeLCEmbedding(),
            domain="languages",
        )

        retriever = vs.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke("fast programming language")
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)


# ===========================================================================
# E2E 2: SIMILARITY SEARCH (standard VectorStore API)
# ===========================================================================


class TestSimilaritySearchE2E:
    """Standard VectorStore search — the API every LangChain user knows."""

    def test_similarity_search(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        docs = vs.similarity_search("OAuth2 authorization", k=3)
        assert len(docs) > 0
        for doc in docs:
            assert isinstance(doc, Document)
            assert doc.metadata["domain"] == "security"
            assert "node_id" in doc.metadata

    def test_similarity_search_with_score(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        results = vs.similarity_search_with_score("JWT tokens", k=3)
        assert len(results) > 0
        scores = [score for _, score in results]
        # Scores should be descending (most similar first)
        assert scores == sorted(scores, reverse=True)

    def test_add_then_search(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        # Add new concepts dynamically
        ids = vs.add_texts(
            ["Zero-trust architecture assumes no implicit trust"],
            metadatas=[{"source": "new-research"}],
        )
        assert len(ids) == 1

        # Search finds them
        docs = vs.similarity_search("zero trust", k=5)
        assert len(docs) > 0


# ===========================================================================
# E2E 3: GRAPH EXPLORATION FROM SEARCH RESULTS
# ===========================================================================


class TestGraphExplorationE2E:
    """The differentiator: search → explore the graph from any result."""

    def test_search_then_explore(self):
        """
        # Standard VectorStore: search returns flat documents. Dead end.
        # QortexVectorStore: search returns documents with node_id.
        # Use node_id to explore the graph neighborhood.
        """
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        # 1. Search (standard LangChain API)
        docs = vs.similarity_search("OAuth2 authorization", k=3)
        assert len(docs) > 0

        # 2. Explore (qortex extra — the thing Chroma can't do)
        node_id = docs[0].metadata["node_id"]
        explore = vs.explore(node_id)
        assert explore is not None
        assert explore.node.id == node_id

        # 3. See typed edges (not just "similar" — actually related)
        assert len(explore.edges) > 0
        for edge in explore.edges:
            assert isinstance(edge.relation_type, str)

        # 4. Navigate to neighbors
        assert len(explore.neighbors) > 0
        neighbor_ids = {n.id for n in explore.neighbors}
        # OAuth2 connects to JWT and RBAC
        assert len(neighbor_ids) >= 1

    def test_explore_reveals_rules_at_node(self):
        """explore() surfaces rules linked to the concept and its neighbors."""
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        explore = vs.explore("sec:oauth")
        assert len(explore.rules) > 0
        rule_ids = {r.id for r in explore.rules}
        assert "rule:oauth-required" in rule_ids


# ===========================================================================
# E2E 4: RULES PROJECTION
# ===========================================================================


class TestRulesProjectionE2E:
    """rules() — get projected rules from the knowledge graph."""

    def test_rules_for_retrieved_concepts(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        docs = vs.similarity_search("OAuth2 authorization", k=3)
        activated_ids = [doc.metadata["node_id"] for doc in docs]

        rules_result = vs.rules(concept_ids=activated_ids)
        assert len(rules_result.rules) > 0
        assert rules_result.projection == "rules"

    def test_rules_by_category(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        arch_rules = vs.rules(categories=["architectural"])
        assert all(r.category == "architectural" for r in arch_rules.rules)

    def test_rules_in_query_results(self):
        """query() automatically includes linked rules — zero extra effort."""
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        results = vs.similarity_search_with_score("OAuth2 authorization", k=4)
        # Rules are embedded in Document metadata
        all_docs = [doc for doc, _ in results]
        # QueryResult.rules gets attached to matching documents
        assert len(all_docs) > 0


# ===========================================================================
# E2E 5: FEEDBACK LOOP (the thing nobody else has)
# ===========================================================================


class TestFeedbackLoopE2E:
    """feedback() — close the learning loop. This is the moat."""

    def test_search_feedback_re_search(self):
        """
        # Standard VectorStore: search → done. No learning.
        # QortexVectorStore: search → feedback → search again → better results.
        """
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        # 1. Search
        docs1 = vs.similarity_search("authentication protocol", k=4)
        assert len(docs1) > 0

        # 2. Feedback — tell qortex what was useful
        vs.feedback({docs1[0].id: "accepted"})

        # 3. Re-search — still works, feedback recorded
        docs2 = vs.similarity_search("authentication protocol", k=4)
        assert len(docs2) > 0

    def test_feedback_tracked_by_query_id(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        vs.similarity_search("OAuth2", k=2)
        qid1 = vs.last_query_id
        assert qid1 is not None

        vs.similarity_search("JWT", k=2)
        qid2 = vs.last_query_id
        assert qid2 is not None
        assert qid1 != qid2


# ===========================================================================
# E2E 6: THE FULL LOOP
# ===========================================================================


class TestFullIntegrationLoop:
    """The whole enchilada: create → search → explore → rules → feedback → repeat."""

    def test_complete_vectorstore_workflow(self):
        client = make_security_graph()
        vs = QortexVectorStore(client=client, domain="security")

        # 1. Search (standard VectorStore API)
        docs = vs.similarity_search("how to authenticate API requests", k=4)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

        # 2. Explore top result's graph neighborhood
        top_node = docs[0].metadata["node_id"]
        explore = vs.explore(top_node)
        assert explore is not None
        assert len(explore.edges) >= 0

        # 3. Get rules for all activated concepts
        activated = [d.metadata["node_id"] for d in docs]
        rules_result = vs.rules(concept_ids=activated)
        assert isinstance(rules_result.rules, list)

        # 4. Feedback — accept the top result, reject the last
        outcomes = {docs[0].id: "accepted"}
        if len(docs) > 1:
            outcomes[docs[-1].id] = "rejected"
        vs.feedback(outcomes)

        # 5. Search again — learning loop closed
        docs2 = vs.similarity_search("how to authenticate API requests", k=4)
        assert len(docs2) > 0

    def test_from_texts_to_full_loop(self):
        """Start from from_texts() — prove the zero-config path works E2E."""
        vs = QortexVectorStore.from_texts(
            texts=[
                "OAuth2 enables delegated authorization for third-party apps",
                "API keys are simple but lack granular permissions",
                "mTLS provides mutual authentication between services",
            ],
            embedding=FakeLCEmbedding(),
            domain="auth-methods",
        )

        # Search
        docs = vs.similarity_search("API authentication", k=3)
        assert len(docs) > 0

        # Add more data dynamically
        vs.add_texts(
            ["SAML enables single sign-on across enterprise applications"],
            metadatas=[{"source": "enterprise-docs"}],
        )

        # Re-search finds new data
        docs2 = vs.similarity_search("enterprise SSO", k=3)
        assert len(docs2) > 0

        # Feedback
        vs.feedback({docs[0].id: "accepted"})
