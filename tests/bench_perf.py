"""Perf benchmark: QortexKnowledge vs vanilla vector search.

Measures latency overhead of graph-enhanced retrieval vs pure cosine.

Run: uv run pytest tests/bench_perf.py -v -s
"""

from __future__ import annotations

import statistics
import time

import pytest

from qortex.adapters.agno import QortexKnowledge
from qortex.client import LocalQortexClient
from qortex.core.memory import InMemoryBackend
from qortex.core.models import RelationType
from qortex.vec.embeddings import SentenceTransformerEmbedding
from qortex.vec.index import NumpyVectorIndex

QUERIES = [
    "How should a mobile app handle OAuth2 authentication securely?",
    "Compare different token formats and session management approaches",
    "How to implement enterprise single sign-on for corporate apps?",
    "Secure machine to machine authentication in microservices",
    "What are best practices for API key rotation?",
    "How does refresh token rotation prevent replay attacks?",
    "Explain the differences between SAML and OpenID Connect",
    "When should I use mutual TLS vs API keys?",
]

CONCEPTS = [
    ("OAuth2", "Authorization framework for delegated access using access tokens and scopes"),
    ("JWT", "JSON Web Tokens — self-contained tokens encoding claims as signed JSON payloads"),
    ("OpenID Connect", "Identity layer on OAuth2 providing authentication and ID tokens"),
    ("PKCE", "Proof Key for Code Exchange — prevents authorization code interception attacks"),
    ("Refresh Token", "Long-lived token used to obtain new access tokens silently"),
    ("SAML", "Security Assertion Markup Language — XML-based SSO protocol for enterprises"),
    ("mTLS", "Mutual TLS — client certificate authentication for machine-to-machine"),
    ("API Key", "Simple bearer credential for server-to-server, no user delegation"),
    ("Session Cookie", "Server-side session tracked via HTTP cookie, stateful"),
    ("CORS", "Cross-Origin Resource Sharing — browser security policy for cross-domain requests"),
    ("OAuth1", "Legacy authorization protocol using request signing and nonces"),
    ("HTTP Basic Auth", "Simple username/password authentication sent as base64 in HTTP header"),
    ("Kerberos", "Network authentication protocol using ticket-granting tickets"),
    ("LDAP", "Lightweight Directory Access Protocol for directory services"),
    ("RADIUS", "Remote Authentication Dial-In User Service for network access control"),
    ("X.509 Certificate", "Public key certificate standard for identity verification in PKI"),
    ("Digest Authentication", "HTTP authentication using challenge-response with MD5 hashing"),
    ("SCRAM", "Salted Challenge Response Authentication Mechanism for password-based auth"),
    ("WebAuthn", "Web Authentication API for passwordless authentication using FIDO2"),
    ("TOTP", "Time-based One-Time Password used in two-factor authentication apps"),
]

EDGES = [
    ("OpenID Connect", "OAuth2", "refines"),
    ("PKCE", "OAuth2", "supports"),
    ("JWT", "OAuth2", "uses"),
    ("Refresh Token", "OAuth2", "part_of"),
    ("SAML", "OpenID Connect", "similar_to"),
    ("mTLS", "OAuth2", "supports"),
    ("API Key", "mTLS", "alternative_to"),
    ("Session Cookie", "JWT", "alternative_to"),
]

RULES = [
    ("Always use PKCE for public clients", "security"),
    ("Refresh tokens must be rotated on each use", "security"),
    ("JWTs should have short expiry (5-15 minutes)", "security"),
]


def _timeit(fn, warmup=2, runs=20):
    """Run fn warmup+runs times, return (median_ms, p95_ms, all_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = statistics.median(times)
    p95 = times[int(len(times) * 0.95)]
    return median, p95, times


@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformerEmbedding()


@pytest.fixture(scope="module")
def qortex_setup(embedding_model):
    """Full qortex: graph + vec + rules."""
    vi = NumpyVectorIndex(dimensions=384)
    backend = InMemoryBackend(vector_index=vi)
    backend.connect()
    client = LocalQortexClient(
        vector_index=vi, backend=backend, embedding_model=embedding_model, mode="auto"
    )
    client.ingest_structured(
        domain="auth",
        concepts=[{"name": n, "description": d} for n, d in CONCEPTS],
        edges=[{"source": s, "target": t, "relation_type": r} for s, t, r in EDGES],
        rules=[{"text": t, "category": c} for t, c in RULES],
    )
    knowledge = QortexKnowledge(client=client, domains=["auth"], top_k=5)
    return knowledge, client


@pytest.fixture(scope="module")
def vanilla_setup(embedding_model):
    """Vanilla: just a NumpyVectorIndex, no graph."""
    vi = NumpyVectorIndex(dimensions=384)
    texts = [f"{n}: {d}" for n, d in CONCEPTS]
    embeddings = embedding_model.embed(texts)
    ids = [f"v:{i}" for i in range(len(texts))]
    vi.add(ids, embeddings)
    return vi, embedding_model


class TestPerformance:

    def test_embedding_latency(self, embedding_model):
        """Baseline: how fast is embedding a query?"""
        med, p95, _ = _timeit(lambda: embedding_model.embed(["test query"]))
        print(f"\n  Embedding latency:  median={med:.2f}ms  p95={p95:.2f}ms")

    def test_vanilla_search_latency(self, vanilla_setup, embedding_model):
        """Vanilla vector search: embed + cosine."""
        vi, emb = vanilla_setup

        def search():
            for q in QUERIES:
                vec = emb.embed([q])[0]
                vi.search(vec, top_k=5)

        med, p95, _ = _timeit(search)
        per_query = med / len(QUERIES)
        print(f"\n  Vanilla search ({len(QUERIES)} queries):  median={med:.2f}ms  p95={p95:.2f}ms  per_query={per_query:.2f}ms")

    def test_qortex_retrieve_latency(self, qortex_setup):
        """Qortex retrieve: embed + vec search + graph PPR + rules."""
        knowledge, _ = qortex_setup

        def search():
            for q in QUERIES:
                knowledge.retrieve(q, top_k=5)

        med, p95, _ = _timeit(search)
        per_query = med / len(QUERIES)
        print(f"\n  Qortex retrieve ({len(QUERIES)} queries):  median={med:.2f}ms  p95={p95:.2f}ms  per_query={per_query:.2f}ms")

    def test_qortex_search_tool_latency(self, qortex_setup):
        """Qortex search tool: retrieve + format + rules text."""
        knowledge, _ = qortex_setup
        tools = knowledge.get_tools()
        search = tools[0]

        def search_all():
            for q in QUERIES:
                search(q)

        med, p95, _ = _timeit(search_all)
        per_query = med / len(QUERIES)
        print(f"\n  Qortex search_tool ({len(QUERIES)} queries):  median={med:.2f}ms  p95={p95:.2f}ms  per_query={per_query:.2f}ms")

    def test_qortex_explore_latency(self, qortex_setup):
        """Qortex explore: BFS graph traversal."""
        knowledge, client = qortex_setup
        tools = knowledge.get_tools()
        explore = [t for t in tools if t.__name__ == "explore_knowledge_graph"][0]

        # Get a real node_id
        result = client.query(context="OAuth2", top_k=1)
        node_id = result.items[0].node_id

        med, p95, _ = _timeit(lambda: explore(node_id, depth=2))
        print(f"\n  Qortex explore (depth=2):  median={med:.2f}ms  p95={p95:.2f}ms")

    def test_qortex_feedback_latency(self, qortex_setup):
        """Qortex feedback: record outcome."""
        knowledge, _ = qortex_setup
        # Do a search to get a query_id
        knowledge.retrieve("OAuth2")
        tools = knowledge.get_tools()
        feedback = [t for t in tools if t.__name__ == "report_knowledge_feedback"][0]

        med, p95, _ = _timeit(lambda: feedback("some-id", "accepted"))
        print(f"\n  Qortex feedback:  median={med:.2f}ms  p95={p95:.2f}ms")

    def test_overhead_summary(self, qortex_setup, vanilla_setup, embedding_model):
        """Head-to-head: qortex overhead vs vanilla."""
        knowledge, _ = qortex_setup
        vi, emb = vanilla_setup

        def vanilla_batch():
            for q in QUERIES:
                vec = emb.embed([q])[0]
                vi.search(vec, top_k=5)

        def qortex_batch():
            for q in QUERIES:
                knowledge.retrieve(q, top_k=5)

        v_med, v_p95, _ = _timeit(vanilla_batch)
        q_med, q_p95, _ = _timeit(qortex_batch)

        v_per = v_med / len(QUERIES)
        q_per = q_med / len(QUERIES)
        overhead = ((q_per - v_per) / v_per) * 100 if v_per > 0 else 0

        print(f"\n  {'':40} {'Median':>10} {'P95':>10} {'Per Query':>10}")
        print(f"  {'-'*75}")
        print(f"  {'Vanilla (embed + cosine)':40} {v_med:>8.2f}ms {v_p95:>8.2f}ms {v_per:>8.2f}ms")
        print(f"  {'Qortex (embed + vec + graph + rules)':40} {q_med:>8.2f}ms {q_p95:>8.2f}ms {q_per:>8.2f}ms")
        print(f"  {'-'*75}")
        print(f"  {'Overhead':40} {overhead:>+8.1f}%")
        print(f"  {'Overhead per query':40} {q_per - v_per:>+8.2f}ms")
