"""Benchmark: AutoGen QortexMemory vs vanilla cosine search.

Same corpus and methodology as bench_crewai_vs_vanilla.py, but exercises the
AutoGen adapter path (QortexMemory.query -> client.query, all async).

Run: uv run pytest tests/bench_autogen_vs_vanilla.py -v -s
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from qortex.adapters.autogen import QortexMemory
from qortex.client import LocalQortexClient
from qortex.core.memory import InMemoryBackend
from qortex.vec.embeddings import SentenceTransformerEmbedding
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Same auth corpus as bench_crewai_vs_vanilla.py
# ---------------------------------------------------------------------------

AUTH_CONCEPTS = [
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
    # Distractors
    (
        "OAuth1",
        "Legacy authorization protocol using request signing and nonces, predecessor to OAuth2",
    ),
    ("HTTP Basic Auth", "Simple username/password authentication sent as base64 in HTTP header"),
    (
        "Kerberos",
        "Network authentication protocol using ticket-granting tickets and symmetric keys",
    ),
    ("LDAP", "Lightweight Directory Access Protocol for directory services and user lookup"),
    ("RADIUS", "Remote Authentication Dial-In User Service for network access control"),
    ("X.509 Certificate", "Public key certificate standard for identity verification in PKI"),
    ("Digest Authentication", "HTTP authentication using challenge-response with MD5 hashing"),
    ("SCRAM", "Salted Challenge Response Authentication Mechanism for password-based auth"),
    ("WebAuthn", "Web Authentication API for passwordless authentication using FIDO2 credentials"),
    ("TOTP", "Time-based One-Time Password used in two-factor authentication apps"),
]

AUTH_EDGES = [
    ("OpenID Connect", "OAuth2", "refines"),
    ("PKCE", "OAuth2", "supports"),
    ("JWT", "OAuth2", "uses"),
    ("Refresh Token", "OAuth2", "part_of"),
    ("SAML", "OpenID Connect", "similar_to"),
    ("mTLS", "OAuth2", "supports"),
    ("API Key", "mTLS", "alternative_to"),
    ("Session Cookie", "JWT", "alternative_to"),
]

AUTH_RULES = [
    ("Always use PKCE for public clients (SPAs, mobile apps)", "security"),
    ("Refresh tokens must be rotated on each use", "security"),
    ("JWTs should have short expiry (5-15 minutes)", "security"),
    ("Prefer OpenID Connect over raw OAuth2 for user-facing login", "architecture"),
    ("Use mTLS for service mesh and internal microservice auth", "architecture"),
]

EVAL_QUERIES = [
    {
        "query": "How should a mobile app handle OAuth2 authentication securely?",
        "expected": {"OAuth2", "PKCE", "Refresh Token", "OpenID Connect"},
        "distractors": {"OAuth1", "HTTP Basic Auth", "WebAuthn"},
    },
    {
        "query": "Compare different token formats and session management approaches",
        "expected": {"JWT", "Session Cookie", "API Key", "Refresh Token"},
        "distractors": {"TOTP", "Kerberos", "SCRAM"},
    },
    {
        "query": "How to implement enterprise single sign-on for corporate apps?",
        "expected": {"SAML", "OpenID Connect", "OAuth2"},
        "distractors": {"LDAP", "Kerberos", "RADIUS"},
    },
    {
        "query": "Secure machine to machine authentication in microservices",
        "expected": {"mTLS", "API Key", "OAuth2"},
        "distractors": {"X.509 Certificate", "RADIUS", "Digest Authentication"},
    },
]


# ---------------------------------------------------------------------------
# Vanilla baseline
# ---------------------------------------------------------------------------


@dataclass
class VanillaVectorSearch:
    index: NumpyVectorIndex
    embedding: SentenceTransformerEmbedding
    concepts: list[tuple[str, str]]
    _ids: list[str] = field(default_factory=list)

    def setup(self) -> None:
        texts = [f"{name}: {desc}" for name, desc in self.concepts]
        embeddings = self.embedding.embed(texts)
        self._ids = [f"vanilla:{i}" for i in range(len(texts))]
        self.index.add(self._ids, embeddings)
        self._name_map = {self._ids[i]: self.concepts[i][0] for i in range(len(self.concepts))}

    def search(self, query: str, top_k: int = 5) -> list[str]:
        q_emb = self.embedding.embed([query])[0]
        results = self.index.search(q_emb, top_k=top_k)
        names = []
        for r in results:
            rid = r.id if hasattr(r, "id") else r[0]
            if rid in self._name_map:
                names.append(self._name_map[rid])
        return names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> float:
    s = set(retrieved[:k])
    return len(s & expected) / len(s) if s else 0.0


def recall_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> float:
    s = set(retrieved[:k])
    return len(s & expected) / len(expected) if expected else 1.0


def ndcg_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, n in enumerate(retrieved[:k]) if n in expected)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformerEmbedding()


@pytest.fixture(scope="module")
def autogen_memory(embedding_model):
    """QortexMemory with full graph."""
    vi = NumpyVectorIndex(dimensions=384)
    backend = InMemoryBackend(vector_index=vi)
    backend.connect()
    client = LocalQortexClient(
        vector_index=vi,
        backend=backend,
        embedding_model=embedding_model,
        mode="auto",
    )
    client.ingest_structured(
        domain="auth",
        concepts=[{"name": n, "description": d} for n, d in AUTH_CONCEPTS],
        edges=[{"source": s, "target": t, "relation_type": r} for s, t, r in AUTH_EDGES],
        rules=[{"text": t, "category": c} for t, c in AUTH_RULES],
    )
    return QortexMemory(client=client, domains=["auth"])


@pytest.fixture(scope="module")
def vanilla_search(embedding_model):
    vi = NumpyVectorIndex(dimensions=384)
    v = VanillaVectorSearch(index=vi, embedding=embedding_model, concepts=AUTH_CONCEPTS)
    v.setup()
    return v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoGenBenchmark:
    """Head-to-head: QortexMemory (AutoGen adapter) vs vanilla cosine search."""

    def _extract_names(self, results: list[Any]) -> list[str]:
        """Extract concept names from MemoryContent results."""
        names = []
        for r in results:
            content = r.content if hasattr(r, "content") else r.get("content", "")
            for cname, _ in AUTH_CONCEPTS:
                if cname.lower() in content.lower() and cname not in names:
                    names.append(cname)
        return names

    @pytest.mark.asyncio
    async def test_autogen_vs_vanilla(self, autogen_memory, vanilla_search):
        TOP_K = 5
        q_precs, v_precs = [], []
        q_recs, v_recs = [], []
        q_ndcgs, v_ndcgs = [], []
        q_dists, v_dists = [], []
        q_latencies, v_latencies = [], []

        for case in EVAL_QUERIES:
            query = case["query"]
            expected = case["expected"]
            distractors = case.get("distractors", set())

            # AutoGen adapter path (async)
            t0 = time.perf_counter()
            result = await autogen_memory.query(query)
            q_latencies.append((time.perf_counter() - t0) * 1000)
            results = result.results if hasattr(result, "results") else result["results"]
            q_names = self._extract_names(results)

            # Vanilla path
            t0 = time.perf_counter()
            v_names = vanilla_search.search(query, top_k=TOP_K)
            v_latencies.append((time.perf_counter() - t0) * 1000)

            q_precs.append(precision_at_k(q_names, expected, TOP_K))
            v_precs.append(precision_at_k(v_names, expected, TOP_K))
            q_recs.append(recall_at_k(q_names, expected, TOP_K))
            v_recs.append(recall_at_k(v_names, expected, TOP_K))
            q_ndcgs.append(ndcg_at_k(q_names, expected, TOP_K))
            v_ndcgs.append(ndcg_at_k(v_names, expected, TOP_K))
            q_dists.append(len(set(q_names) & distractors))
            v_dists.append(len(set(v_names) & distractors))

        # Print per-query breakdown
        header = f"{'Query':<58} {'Q-P@5':>6} {'V-P@5':>6} {'Q-R@5':>6} {'V-R@5':>6} {'Q-nDCG':>7} {'V-nDCG':>7} {'Q-Dist':>7} {'V-Dist':>7} {'Q-ms':>6} {'V-ms':>6}"
        print(f"\n{header}")
        print("-" * len(header))
        for i, case in enumerate(EVAL_QUERIES):
            print(
                f"{case['query'][:58]:<58} "
                f"{q_precs[i]:>6.2f} {v_precs[i]:>6.2f} "
                f"{q_recs[i]:>6.2f} {v_recs[i]:>6.2f} "
                f"{q_ndcgs[i]:>7.3f} {v_ndcgs[i]:>7.3f} "
                f"{q_dists[i]:>7d} {v_dists[i]:>7d} "
                f"{q_latencies[i]:>6.1f} {v_latencies[i]:>6.1f}"
            )
        print("-" * len(header))

        avg_qp = statistics.mean(q_precs)
        avg_vp = statistics.mean(v_precs)
        avg_qr = statistics.mean(q_recs)
        avg_vr = statistics.mean(v_recs)
        avg_qn = statistics.mean(q_ndcgs)
        avg_vn = statistics.mean(v_ndcgs)

        print(
            f"{'AVERAGE':<58} "
            f"{avg_qp:>6.2f} {avg_vp:>6.2f} "
            f"{avg_qr:>6.2f} {avg_vr:>6.2f} "
            f"{avg_qn:>7.3f} {avg_vn:>7.3f} "
            f"{sum(q_dists):>7d} {sum(v_dists):>7d} "
            f"{statistics.mean(q_latencies):>6.1f} {statistics.mean(v_latencies):>6.1f}"
        )

        # Summary
        p_delta = (avg_qp - avg_vp) / avg_vp * 100 if avg_vp else 0
        r_delta = (avg_qr - avg_vr) / avg_vr * 100 if avg_vr else 0
        n_delta = (avg_qn - avg_vn) / avg_vn * 100 if avg_vn else 0
        print(f"\n  Precision delta: {p_delta:+.0f}%")
        print(f"  Recall delta:    {r_delta:+.0f}%")
        print(f"  nDCG delta:      {n_delta:+.0f}%")
        print(f"  Qortex latency:  {statistics.mean(q_latencies):.1f}ms avg")
        print(f"  Vanilla latency: {statistics.mean(v_latencies):.1f}ms avg")

        # Qortex should match or beat vanilla
        assert avg_qr >= avg_vr
        assert sum(q_dists) <= sum(v_dists)
