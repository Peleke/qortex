"""Comparative benchmark: qortex adapters vs each other on the same workload.

Adapters tested:
    qortex-graph   — GraphRAGAdapter (PPR + vec + teleportation factors)
    qortex-vec     — VecOnlyAdapter (pure cosine similarity)
    agno           — QortexKnowledge (agno KnowledgeProtocol)
    langchain      — QortexRetriever (langchain BaseRetriever)
    autogen        — QortexMemory (autogen Memory interface)
    mastra         — QortexVectorStore (Mastra MastraVector interface)

All adapters share the same corpus, embedding model, and backend.
Metrics: recall@K, MRR, precision@K, latency_p50/p95.

Usage:
    python -m benchmarks.comparative.run
    python -m benchmarks.comparative.run --top-k 10 --output results.json
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qortex.client import LocalQortexClient
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, RelationType
from qortex.vec.index import NumpyVectorIndex


# ---------------------------------------------------------------------------
# Corpus: authentication domain (same as eval_agno_vs_qortex.py)
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
    ("OAuth1", "Legacy authorization protocol using request signing and nonces"),
    ("HTTP Basic Auth", "Simple username/password authentication sent as base64 in HTTP header"),
    ("Kerberos", "Network authentication protocol using ticket-granting tickets and symmetric keys"),
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
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> float:
    """Fraction of expected items found in top-k results."""
    top_k = set(retrieved[:k])
    if not expected:
        return 1.0
    return len(top_k & expected) / len(expected)


def precision_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = set(retrieved[:k])
    if not top_k:
        return 0.0
    return len(top_k & expected) / len(top_k)


def mrr(retrieved: list[str], expected: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item."""
    for i, name in enumerate(retrieved):
        if name in expected:
            return 1.0 / (i + 1)
    return 0.0


@dataclass
class AdapterResult:
    """Results for a single adapter across all queries."""

    adapter_name: str
    recalls: list[float] = field(default_factory=list)
    precisions: list[float] = field(default_factory=list)
    mrrs: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    distractor_counts: list[int] = field(default_factory=list)

    @property
    def avg_recall(self) -> float:
        return statistics.mean(self.recalls) if self.recalls else 0.0

    @property
    def avg_precision(self) -> float:
        return statistics.mean(self.precisions) if self.precisions else 0.0

    @property
    def avg_mrr(self) -> float:
        return statistics.mean(self.mrrs) if self.mrrs else 0.0

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def total_distractors(self) -> int:
        return sum(self.distractor_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter_name,
            "avg_recall": round(self.avg_recall, 4),
            "avg_precision": round(self.avg_precision, 4),
            "avg_mrr": round(self.avg_mrr, 4),
            "p50_latency_ms": round(self.p50_latency, 2),
            "p95_latency_ms": round(self.p95_latency, 2),
            "total_distractors": self.total_distractors,
        }


# ---------------------------------------------------------------------------
# Adapter runners: extract concept names from each adapter's output
# ---------------------------------------------------------------------------


def _extract_names_from_items(items: list[Any]) -> list[str]:
    """Extract concept names from QueryItem-like objects."""
    names = []
    for item in items:
        content = ""
        if hasattr(item, "content"):
            content = item.content
        elif hasattr(item, "name"):
            content = item.name
        elif isinstance(item, dict):
            content = item.get("content", "") or item.get("name", "")

        for concept_name, _ in AUTH_CONCEPTS:
            if concept_name.lower() in content.lower() and concept_name not in names:
                names.append(concept_name)
    return names


def run_qortex_native(
    client: LocalQortexClient, query: str, top_k: int, mode: str = "auto"
) -> list[str]:
    """Run qortex native retrieval (graph or vec mode)."""
    result = client.query(context=query, top_k=top_k, mode=mode)
    return _extract_names_from_items(result.items)


def run_agno(client: LocalQortexClient, query: str, top_k: int) -> list[str]:
    """Run via agno adapter."""
    from qortex.adapters.agno import QortexKnowledge

    knowledge = QortexKnowledge(client=client, domains=["auth"], top_k=top_k)
    docs = knowledge.retrieve(query, top_k=top_k)
    return _extract_names_from_items(docs)


def run_langchain(client: LocalQortexClient, query: str, top_k: int) -> list[str]:
    """Run via langchain adapter."""
    try:
        from qortex.adapters.langchain import QortexRetriever

        retriever = QortexRetriever(client=client, domains=["auth"], top_k=top_k)
        docs = retriever.invoke(query)
        return _extract_names_from_items(docs)
    except ImportError:
        return []


def run_autogen(client: LocalQortexClient, query: str, top_k: int) -> list[str]:
    """Run via autogen adapter."""
    import asyncio

    from qortex.adapters.autogen import QortexMemory

    memory = QortexMemory(client=client, domains=["auth"], top_k=top_k)
    result = asyncio.get_event_loop().run_until_complete(memory.query(query))
    items = result.results if hasattr(result, "results") else result.get("results", [])
    return _extract_names_from_items(items)


def run_mastra(client: LocalQortexClient, query: str, top_k: int) -> list[str]:
    """Run via mastra adapter (vector query only — no text query)."""
    # Mastra is a vector store interface, not text-query. Skip in this benchmark.
    return []


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


ADAPTERS: dict[str, Any] = {
    "qortex-graph": lambda c, q, k: run_qortex_native(c, q, k, mode="graph"),
    "qortex-vec": lambda c, q, k: run_qortex_native(c, q, k, mode="vec"),
    "agno": run_agno,
    "langchain": run_langchain,
    "autogen": run_autogen,
}


def setup_client() -> LocalQortexClient:
    """Create a LocalQortexClient with the auth corpus loaded."""
    from qortex.vec.embeddings import SentenceTransformerEmbedding

    embedding = SentenceTransformerEmbedding()
    vi = NumpyVectorIndex(dimensions=embedding.dimensions)
    backend = InMemoryBackend(vector_index=vi)
    backend.connect()

    client = LocalQortexClient(
        vector_index=vi,
        backend=backend,
        embedding_model=embedding,
        mode="auto",
    )

    client.ingest_structured(
        domain="auth",
        concepts=[{"name": n, "description": d} for n, d in AUTH_CONCEPTS],
        edges=[{"source": s, "target": t, "relation_type": r} for s, t, r in AUTH_EDGES],
        rules=[{"text": t, "category": c} for t, c in AUTH_RULES],
    )

    return client


def run_benchmark(top_k: int = 5, output: str | None = None) -> dict[str, Any]:
    """Run all adapters on all queries and compute metrics."""
    print("Setting up client with auth corpus...")
    client = setup_client()

    results: dict[str, AdapterResult] = {}

    for adapter_name, run_fn in ADAPTERS.items():
        print(f"\nRunning: {adapter_name}")
        ar = AdapterResult(adapter_name=adapter_name)

        for case in EVAL_QUERIES:
            query = case["query"]
            expected = case["expected"]
            distractors = case.get("distractors", set())

            t0 = time.perf_counter()
            try:
                names = run_fn(client, query, top_k)
            except Exception as e:
                print(f"  ERROR: {adapter_name} failed on '{query[:50]}': {e}")
                names = []
            elapsed_ms = (time.perf_counter() - t0) * 1000

            ar.recalls.append(recall_at_k(names, expected, k=top_k))
            ar.precisions.append(precision_at_k(names, expected, k=top_k))
            ar.mrrs.append(mrr(names, expected))
            ar.latencies_ms.append(elapsed_ms)
            ar.distractor_counts.append(len(set(names) & distractors))

        results[adapter_name] = ar

    # Print results table
    print(f"\n{'Adapter':<16} {'Recall@{}'.format(top_k):>10} {'Prec@{}'.format(top_k):>10} "
          f"{'MRR':>8} {'p50ms':>8} {'p95ms':>8} {'Dist':>6}")
    print("-" * 72)
    for ar in results.values():
        print(
            f"{ar.adapter_name:<16} {ar.avg_recall:>10.3f} {ar.avg_precision:>10.3f} "
            f"{ar.avg_mrr:>8.3f} {ar.p50_latency:>8.1f} {ar.p95_latency:>8.1f} "
            f"{ar.total_distractors:>6d}"
        )

    # Save JSON
    output_data = {
        "top_k": top_k,
        "queries": len(EVAL_QUERIES),
        "adapters": {name: ar.to_dict() for name, ar in results.items()},
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {out_path}")

    return output_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Comparative adapter benchmark")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--output", "-o", help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(top_k=args.top_k, output=args.output)


if __name__ == "__main__":
    main()
