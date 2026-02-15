"""Tests for the AutoGen (AG2) Memory adapter.

Tests verify:
1. QortexMemory implements all 5 async Memory methods
2. query() calls client.query() and returns MemoryContent-compatible results
3. add() calls client.ingest_text()
4. update_context() extracts last message, queries, injects system message
5. clear() and close() are safe no-ops
6. feedback() closes the learning loop (qortex extension)
7. Config serialization round-trips

Run: uv run pytest tests/test_autogen_adapter.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from qortex.adapters.autogen import _HAS_AUTOGEN, QortexMemory, QortexMemoryConfig
from qortex.client import (
    FeedbackResult,
    IngestResult,
    LocalQortexClient,
    QueryItem,
    QueryResult,
    RuleItem,
)
from qortex.core.memory import InMemoryBackend
from qortex.vec.embeddings import SentenceTransformerEmbedding
from qortex.vec.index import NumpyVectorIndex

# ---------------------------------------------------------------------------
# Auth corpus (same as crewai/agno tests for consistency)
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
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_model():
    return SentenceTransformerEmbedding()


@pytest.fixture(scope="module")
def real_client(embedding_model):
    """A real LocalQortexClient with the auth corpus loaded."""
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
    return client


@pytest.fixture
def memory(real_client):
    """QortexMemory backed by a real client with auth corpus."""
    return QortexMemory(
        client=real_client,
        domains=["auth"],
        top_k=5,
        score_threshold=0.0,
    )


@pytest.fixture
def mock_client():
    """A mock QortexClient for unit tests."""
    client = MagicMock()
    client.query.return_value = QueryResult(
        items=[
            QueryItem(
                id="test:1",
                content="OAuth2: Authorization framework",
                score=0.92,
                domain="auth",
                node_id="auth:oauth2",
            ),
            QueryItem(
                id="test:2",
                content="PKCE: Proof Key for Code Exchange",
                score=0.85,
                domain="auth",
                node_id="auth:pkce",
            ),
        ],
        query_id="mock-query-id",
        rules=[
            RuleItem(
                id="rule:1",
                text="Always use PKCE for public clients",
                domain="auth",
                category="security",
            ),
        ],
    )
    client.ingest_text.return_value = IngestResult(
        domain="auth",
        source="raw_text",
        concepts=1,
        edges=0,
        rules=0,
    )
    client.feedback.return_value = FeedbackResult(
        status="recorded",
        query_id="mock-query-id",
        outcome_count=1,
        source="autogen",
    )
    return client


@pytest.fixture
def mock_memory(mock_client):
    """QortexMemory backed by a mock client."""
    return QortexMemory(
        client=mock_client,
        domains=["auth"],
        top_k=5,
    )


# ---------------------------------------------------------------------------
# Unit tests (mock client)
# ---------------------------------------------------------------------------


class TestQortexMemoryUnit:
    """Unit tests using mocked QortexClient."""

    @pytest.mark.asyncio
    async def test_query_calls_client(self, mock_memory, mock_client):
        await mock_memory.query("What is OAuth2?")
        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args
        assert call_kwargs.kwargs["context"] == "What is OAuth2?"
        assert call_kwargs.kwargs["domains"] == ["auth"]
        assert call_kwargs.kwargs["top_k"] == 5

    @pytest.mark.asyncio
    async def test_query_returns_results(self, mock_memory):
        result = await mock_memory.query("What is OAuth2?")
        if _HAS_AUTOGEN:
            assert hasattr(result, "results")
            assert len(result.results) == 2
            assert result.results[0].metadata["score"] == 0.92
        else:
            assert "results" in result
            assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_query_accepts_string(self, mock_memory):
        await mock_memory.query("test query")
        assert mock_memory.last_query_id == "mock-query-id"

    @pytest.mark.asyncio
    async def test_query_accepts_dict(self, mock_memory):
        await mock_memory.query({"content": "test query"})
        call_kwargs = mock_memory._client.query.call_args
        assert call_kwargs.kwargs["context"] == "test query"

    @pytest.mark.asyncio
    async def test_query_stores_query_id(self, mock_memory):
        await mock_memory.query("test")
        assert mock_memory.last_query_id == "mock-query-id"

    @pytest.mark.asyncio
    async def test_add_calls_ingest_text(self, mock_memory, mock_client):
        await mock_memory.add("New concept about OAuth2 scopes")
        mock_client.ingest_text.assert_called_once_with(
            text="New concept about OAuth2 scopes",
            domain="auth",
        )

    @pytest.mark.asyncio
    async def test_add_accepts_dict(self, mock_memory, mock_client):
        await mock_memory.add({"content": "New concept"})
        mock_client.ingest_text.assert_called_once()
        assert mock_client.ingest_text.call_args.kwargs["text"] == "New concept"

    @pytest.mark.asyncio
    async def test_add_skips_empty(self, mock_memory, mock_client):
        await mock_memory.add("")
        mock_client.ingest_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_is_noop(self, mock_memory):
        await mock_memory.clear()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_is_noop(self, mock_memory):
        await mock_memory.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_feedback_calls_client(self, mock_memory, mock_client):
        await mock_memory.query("test")
        await mock_memory.feedback({"auth:oauth2": "accepted"})
        mock_client.feedback.assert_called_once_with(
            query_id="mock-query-id",
            outcomes={"auth:oauth2": "accepted"},
            source="autogen",
        )

    @pytest.mark.asyncio
    async def test_feedback_noop_without_query(self, mock_memory, mock_client):
        """Feedback before any query is a no-op."""
        await mock_memory.feedback({"auth:oauth2": "accepted"})
        mock_client.feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_context_with_dict_messages(self, mock_memory):
        """update_context works with plain list of message dicts."""
        context = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How does OAuth2 work?"},
        ]
        await mock_memory.update_context(context)
        assert mock_memory._client.query.called
        call_kwargs = mock_memory._client.query.call_args
        assert call_kwargs.kwargs["context"] == "How does OAuth2 work?"

    @pytest.mark.asyncio
    async def test_update_context_empty_messages(self, mock_memory):
        """update_context with empty messages returns empty result."""
        _result = await mock_memory.update_context([])
        mock_memory._client.query.assert_not_called()

    def test_config_defaults(self):
        client = MagicMock()
        mem = QortexMemory(client=client)
        assert mem.config.domains is None
        assert mem.config.top_k == 5
        assert mem.config.score_threshold == 0.0
        assert mem.config.feedback_source == "autogen"

    def test_config_custom(self):
        client = MagicMock()
        cfg = QortexMemoryConfig(domains=["auth"], top_k=10, score_threshold=0.5)
        mem = QortexMemory(client=client, config=cfg)
        assert mem.config.domains == ["auth"]
        assert mem.config.top_k == 10
        assert mem.config.score_threshold == 0.5

    def test_to_config_roundtrip(self, mock_memory):
        cfg = mock_memory._to_config()
        assert cfg.domains == ["auth"]
        assert cfg.top_k == 5

    def test_component_type(self):
        assert QortexMemory.component_type == "memory"


# ---------------------------------------------------------------------------
# Integration tests (real client)
# ---------------------------------------------------------------------------


class TestQortexMemoryIntegration:
    """Integration tests using a real LocalQortexClient + InMemoryBackend."""

    @pytest.mark.asyncio
    async def test_query_returns_relevant_results(self, memory):
        result = await memory.query("What is OAuth2?")
        results = result.results if hasattr(result, "results") else result["results"]
        assert len(results) > 0
        # OAuth2 should be in the top results
        contents = []
        for r in results:
            c = r.content if hasattr(r, "content") else r["content"]
            contents.append(c.lower())
        assert any("oauth2" in c for c in contents)

    @pytest.mark.asyncio
    async def test_query_scores_are_normalized(self, memory):
        result = await memory.query("OAuth2 authentication")
        results = result.results if hasattr(result, "results") else result["results"]
        for r in results:
            meta = r.metadata if hasattr(r, "metadata") else r["metadata"]
            score = meta["score"]
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_query_includes_graph_metadata(self, memory):
        result = await memory.query("OAuth2")
        results = result.results if hasattr(result, "results") else result["results"]
        if results:
            meta = results[0].metadata if hasattr(results[0], "metadata") else results[0]["metadata"]
            assert "domain" in meta
            assert "node_id" in meta
            assert meta["domain"] == "auth"

    @pytest.mark.asyncio
    async def test_cross_cutting_query(self, memory):
        """Graph-enhanced retrieval should find related concepts."""
        result = await memory.query("Enterprise SSO for corporate apps")
        results = result.results if hasattr(result, "results") else result["results"]
        contents = []
        for r in results:
            c = r.content if hasattr(r, "content") else r["content"]
            contents.append(c.lower())
        combined = " ".join(contents)
        # Should find SAML and/or OpenID Connect via graph edges
        assert "saml" in combined or "openid" in combined

    @pytest.mark.asyncio
    async def test_update_context_with_messages(self, memory):
        messages = [
            {"role": "user", "content": "How should a mobile app handle OAuth2?"},
        ]
        _result = await memory.update_context(messages)
        assert memory.last_query_id is not None

    @pytest.mark.asyncio
    async def test_feedback_after_query(self, memory):
        result = await memory.query("OAuth2 PKCE")
        results = result.results if hasattr(result, "results") else result["results"]
        if results:
            meta = results[0].metadata if hasattr(results[0], "metadata") else results[0]["metadata"]
            node_id = meta["node_id"]
            await memory.feedback({node_id: "accepted"})
            # Should not raise

    @pytest.mark.asyncio
    async def test_all_five_methods_exist(self, memory):
        """Verify all 5 Memory ABC methods are present and async."""
        assert asyncio.iscoroutinefunction(memory.update_context)
        assert asyncio.iscoroutinefunction(memory.query)
        assert asyncio.iscoroutinefunction(memory.add)
        assert asyncio.iscoroutinefunction(memory.clear)
        assert asyncio.iscoroutinefunction(memory.close)

    @pytest.mark.asyncio
    async def test_query_result_shape(self, memory):
        """Result shape matches MemoryQueryResult regardless of autogen presence."""
        result = await memory.query("OAuth2")
        if _HAS_AUTOGEN:
            from autogen_core.memory import MemoryQueryResult

            assert isinstance(result, MemoryQueryResult)
        else:
            assert isinstance(result, dict)
            assert "results" in result
            for r in result["results"]:
                assert "content" in r
                assert "mime_type" in r
                assert "metadata" in r
