"""Tests for HttpQortexClient.

Uses Starlette TestClient as the backing server — no real HTTP needed.
Verifies that HttpQortexClient correctly deserializes responses into
protocol result types (QueryResult, DomainInfo, etc.).
"""

from __future__ import annotations

import hashlib

import pytest
from starlette.testclient import TestClient

from qortex.api.app import create_app
from qortex.api.middleware import AuthConfig
from qortex.client import (
    DomainInfo,
    ExploreResult,
    FeedbackResult,
    IngestResult,
    QueryResult,
    RulesResult,
    StatusResult,
)
from qortex.core.memory import InMemoryBackend
from qortex.http_client import HttpQortexClient
from qortex.service import QortexService
from qortex.vec.index import NumpyVectorIndex

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
def service() -> QortexService:
    vi = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vi)
    backend.connect()
    return QortexService(
        backend=backend,
        vector_index=vi,
        embedding_model=FakeEmbedding(),
    )


@pytest.fixture
def http_client(service) -> HttpQortexClient:
    """HttpQortexClient backed by a Starlette TestClient."""
    # Disable auth for HTTP client tests
    auth_config = AuthConfig.__new__(AuthConfig)
    auth_config.enabled = False
    auth_config._key_hashes = set()
    auth_config._hmac_secret = None
    auth_config._hmac_max_age = 300
    app = create_app(service=service, auth_config=auth_config)
    test_client = TestClient(app)

    client = HttpQortexClient.__new__(HttpQortexClient)
    client._client = test_client
    return client


def _seed(service: QortexService):
    service.ingest_structured(
        concepts=[
            {"name": "Python", "description": "Programming language"},
            {"name": "Rust", "description": "Systems language"},
        ],
        domain="languages",
    )


class TestHttpClientQuery:
    def test_query_empty(self, http_client):
        result = http_client.query("hello")
        assert isinstance(result, QueryResult)
        assert result.items == []

    def test_query_with_data(self, http_client, service):
        _seed(service)
        result = http_client.query("python programming")
        assert isinstance(result, QueryResult)
        assert len(result.items) > 0
        assert result.items[0].content


class TestHttpClientStatus:
    def test_status(self, http_client):
        result = http_client.status()
        assert isinstance(result, StatusResult)
        assert result.status == "ok"
        assert result.backend == "InMemoryBackend"
        assert result.vector_search is True


class TestHttpClientDomains:
    def test_domains_empty(self, http_client):
        result = http_client.domains()
        assert isinstance(result, list)
        assert result == []

    def test_domains_after_ingest(self, http_client, service):
        _seed(service)
        result = http_client.domains()
        assert len(result) > 0
        assert isinstance(result[0], DomainInfo)
        assert result[0].name == "languages"


class TestHttpClientIngest:
    def test_ingest_text(self, http_client):
        result = http_client.ingest_text(
            text="Machine learning overview",
            domain="ml",
        )
        assert isinstance(result, IngestResult)
        assert result.domain == "ml"

    def test_ingest_structured(self, http_client):
        result = http_client.ingest_structured(
            concepts=[{"name": "Test", "description": "A test"}],
            domain="test",
        )
        assert isinstance(result, IngestResult)
        assert result.concepts == 1


class TestHttpClientFeedback:
    def test_feedback(self, http_client):
        result = http_client.feedback(
            query_id="q1",
            outcomes={"item1": "accepted"},
            source="test",
        )
        assert isinstance(result, FeedbackResult)
        assert result.status == "recorded"
        assert result.outcome_count == 1


class TestHttpClientExplore:
    def test_explore_not_found(self, http_client):
        result = http_client.explore("nonexistent")
        assert result is None

    def test_explore_found(self, http_client, service):
        _seed(service)
        nodes = list(service.backend._nodes.values())
        if nodes:
            result = http_client.explore(nodes[0].id)
            assert isinstance(result, ExploreResult)
            assert result.node.id == nodes[0].id


class TestHttpClientRules:
    def test_rules_empty(self, http_client):
        result = http_client.rules()
        assert isinstance(result, RulesResult)
        assert result.rules == []
