"""Tests for the qortex REST API.

Uses Starlette's TestClient (ASGI-level, no real HTTP server needed).
Same fixture pattern as test_mcp_server.py.
"""

from __future__ import annotations

import hashlib

import pytest
from starlette.testclient import TestClient

from qortex.api.app import create_app
from qortex.api.middleware import AuthConfig
from qortex.core.memory import InMemoryBackend
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
def vector_index():
    return NumpyVectorIndex(dimensions=DIMS)


@pytest.fixture
def backend(vector_index) -> InMemoryBackend:
    b = InMemoryBackend(vector_index=vector_index)
    b.connect()
    return b


@pytest.fixture
def service(backend, vector_index) -> QortexService:
    return QortexService(
        backend=backend,
        vector_index=vector_index,
        embedding_model=FakeEmbedding(),
    )


@pytest.fixture
def client(service) -> TestClient:
    # Disable auth for API tests (auth has its own test file)
    auth_config = AuthConfig.__new__(AuthConfig)
    auth_config.enabled = False
    auth_config._key_hashes = set()
    auth_config._hmac_secret = None
    auth_config._hmac_max_age = 300
    app = create_app(service=service, auth_config=auth_config)
    return TestClient(app)


def _seed(client: TestClient):
    """Seed data via the API."""
    client.post(
        "/v1/ingest/structured",
        json={
            "concepts": [
                {"name": "Python", "description": "A programming language"},
                {"name": "FastAPI", "description": "A web framework"},
            ],
            "domain": "tech",
        },
    )


class TestHealth:
    def test_health(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestStatus:
    def test_status(self, client):
        resp = client.get("/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["backend"] == "InMemoryBackend"


class TestDomains:
    def test_domains_empty(self, client):
        resp = client.get("/v1/domains")
        assert resp.status_code == 200
        assert resp.json()["domains"] == []

    def test_domains_after_ingest(self, client):
        _seed(client)
        resp = client.get("/v1/domains")
        names = [d["name"] for d in resp.json()["domains"]]
        assert "tech" in names


class TestQuery:
    def test_query_empty_graph(self, client):
        resp = client.post("/v1/query", json={"context": "hello"})
        assert resp.status_code == 200
        assert resp.json()["items"] == []

    def test_query_with_data(self, client):
        _seed(client)
        resp = client.post("/v1/query", json={"context": "python programming"})
        assert resp.status_code == 200
        assert len(resp.json()["items"]) > 0

    def test_query_missing_context(self, client):
        resp = client.post("/v1/query", json={})
        assert resp.status_code == 400

    def test_query_invalid_json(self, client):
        resp = client.post("/v1/query", content=b"not json")
        assert resp.status_code == 400


class TestFeedback:
    def test_feedback(self, client):
        resp = client.post(
            "/v1/feedback",
            json={
                "query_id": "q1",
                "outcomes": {"item1": "accepted"},
                "source": "test",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_feedback_missing_fields(self, client):
        resp = client.post("/v1/feedback", json={"query_id": "q1"})
        assert resp.status_code == 400


class TestIngest:
    def test_ingest_structured(self, client):
        resp = client.post(
            "/v1/ingest/structured",
            json={
                "concepts": [{"name": "Test", "description": "A test concept"}],
                "domain": "test",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["concepts"] == 1
        assert data["domain"] == "test"

    def test_ingest_text(self, client):
        resp = client.post(
            "/v1/ingest/text",
            json={"text": "Machine learning is AI", "domain": "ml"},
        )
        assert resp.status_code == 200
        assert resp.json()["domain"] == "ml"

    def test_ingest_text_missing_fields(self, client):
        resp = client.post("/v1/ingest/text", json={"text": "hello"})
        assert resp.status_code == 400

    def test_ingest_file_not_found(self, client):
        resp = client.post(
            "/v1/ingest",
            json={"source_path": "/nonexistent.md", "domain": "test"},
        )
        assert resp.status_code == 200  # returns error in body, not HTTP error
        assert "error" in resp.json()


class TestExplore:
    def test_explore_not_found(self, client):
        resp = client.post("/v1/explore", json={"node_id": "nonexistent"})
        assert resp.status_code == 404

    def test_explore_found(self, client, service):
        _seed(client)
        nodes = list(service.backend._nodes.values())
        if nodes:
            resp = client.post("/v1/explore", json={"node_id": nodes[0].id})
            assert resp.status_code == 200
            assert resp.json()["node"]["id"] == nodes[0].id


class TestRules:
    def test_rules_empty(self, client):
        resp = client.post("/v1/rules", json={})
        assert resp.status_code == 200
        assert resp.json()["rules"] == []


class TestStats:
    def test_stats(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "knowledge" in data
        assert "learning" in data
        assert "activity" in data


class TestLearning:
    def test_learning_select(self, client):
        resp = client.post(
            "/v1/learning/select",
            json={
                "learner": "test",
                "candidates": [
                    {"id": "arm1", "metadata": {}},
                    {"id": "arm2", "metadata": {}},
                ],
                "k": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["selected_arms"]) == 1

    def test_learning_observe(self, client):
        # Select first to create the learner
        client.post(
            "/v1/learning/select",
            json={
                "learner": "test",
                "candidates": [{"id": "arm1", "metadata": {}}],
            },
        )
        resp = client.post(
            "/v1/learning/observe",
            json={"learner": "test", "arm_id": "arm1", "reward": 1.0},
        )
        assert resp.status_code == 200
        assert "alpha" in resp.json()

    def test_learning_posteriors(self, client):
        client.post(
            "/v1/learning/select",
            json={
                "learner": "test",
                "candidates": [{"id": "arm1", "metadata": {}}],
            },
        )
        resp = client.get("/v1/learning/test/posteriors")
        assert resp.status_code == 200
        assert resp.json()["learner"] == "test"

    def test_learning_metrics(self, client):
        client.post(
            "/v1/learning/select",
            json={
                "learner": "test",
                "candidates": [{"id": "arm1", "metadata": {}}],
            },
        )
        resp = client.get("/v1/learning/test/metrics")
        assert resp.status_code == 200


class TestIngestMessage:
    def test_ingest_message(self, client):
        resp = client.post(
            "/v1/ingest/message",
            json={"text": "Hello world from session", "session_id": "sess-001"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-001"
        assert data["chunks"] >= 1

    def test_ingest_message_missing_fields(self, client):
        resp = client.post("/v1/ingest/message", json={"text": "hello"})
        assert resp.status_code == 400

    def test_ingest_message_empty_text(self, client):
        resp = client.post(
            "/v1/ingest/message",
            json={"text": "", "session_id": "sess-002"},
        )
        # Empty text is rejected at route validation level
        assert resp.status_code == 400


class TestLearningSessionsAndReset:
    def test_session_start_and_end(self, client):
        resp = client.post(
            "/v1/learning/sessions/start",
            json={"learner": "test", "session_name": "test-session"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["learner"] == "test"

        # End the session
        resp = client.post(
            "/v1/learning/sessions/end",
            json={"session_id": data["session_id"]},
        )
        assert resp.status_code == 200

    def test_session_start_missing_fields(self, client):
        resp = client.post(
            "/v1/learning/sessions/start",
            json={"learner": "test"},
        )
        assert resp.status_code == 400

    def test_learning_reset(self, client):
        # Create a learner first
        client.post(
            "/v1/learning/select",
            json={
                "learner": "reset-test",
                "candidates": [{"id": "arm1", "metadata": {}}],
            },
        )
        resp = client.post(
            "/v1/learning/reset",
            json={"learner": "reset-test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reset"
        assert data["learner"] == "reset-test"

    def test_learning_reset_missing_learner(self, client):
        resp = client.post("/v1/learning/reset", json={})
        assert resp.status_code == 400


class TestSources:
    def test_source_list_empty(self, client):
        resp = client.get("/v1/sources")
        assert resp.status_code == 200
        assert resp.json()["sources"] == []
