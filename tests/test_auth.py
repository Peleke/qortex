"""Tests for API authentication middleware.

Covers API key auth, HMAC signature auth, public path bypass,
and disabled-auth passthrough.
"""

from __future__ import annotations

import hashlib
import hmac
import time

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
def service() -> QortexService:
    vi = NumpyVectorIndex(dimensions=DIMS)
    backend = InMemoryBackend(vector_index=vi)
    backend.connect()
    return QortexService(backend=backend, vector_index=vi, embedding_model=FakeEmbedding())


# ---------------------------------------------------------------------------
# Auth disabled (no keys configured)
# ---------------------------------------------------------------------------


class TestAuthDisabled:
    def test_all_endpoints_open_when_no_keys(self, service):
        config = AuthConfig.__new__(AuthConfig)
        config.enabled = False
        config._key_hashes = set()
        config._hmac_secret = None
        config._hmac_max_age = 300

        app = create_app(service=service, auth_config=config)
        client = TestClient(app)

        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/status").status_code == 200
        assert client.get("/v1/domains").status_code == 200


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------


class TestAPIKeyAuth:
    @pytest.fixture
    def auth_config(self):
        config = AuthConfig.__new__(AuthConfig)
        config.enabled = True
        config._key_hashes = {hashlib.sha256(b"test-key-123").hexdigest()}
        config._hmac_secret = None
        config._hmac_max_age = 300
        return config

    @pytest.fixture
    def client(self, service, auth_config):
        app = create_app(service=service, auth_config=auth_config)
        return TestClient(app)

    def test_health_always_open(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    def test_protected_endpoint_rejects_no_key(self, client):
        resp = client.get("/v1/status")
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json()["error"]

    def test_protected_endpoint_rejects_bad_key(self, client):
        resp = client.get("/v1/status", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_protected_endpoint_accepts_valid_key(self, client):
        resp = client.get("/v1/status", headers={"Authorization": "Bearer test-key-123"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_bearer_prefix_required(self, client):
        resp = client.get("/v1/status", headers={"Authorization": "test-key-123"})
        assert resp.status_code == 401

    def test_post_endpoint_with_key(self, client):
        resp = client.post(
            "/v1/query",
            json={"context": "hello"},
            headers={"Authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# HMAC auth
# ---------------------------------------------------------------------------


class TestHMACAuth:
    HMAC_SECRET = "super-secret-hmac-key"

    @pytest.fixture
    def auth_config(self):
        config = AuthConfig.__new__(AuthConfig)
        config.enabled = True
        config._key_hashes = set()  # no API keys — HMAC only
        config._hmac_secret = self.HMAC_SECRET.encode()
        config._hmac_max_age = 300
        return config

    @pytest.fixture
    def client(self, service, auth_config):
        app = create_app(service=service, auth_config=auth_config)
        return TestClient(app)

    def _sign(self, body: bytes, timestamp: int | None = None) -> tuple[str, str]:
        ts = str(timestamp or int(time.time()))
        message = f"{ts}.".encode() + body
        sig = hmac.new(self.HMAC_SECRET.encode(), message, hashlib.sha256).hexdigest()
        return sig, ts

    def test_hmac_valid_signature(self, client):
        body = b'{"context": "test"}'
        sig, ts = self._sign(body)
        resp = client.post(
            "/v1/query",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Qortex-Signature": sig,
                "X-Qortex-Timestamp": ts,
            },
        )
        assert resp.status_code == 200

    def test_hmac_invalid_signature(self, client):
        body = b'{"context": "test"}'
        resp = client.post(
            "/v1/query",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Qortex-Signature": "badhex",
                "X-Qortex-Timestamp": str(int(time.time())),
            },
        )
        assert resp.status_code == 401

    def test_hmac_expired_timestamp(self, client):
        body = b'{"context": "test"}'
        old_ts = int(time.time()) - 600  # 10 minutes ago
        sig, ts = self._sign(body, old_ts)
        resp = client.post(
            "/v1/query",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Qortex-Signature": sig,
                "X-Qortex-Timestamp": ts,
            },
        )
        assert resp.status_code == 401

    def test_hmac_tampered_body(self, client):
        body = b'{"context": "test"}'
        sig, ts = self._sign(body)
        tampered = b'{"context": "hacked"}'
        resp = client.post(
            "/v1/query",
            content=tampered,
            headers={
                "Content-Type": "application/json",
                "X-Qortex-Signature": sig,
                "X-Qortex-Timestamp": ts,
            },
        )
        assert resp.status_code == 401

    def test_health_bypasses_hmac(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Combined auth (API key OR HMAC)
# ---------------------------------------------------------------------------


class TestCombinedAuth:
    HMAC_SECRET = "combo-secret"

    @pytest.fixture
    def auth_config(self):
        config = AuthConfig.__new__(AuthConfig)
        config.enabled = True
        config._key_hashes = {hashlib.sha256(b"combo-key").hexdigest()}
        config._hmac_secret = self.HMAC_SECRET.encode()
        config._hmac_max_age = 300
        return config

    @pytest.fixture
    def client(self, service, auth_config):
        app = create_app(service=service, auth_config=auth_config)
        return TestClient(app)

    def test_api_key_works(self, client):
        resp = client.get("/v1/status", headers={"Authorization": "Bearer combo-key"})
        assert resp.status_code == 200

    def test_hmac_works(self, client):
        body = b'{"context": "test"}'
        ts = str(int(time.time()))
        message = f"{ts}.".encode() + body
        sig = hmac.new(self.HMAC_SECRET.encode(), message, hashlib.sha256).hexdigest()
        resp = client.post(
            "/v1/query",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Qortex-Signature": sig,
                "X-Qortex-Timestamp": ts,
            },
        )
        assert resp.status_code == 200

    def test_neither_rejects(self, client):
        resp = client.get("/v1/status")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# AuthConfig from env vars
# ---------------------------------------------------------------------------


class TestAuthConfigFromEnv:
    def test_loads_api_keys(self, monkeypatch):
        monkeypatch.setenv("QORTEX_API_KEYS", "key1,key2")
        monkeypatch.delenv("QORTEX_HMAC_SECRET", raising=False)
        config = AuthConfig()
        assert config.enabled is True
        assert config.verify_api_key("key1") is True
        assert config.verify_api_key("key2") is True
        assert config.verify_api_key("key3") is False

    def test_loads_hmac_secret(self, monkeypatch):
        monkeypatch.delenv("QORTEX_API_KEYS", raising=False)
        monkeypatch.setenv("QORTEX_HMAC_SECRET", "my-secret")
        config = AuthConfig()
        assert config.enabled is True
        assert config._hmac_secret == b"my-secret"

    def test_disabled_when_empty(self, monkeypatch):
        monkeypatch.delenv("QORTEX_API_KEYS", raising=False)
        monkeypatch.delenv("QORTEX_HMAC_SECRET", raising=False)
        config = AuthConfig()
        assert config.enabled is False

    def test_custom_hmac_max_age(self, monkeypatch):
        monkeypatch.setenv("QORTEX_HMAC_SECRET", "s")
        monkeypatch.setenv("QORTEX_HMAC_MAX_AGE", "60")
        monkeypatch.delenv("QORTEX_API_KEYS", raising=False)
        config = AuthConfig()
        assert config._hmac_max_age == 60
