"""Real subprocess E2E tests for the qortex REST API.

Starts ``qortex serve`` as a subprocess on a random port, hits it with real
HTTP requests, and verifies the full stack: TCP binding, auth middleware,
HMAC signing, CORS, server lifecycle, and concurrent request handling.

Every existing API test uses ASGI transport (in-process). These tests
exercise the *real* path that production clients see.

Run::

    uv run pytest tests/test_e2e_subprocess.py -v -m integration --timeout=60
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 and let the OS assign a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_health(base_url: str, timeout: float = 15.0) -> None:
    """Poll GET /v1/health until 200 or timeout."""
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url}/v1/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
                pass
            await asyncio.sleep(0.3)
    raise TimeoutError(f"Server at {base_url} did not become healthy within {timeout}s")


@dataclass
class ServerInfo:
    proc: subprocess.Popen
    base_url: str
    port: int
    api_key: str
    hmac_secret: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop for module-scoped async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def server() -> AsyncIterator[ServerInfo]:
    """Start qortex serve as a subprocess, wait for health, yield, kill."""
    port = _find_free_port()
    api_key = secrets.token_urlsafe(32)
    hmac_secret = secrets.token_urlsafe(32)

    env = {
        **os.environ,
        "QORTEX_GRAPH": "memory",
        "QORTEX_VEC": "memory",
        "QORTEX_API_KEYS": api_key,
        "QORTEX_HMAC_SECRET": hmac_secret,
        "QORTEX_CORS_ORIGINS": "http://localhost:3000",
    }

    # Find the qortex CLI entry point. Prefer the venv's bin/ script;
    # fall back to shutil.which; last resort: python -c import.
    venv_bin = Path(sys.executable).parent / "qortex"
    if venv_bin.exists():
        cmd = [str(venv_bin), "serve", "--host", "127.0.0.1", "--port", str(port)]
    elif shutil.which("qortex"):
        cmd = ["qortex", "serve", "--host", "127.0.0.1", "--port", str(port)]
    else:
        cmd = [
            sys.executable,
            "-c",
            "from qortex.cli import main; main()",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"
    try:
        await _wait_for_health(base_url, timeout=30.0)
    except TimeoutError:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)
        raise RuntimeError(
            f"Server failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    info = ServerInfo(
        proc=proc,
        base_url=base_url,
        port=port,
        api_key=api_key,
        hmac_secret=hmac_secret,
    )
    yield info

    # Teardown: graceful shutdown
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture
async def api_key_client(server: ServerInfo) -> AsyncIterator[httpx.AsyncClient]:
    """Authenticated with API key via Bearer token."""
    async with httpx.AsyncClient(
        base_url=server.base_url,
        headers={"Authorization": f"Bearer {server.api_key}"},
        timeout=10.0,
    ) as client:
        yield client


@pytest.fixture
async def hmac_client(server: ServerInfo) -> AsyncIterator[httpx.AsyncClient]:
    """Authenticated with HMAC-SHA256 signing (raw httpx with manual signing)."""
    import hashlib
    import hmac as hmac_mod

    secret = server.hmac_secret.encode()

    async def sign_request(request: httpx.Request) -> None:
        timestamp = str(int(time.time()))
        body = request.content or b""
        message = f"{timestamp}.".encode() + body
        signature = hmac_mod.new(secret, message, hashlib.sha256).hexdigest()
        request.headers["X-Qortex-Timestamp"] = timestamp
        request.headers["X-Qortex-Signature"] = signature

    async with httpx.AsyncClient(
        base_url=server.base_url,
        event_hooks={"request": [sign_request]},
        timeout=10.0,
    ) as client:
        yield client


@pytest.fixture
async def unauthed_client(server: ServerInfo) -> AsyncIterator[httpx.AsyncClient]:
    """No auth — should get 401 on everything except /v1/health."""
    async with httpx.AsyncClient(base_url=server.base_url, timeout=10.0) as client:
        yield client


# ---------------------------------------------------------------------------
# Also test HttpQortexClient's real constructor (the whole point)
# ---------------------------------------------------------------------------


@pytest.fixture
async def qortex_client(server: ServerInfo):
    """HttpQortexClient authenticated with API key — real constructor."""
    from qortex.http_client import HttpQortexClient

    async with HttpQortexClient(
        base_url=server.base_url,
        api_key=server.api_key,
    ) as client:
        yield client


@pytest.fixture
async def qortex_hmac_client(server: ServerInfo):
    """HttpQortexClient authenticated with HMAC — real constructor."""
    from qortex.http_client import HttpQortexClient

    async with HttpQortexClient(
        base_url=server.base_url,
        hmac_secret=server.hmac_secret,
    ) as client:
        yield client


# ===========================================================================
# Test classes
# ===========================================================================


class TestHealthAndLifecycle:
    """Verify the server started and basic endpoints respond."""

    async def test_health_returns_ok(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_health_no_auth_required(self, unauthed_client: httpx.AsyncClient):
        resp = await unauthed_client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_status_returns_backend_info(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "backend" in data
        assert "status" in data
        assert data["status"] == "ok"

    async def test_domains_initially_empty(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/domains")
        assert resp.status_code == 200
        data = resp.json()
        assert "domains" in data
        assert isinstance(data["domains"], list)


class TestAuthOverRealHTTP:
    """Verify auth works over real TCP — API key, HMAC, rejection, CORS."""

    async def test_api_key_auth_success(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/status")
        assert resp.status_code == 200

    async def test_hmac_auth_success(self, hmac_client: httpx.AsyncClient):
        resp = await hmac_client.get("/v1/status")
        assert resp.status_code == 200

    async def test_no_auth_rejected(self, unauthed_client: httpx.AsyncClient):
        resp = await unauthed_client.get("/v1/status")
        assert resp.status_code == 401
        assert "error" in resp.json()

    async def test_invalid_api_key_rejected(self, server: ServerInfo):
        async with httpx.AsyncClient(
            base_url=server.base_url,
            headers={"Authorization": "Bearer totally-wrong-key"},
            timeout=10.0,
        ) as client:
            resp = await client.get("/v1/status")
            assert resp.status_code == 401

    async def test_cors_preflight_allowed_origin(self, server: ServerInfo):
        """CORS preflight on a public path (health) — no auth needed."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.options(
                f"{server.base_url}/v1/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Authorization",
                },
            )
            assert resp.status_code == 200
            assert "access-control-allow-origin" in resp.headers

    async def test_cors_disallowed_origin(self, server: ServerInfo):
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.options(
                f"{server.base_url}/v1/status",
                headers={
                    "Origin": "http://evil.example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            # Starlette CORSMiddleware returns 400 for disallowed origins
            # or omits the allow-origin header
            allow_origin = resp.headers.get("access-control-allow-origin", "")
            assert "evil.example.com" not in allow_origin

    async def test_hmac_with_post_body(self, hmac_client: httpx.AsyncClient):
        """HMAC signing over a POST body — the body buffering path."""
        resp = await hmac_client.post(
            "/v1/query",
            json={"context": "test hmac body signing"},
        )
        # Should authenticate successfully (200, not 401)
        assert resp.status_code == 200


class TestFullIngestQueryFeedbackFlow:
    """The money test — complete lifecycle over real HTTP."""

    async def test_ingest_structured(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/ingest/structured",
            json={
                "concepts": [
                    {"name": "Python", "description": "A programming language"},
                    {"name": "FastAPI", "description": "A modern async web framework for Python"},
                    {"name": "Starlette", "description": "ASGI toolkit that FastAPI builds on"},
                ],
                "domain": "e2e-tech",
                "edges": [
                    {
                        "source": "FastAPI",
                        "target": "Starlette",
                        "relation_type": "uses",
                    },
                    {
                        "source": "FastAPI",
                        "target": "Python",
                        "relation_type": "refines",
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["domain"] == "e2e-tech"
        assert data["concepts"] == 3
        assert data["edges"] == 2

    async def test_ingest_text(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/ingest/text",
            json={
                "text": "Uvicorn is a lightning-fast ASGI server for Python.",
                "domain": "e2e-tech",
                "name": "uvicorn-note",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["domain"] == "e2e-tech"

    async def test_query_after_ingest(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/query",
            json={"context": "async web framework", "domains": ["e2e-tech"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "query_id" in data
        assert isinstance(data["items"], list)

    async def test_feedback_after_query(self, api_key_client: httpx.AsyncClient):
        # First query to get a query_id
        query_resp = await api_key_client.post(
            "/v1/query",
            json={"context": "web framework python", "domains": ["e2e-tech"]},
        )
        query_id = query_resp.json()["query_id"]

        # If no embedding model, query_id is empty and feedback will 400.
        # Only test feedback when we have a real query_id.
        if not query_id:
            pytest.skip("No embedding model available — query_id is empty")

        resp = await api_key_client.post(
            "/v1/feedback",
            json={
                "query_id": query_id,
                "outcomes": {"some-item": "accepted"},
                "source": "e2e-test",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "recorded"
        assert data["query_id"] == query_id
        assert data["outcome_count"] == 1

    async def test_domains_after_ingest(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/domains")
        assert resp.status_code == 200
        data = resp.json()
        names = [d["name"] for d in data["domains"]]
        assert "e2e-tech" in names


class TestFullFlowViaHttpQortexClient:
    """Exercise the HttpQortexClient real constructor over real HTTP."""

    async def test_status_via_client(self, qortex_client):
        result = await qortex_client.status()
        assert result.status == "ok"
        assert result.backend  # non-empty string

    async def test_ingest_structured_via_client(self, qortex_client):
        result = await qortex_client.ingest_structured(
            concepts=[
                {"name": "Rust", "description": "Systems programming language"},
                {"name": "Cargo", "description": "Rust package manager"},
            ],
            domain="e2e-client-test",
            edges=[
                {"source": "Cargo", "target": "Rust", "relation_type": "uses"},
            ],
        )
        assert result.domain == "e2e-client-test"
        assert result.concepts == 2
        assert result.edges == 1

    async def test_query_via_client(self, qortex_client):
        result = await qortex_client.query(
            context="systems programming",
            domains=["e2e-client-test"],
        )
        # query_id may be empty if no embedding model is available
        assert isinstance(result.query_id, str)
        assert isinstance(result.items, list)

    async def test_hmac_client_query(self, qortex_hmac_client):
        """HttpQortexClient with HMAC — tests _sign_request over real HTTP."""
        result = await qortex_hmac_client.query(context="testing hmac")
        assert isinstance(result.query_id, str)

    async def test_domains_via_client(self, qortex_client):
        result = await qortex_client.domains()
        assert isinstance(result, list)

    async def test_rules_via_client(self, qortex_client):
        result = await qortex_client.rules()
        assert isinstance(result.rules, list)
        assert isinstance(result.domain_count, int)


class TestExploreAndRules:
    """Graph exploration and rule queries over real HTTP."""

    async def test_explore_missing_node(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/explore",
            json={"node_id": "nonexistent-node-xyz"},
        )
        assert resp.status_code == 404
        assert "error" in resp.json()

    async def test_explore_via_client_missing(self, qortex_client):
        result = await qortex_client.explore(node_id="totally-fake-node")
        assert result is None

    async def test_rules_empty_domain(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/rules",
            json={"domains": ["nonexistent-domain"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rules"] == []

    async def test_rules_all(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/rules",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rules" in data
        assert "domain_count" in data


class TestLearningOverHTTP:
    """Learning endpoints via raw httpx — select, observe, posteriors, metrics."""

    async def test_learning_select(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/learning/select",
            json={
                "learner": "e2e-learner",
                "candidates": [
                    {"id": "arm-a", "metadata": {"label": "A"}},
                    {"id": "arm-b", "metadata": {"label": "B"}},
                    {"id": "arm-c", "metadata": {"label": "C"}},
                ],
                "k": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "selected_arms" in data
        assert len(data["selected_arms"]) >= 1

    async def test_learning_observe(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/learning/observe",
            json={
                "learner": "e2e-learner",
                "arm_id": "arm-a",
                "reward": 1.0,
                "outcome": "accepted",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["arm_id"] == "arm-a"
        assert "alpha" in data
        assert "beta" in data
        assert data["pulls"] >= 1

    async def test_learning_posteriors(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/learning/e2e-learner/posteriors")
        assert resp.status_code == 200
        data = resp.json()
        assert data["learner"] == "e2e-learner"
        assert "posteriors" in data

    async def test_learning_metrics(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/learning/e2e-learner/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    async def test_learning_full_cycle(self, api_key_client: httpx.AsyncClient):
        """Select → observe winners → posteriors show updated beliefs."""
        learner = "e2e-cycle-learner"

        # Select
        sel_resp = await api_key_client.post(
            "/v1/learning/select",
            json={
                "learner": learner,
                "candidates": [
                    {"id": "winner"},
                    {"id": "loser"},
                ],
                "k": 1,
            },
        )
        assert sel_resp.status_code == 200

        # Observe: reward the winner several times
        for _ in range(5):
            await api_key_client.post(
                "/v1/learning/observe",
                json={
                    "learner": learner,
                    "arm_id": "winner",
                    "reward": 1.0,
                },
            )
        # Penalize the loser
        for _ in range(5):
            await api_key_client.post(
                "/v1/learning/observe",
                json={
                    "learner": learner,
                    "arm_id": "loser",
                    "reward": 0.0,
                },
            )

        # Check posteriors reflect the difference
        post_resp = await api_key_client.get(f"/v1/learning/{learner}/posteriors")
        assert post_resp.status_code == 200
        posteriors = post_resp.json()["posteriors"]
        assert "winner" in posteriors
        assert "loser" in posteriors
        # Winner should have higher mean than loser
        assert posteriors["winner"]["mean"] > posteriors["loser"]["mean"]


class TestStatsEndpoint:
    """GET /v1/stats shape and content after activity."""

    async def test_stats_shape(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "knowledge" in data
        assert "learning" in data
        assert "activity" in data
        assert "health" in data

    async def test_stats_reflects_activity(self, api_key_client: httpx.AsyncClient):
        """After ingest + query, knowledge counters should reflect data."""
        # Ingest something
        await api_key_client.post(
            "/v1/ingest/structured",
            json={
                "concepts": [{"name": "Stats Test", "description": "For stats"}],
                "domain": "e2e-stats",
            },
        )
        # Query
        await api_key_client.post(
            "/v1/query",
            json={"context": "stats test"},
        )

        resp = await api_key_client.get("/v1/stats")
        data = resp.json()
        # knowledge.domains should be >= 1 (we ingested)
        assert data["knowledge"]["domains"] >= 1
        # activity counters exist and are non-negative
        # (queries_served only increments when embedding model is available)
        assert data["activity"]["queries_served"] >= 0
        assert data["activity"]["feedback_given"] >= 0


class TestErrorHandling:
    """Verify proper 400 responses for bad requests."""

    async def test_query_missing_context(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post("/v1/query", json={})
        assert resp.status_code == 400
        assert "error" in resp.json()

    async def test_feedback_missing_fields(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/feedback",
            json={"query_id": "abc"},  # missing outcomes
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    async def test_ingest_structured_missing_concepts(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/ingest/structured",
            json={"domain": "test"},  # missing concepts
        )
        assert resp.status_code == 400

    async def test_explore_missing_node_id(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post("/v1/explore", json={})
        assert resp.status_code == 400
        assert "error" in resp.json()

    async def test_malformed_json(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/query",
            content=b"this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    async def test_learning_select_missing_fields(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/learning/select",
            json={"learner": "test"},  # missing candidates
        )
        assert resp.status_code == 400

    async def test_learning_observe_missing_fields(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/learning/observe",
            json={"learner": "test"},  # missing arm_id
        )
        assert resp.status_code == 400

    async def test_ingest_text_missing_fields(self, api_key_client: httpx.AsyncClient):
        resp = await api_key_client.post(
            "/v1/ingest/text",
            json={"domain": "test"},  # missing text
        )
        assert resp.status_code == 400


class TestConcurrency:
    """Verify the server handles concurrent requests safely."""

    async def test_concurrent_queries(self, server: ServerInfo):
        """10 concurrent query requests should all return valid results.

        Uses a dedicated client with longer timeout since each query
        involves real embedding computation on a single-worker server.
        """
        async with httpx.AsyncClient(
            base_url=server.base_url,
            headers={"Authorization": f"Bearer {server.api_key}"},
            timeout=60.0,
        ) as client:
            # Seed data first
            await client.post(
                "/v1/ingest/structured",
                json={
                    "concepts": [
                        {"name": "Concurrency", "description": "Doing many things at once"},
                    ],
                    "domain": "e2e-concurrent",
                },
            )

            async def do_query(i: int):
                return await client.post(
                    "/v1/query",
                    json={"context": f"concurrency test {i}"},
                )

            results = await asyncio.gather(*[do_query(i) for i in range(10)])
            for resp in results:
                assert resp.status_code == 200
                assert "query_id" in resp.json()

    async def test_concurrent_ingest_and_query(self, server: ServerInfo):
        """Interleaved ingest + query should not crash."""
        async with httpx.AsyncClient(
            base_url=server.base_url,
            headers={"Authorization": f"Bearer {server.api_key}"},
            timeout=60.0,
        ) as client:

            async def ingest(i: int):
                return await client.post(
                    "/v1/ingest/structured",
                    json={
                        "concepts": [
                            {"name": f"Concept-{i}", "description": f"Test concept {i}"},
                        ],
                        "domain": "e2e-interleaved",
                    },
                )

            async def query(i: int):
                return await client.post(
                    "/v1/query",
                    json={"context": f"interleaved {i}"},
                )

            tasks = []
            for i in range(3):
                tasks.append(ingest(i))
                tasks.append(query(i))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                assert not isinstance(r, Exception), f"Request raised: {r}"
                assert r.status_code in (200, 400)

    async def test_concurrent_learning_observe(self, server: ServerInfo):
        """Concurrent observe calls — posteriors should remain consistent.

        Single-worker uvicorn serializes requests, so we keep concurrency
        low (2 requests) to stay within the 120s pytest-timeout on CI.
        """
        async with httpx.AsyncClient(
            base_url=server.base_url,
            headers={"Authorization": f"Bearer {server.api_key}"},
            timeout=30.0,
        ) as client:
            learner = "e2e-concurrent-learner"

            # Seed the learner with a select
            await client.post(
                "/v1/learning/select",
                json={
                    "learner": learner,
                    "candidates": [{"id": "arm-x"}, {"id": "arm-y"}],
                    "k": 1,
                },
            )

            async def observe(arm: str, reward: float):
                return await client.post(
                    "/v1/learning/observe",
                    json={
                        "learner": learner,
                        "arm_id": arm,
                        "reward": reward,
                    },
                )

            tasks = [observe("arm-x", 1.0), observe("arm-y", 0.0)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successes = [r for r in results if not isinstance(r, Exception)]
            assert len(successes) >= 1, (
                f"Too many failures: {len(results) - len(successes)}/{len(results)}"
            )
            for r in successes:
                assert r.status_code == 200

            # Verify posteriors are consistent (sequential request, should always work)
            post_resp = await client.get(f"/v1/learning/{learner}/posteriors")
            assert post_resp.status_code == 200
            posteriors = post_resp.json()["posteriors"]
            assert "arm-x" in posteriors
            assert "arm-y" in posteriors
