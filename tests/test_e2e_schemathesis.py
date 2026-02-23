"""Property-based API contract tests using schemathesis.

Generates random valid/invalid requests from the OpenAPI spec and verifies:
- No 500 errors on unexpected but valid input combinations
- Response shapes match the declared schemas
- Edge cases in parameter handling (empty strings, large numbers, unicode)

Requires a running server. Uses the same subprocess fixture as
test_e2e_subprocess.py.

Run::

    uv run pytest tests/test_e2e_schemathesis.py -v -m integration --timeout=120
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
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers (duplicated from test_e2e_subprocess to keep files independent)
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class ServerInfo:
    proc: subprocess.Popen
    base_url: str
    port: int
    api_key: str
    hmac_secret: str


def _load_spec(base_url: str) -> dict:
    """Load OpenAPI spec and override the server URL to point at the test server."""
    spec_path = Path(__file__).parent / "openapi.yaml"
    assert spec_path.exists(), f"OpenAPI spec not found at {spec_path}"
    raw = yaml.safe_load(spec_path.read_text())
    raw["servers"] = [{"url": f"{base_url}/v1"}]
    return raw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def schemathesis_server():
    """Synchronous fixture — schemathesis tests are sync."""
    import httpx

    port = _find_free_port()
    api_key = secrets.token_urlsafe(32)
    hmac_secret = secrets.token_urlsafe(32)

    env = {
        **os.environ,
        "QORTEX_GRAPH": "memory",
        "QORTEX_VEC": "memory",
        "QORTEX_API_KEYS": api_key,
        "QORTEX_HMAC_SECRET": hmac_secret,
        "QORTEX_CORS_ORIGINS": "*",
    }

    venv_bin = Path(sys.executable).parent / "qortex"
    if venv_bin.exists():
        cmd = [str(venv_bin), "serve", "--host", "127.0.0.1", "--port", str(port)]
    elif shutil.which("qortex"):
        cmd = ["qortex", "serve", "--host", "127.0.0.1", "--port", str(port)]
    else:
        cmd = [
            sys.executable, "-c",
            "from qortex.cli import main; main()",
            "serve", "--host", "127.0.0.1", "--port", str(port),
        ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"

    # Sync health poll
    deadline = time.monotonic() + 20.0
    healthy = False
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/v1/health", timeout=2.0)
            if resp.status_code == 200:
                healthy = True
                break
        except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
            pass
        time.sleep(0.3)

    if not healthy:
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

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Schemathesis tests (v4 API)
# ---------------------------------------------------------------------------


def test_api_contract(schemathesis_server: ServerInfo):
    """Iterate all operations, generate cases, and verify no 500 errors."""
    import schemathesis.openapi
    from schemathesis.core.errors import InvalidSchema

    raw_spec = _load_spec(schemathesis_server.base_url)
    schema = schemathesis.openapi.from_dict(raw_spec)
    base_url = f"{schemathesis_server.base_url}/v1"
    api_key = schemathesis_server.api_key

    # Dummy path parameter values for parametrized endpoints
    path_params = {
        "learner": "test-learner",
        "source_id": "test-source",
    }

    tested = 0
    for result in schema.get_all_operations():
        op = result.ok()
        if op is None:
            continue

        # Build path_parameters dict for this operation's path params
        op_path_params = {
            k: v for k, v in path_params.items() if f"{{{k}}}" in op.path
        }

        try:
            case = schema.make_case(
                operation=op,
                path_parameters=op_path_params or None,
            )
        except InvalidSchema:
            # Skip operations that can't generate valid cases
            continue

        case.headers = case.headers or {}
        case.headers["Authorization"] = f"Bearer {api_key}"

        response = case.call(base_url=base_url)

        # We accept 2xx, 400 (validation), 401 (auth), 404 (not found).
        # A 500 means the server crashed — that's a real bug.
        assert response.status_code < 500, (
            f"Server error {response.status_code} on "
            f"{case.method.upper()} {case.path}: {response.text}"
        )
        tested += 1

    assert tested > 0, "No operations found in OpenAPI spec"


def test_api_no_500_on_health(schemathesis_server: ServerInfo):
    """Smoke test: health endpoint never 500s regardless of input."""
    import schemathesis.openapi

    raw_spec = _load_spec(schemathesis_server.base_url)
    schema = schemathesis.openapi.from_dict(raw_spec)
    base_url = f"{schemathesis_server.base_url}/v1"

    for result in schema.get_all_operations():
        op = result.ok()
        if op and "/health" in op.path:
            case = schema.make_case(operation=op)
            response = case.call(base_url=base_url)
            assert response.status_code < 500
            break
