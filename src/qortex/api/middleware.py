"""Middleware stack for the qortex REST API.

Layers (applied bottom-up by Starlette — listed top-down here):
1. CORS — permissive for dev, configurable for prod
2. Request tracing — OTel spans per-request (graceful no-op without otel)
3. Structured request logging — method, path, status, latency
4. Auth — API key or HMAC signature verification
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from qortex.observe.logging import get_logger

logger = get_logger(__name__)

# Paths that never require auth
PUBLIC_PATHS = frozenset({"/v1/health"})


# ---------------------------------------------------------------------------
# Auth configuration
# ---------------------------------------------------------------------------


class AuthConfig:
    """Authentication configuration loaded from env vars.

    Supports two modes (can be combined):
    - API key: ``Authorization: Bearer <key>``
    - HMAC: ``X-Qortex-Signature: <hex-digest>`` with ``X-Qortex-Timestamp``

    Keys/secrets are stored as SHA-256 hashes to avoid holding plaintext in
    memory. Set ``QORTEX_API_KEYS`` (comma-separated) and/or
    ``QORTEX_HMAC_SECRET`` in the environment.

    When neither is set, auth is disabled (open access).
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._key_hashes: set[str] = set()
        self._hmac_secret: bytes | None = None
        self._hmac_max_age: int = 300  # seconds

        raw_keys = os.environ.get("QORTEX_API_KEYS", "").strip()
        if raw_keys:
            for k in raw_keys.split(","):
                k = k.strip()
                if k:
                    self._key_hashes.add(self._hash_key(k))
            self.enabled = True

        raw_hmac = os.environ.get("QORTEX_HMAC_SECRET", "").strip()
        if raw_hmac:
            self._hmac_secret = raw_hmac.encode()
            self.enabled = True

        max_age = os.environ.get("QORTEX_HMAC_MAX_AGE", "").strip()
        if max_age:
            self._hmac_max_age = int(max_age)

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def verify_api_key(self, key: str) -> bool:
        """Constant-time comparison of API key against stored hashes."""
        candidate = self._hash_key(key)
        return any(hmac.compare_digest(candidate, h) for h in self._key_hashes)

    def verify_hmac(self, body: bytes, signature: str, timestamp: str) -> bool:
        """Verify HMAC-SHA256 signature over timestamp + body.

        Signature = HMAC-SHA256(secret, timestamp + "." + body)
        """
        if self._hmac_secret is None:
            return False

        # Replay protection
        try:
            ts = int(timestamp)
        except (ValueError, TypeError):
            return False
        if abs(time.time() - ts) > self._hmac_max_age:
            return False

        message = f"{timestamp}.".encode() + body
        expected = hmac.new(self._hmac_secret, message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# ASGI body buffering helpers
# ---------------------------------------------------------------------------


async def _read_body(receive: Receive) -> bytes:
    """Read the full ASGI request body from the receive channel."""
    chunks: list[bytes] = []
    while True:
        message = await receive()
        body = message.get("body", b"")
        if body:
            chunks.append(body)
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _make_receive(body: bytes) -> Receive:
    """Create a replay receive callable that yields the buffered body.

    After HMAC middleware reads the body for verification, downstream
    handlers need to read it again.  This replays the buffered bytes.
    """

    async def receive() -> dict:
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


# ---------------------------------------------------------------------------
# Auth middleware (ASGI)
# ---------------------------------------------------------------------------


class AuthMiddleware:
    """ASGI middleware for API key + HMAC authentication.

    Skips PUBLIC_PATHS. When auth is disabled (no keys/secrets configured),
    passes all requests through.
    """

    def __init__(self, app: ASGIApp, config: AuthConfig | None = None) -> None:
        self.app = app
        self.config = config or AuthConfig()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.config.enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path.rstrip("/")

        if path in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return

        # Try Bearer token first
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if self.config.verify_api_key(token):
                await self.app(scope, receive, send)
                return

        # Try HMAC signature
        signature = request.headers.get("x-qortex-signature", "")
        timestamp = request.headers.get("x-qortex-timestamp", "")
        if signature and timestamp:
            # Buffer the body so downstream handlers can still read it
            body = await _read_body(receive)
            if self.config.verify_hmac(body, signature, timestamp):
                await self.app(scope, _make_receive(body), send)
                return

        logger.warning("auth.rejected", path=path, method=request.method)
        response = JSONResponse(
            {"error": "Unauthorized. Provide a valid API key or HMAC signature."},
            status_code=401,
        )
        await response(scope, receive, send)


# ---------------------------------------------------------------------------
# Request tracing middleware (OTel)
# ---------------------------------------------------------------------------


class TracingMiddleware:
    """ASGI middleware that wraps each request in an OTel span.

    Graceful no-op when opentelemetry is not installed.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._tracer: Any = None
        self._trace_mod: Any = None
        try:
            from opentelemetry import trace

            self._tracer = trace.get_tracer("qortex.api")
            self._trace_mod = trace
        except ImportError:
            pass

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._tracer is None:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        method = request.method
        path = request.url.path

        with self._tracer.start_as_current_span(
            f"{method} {path}",
            attributes={
                "http.method": method,
                "http.url": str(request.url),
                "http.route": path,
            },
        ) as span:
            status_code = 500

            async def send_wrapper(message: Any) -> None:
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    span.set_attribute("http.status_code", status_code)
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as exc:
                span.set_status(self._trace_mod.StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise
            finally:
                if status_code >= 400:
                    span.set_status(
                        self._trace_mod.StatusCode.ERROR,
                        f"HTTP {status_code}",
                    )


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware:
    """ASGI middleware for structured request/response logging.

    Logs method, path, status code, and latency for every request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Any) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            log_fn = logger.warning if status_code >= 400 else logger.info
            log_fn(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                latency_ms=elapsed,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_middleware(app: Any, auth_config: AuthConfig | None = None) -> None:
    """Add the full middleware stack to a Starlette app.

    Order matters — Starlette applies middleware bottom-up, so the first
    added is the outermost. We want:
      Request → CORS → Tracing → Logging → Auth → Route handler
    """
    # Outermost: CORS (must handle preflight before auth)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Tracing wraps everything including auth failures
    app.add_middleware(TracingMiddleware)
    # Logging captures status + latency
    app.add_middleware(RequestLoggingMiddleware)
    # Auth is innermost — only reached after CORS/tracing/logging
    app.add_middleware(AuthMiddleware, config=auth_config)
