# Build Journal: Network Service Layer — Phase 1 Complete

**Date:** 2026-02-22
**Duration:** ~6 hours (across 2 sessions)
**Status:** Complete

---

## The Goal

Make qortex deployable as a network service. Currently it runs in-process (embedded Python or MCP stdio subprocess). This blocks production deployment: external services can't query the knowledge graph over the network, multi-tenant isn't possible, and framework adapters pay threadpool overhead for what should be native async I/O.

Phase 1 delivers: QortexService extraction, REST API (19 endpoints), async HttpQortexClient, auth middleware (API key + HMAC-SHA256), observability middleware (OTel tracing + structured logging), and the `qortex serve` CLI command.

---

## What We Built

### Architecture

```
                         ┌─────────────────────────────┐
                         │     Framework Adapters       │
                         │  (Agno, AutoGen, LangChain,  │
                         │   Mastra, CrewAI)            │
                         └────────────┬────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
              LocalQortexClient  HttpQortexClient  MCP Server
                    │            (async httpx)      (FastMCP)
                    │                 │                  │
                    │          ┌──────┴──────┐           │
                    │          │  REST API   │           │
                    │          │ (Starlette) │           │
                    │          │  19 routes  │           │
                    │          │  CORS+Auth  │           │
                    │          │  +OTel+Log  │           │
                    │          └──────┬──────┘           │
                    │                 │                  │
                    └────────────────┬┘                  │
                                     │                  │
                              QortexService ◄───────────┘
                                     │
                    ┌────────────────┬┘──────────────────┐
                    │                │                    │
              GraphBackend    VectorIndex          LearningStore
              (Memory/MG)    (Numpy/SQLite)        (JSON/SQLite)
```

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| QortexService | Working | Extracted from MCP server globals |
| REST API (routes.py) | Working | 19 endpoints, thin handlers |
| Auth middleware | Working | API key + HMAC-SHA256, replay protection |
| Tracing middleware | Working | OTel spans, graceful no-op |
| Logging middleware | Working | Structured method/path/status/latency |
| HttpQortexClient | Working | Async httpx, HMAC signing |
| `qortex serve` CLI | Working | uvicorn, host/port/workers/reload |
| CORS config | Working | QORTEX_CORS_ORIGINS env var |

---

## The Journey

### Session 1: Core extraction + REST API + tests

**What we built:**
- Extracted QortexService from mcp/server.py module globals
- Created Starlette REST API with 19 route handlers
- Built HttpQortexClient (initially sync)
- Wrote test_service.py, test_api.py, test_http_client.py

**Key decisions:**
- Starlette over FastAPI (already indirect dep, no pydantic v2 required)
- Methods return dicts (JSON-serializable), clients deserialize to protocol types
- asyncio.to_thread() for sync service methods in async route handlers

### Session 2: Auth + observability + gauntlet + async migration

**What we tried:**
Added auth middleware with API key + HMAC, then hit the ASGI body double-read bug.

**What happened:**
HMAC middleware reads `await request.body()` to verify the signature. This consumes the ASGI receive channel. Route handlers then hang forever waiting for a body that's already been read.

**The fix:**
Buffer the body with `_read_body(receive)`, verify HMAC, then pass a replay callable `_make_receive(body)` to downstream handlers.

**Lesson:**
ASGI receive channels are single-read. Any middleware that inspects the body must buffer and replay it for downstream consumers.

---

### Gauntlet review findings

**1 major (fixed):** CORS allow_origins=['*'] was hardcoded. Now configurable via `QORTEX_CORS_ORIGINS` env var.

**Minors (accepted):**
- HMAC secret stored as raw bytes (inherent — can't hash what you need to sign)
- AuthConfig.__init__() reads env vars (tests use __new__() bypass — acceptable)
- source_connect accepts credentials over HTTP (TLS is reverse proxy concern)
- No request body size limit (Phase 2: add max body middleware)
- OTel span names use resolved paths not route templates (Phase 2: normalize)

### Async HttpQortexClient migration

**What we tried:** Initially built HttpQortexClient with sync httpx.Client.

**What happened:** User correctly identified this as braindead — every consumer is async (AutoGen, route handlers, MCP tools), so they'd all wrap sync calls in asyncio.to_thread(), paying thread overhead for what should be native async I/O.

**The fix:** Converted to httpx.AsyncClient. All methods async. Tests use httpx.ASGITransport instead of Starlette TestClient.

**Lesson:** Match the async characteristics of your client to its primary consumers. Don't make consumers pay for sync→async bridging when the underlying I/O (HTTP) is inherently async.

---

## Test Results

### Phase 1 test suite (80 tests)

**Command:**
```bash
uv run pytest tests/test_service.py tests/test_api.py tests/test_http_client.py tests/test_auth.py -x -q
```

**Result:** 80 passed in 0.35s

### Full regression suite (1789 tests)

**Command:**
```bash
uv run pytest tests/ -q --ignore=tests/integration --ignore=tests/causal \
  --ignore=tests/test_langchain_e2e_dogfood.py --ignore=tests/test_langchain_vectorstore.py
```

**Result:** 1789 passed, 14 failed (pre-existing: OTel SDK, Memgraph, networkx), 131 skipped. Zero regressions.

---

## What's Left

- [ ] Phase 2: Dockerfile + docker-compose (qortex-server service)
- [ ] Phase 2: OpenClaw sandbox systemd integration
- [ ] Phase 2: Request body size limit middleware
- [ ] Phase 2: OTel span name normalization (route templates)
- [ ] Phase 3: pgvector + PostgreSQL backend
- [ ] Phase 3: Unified QortexConfig
- [ ] Merge fix/ci-pr146 into feat/async-learning-extraction (PR #146)

---

## Files Changed

```
src/qortex/
├── service.py              # NEW — QortexService (shared by MCP + REST)
├── http_client.py          # NEW — async HttpQortexClient
├── api/
│   ├── __init__.py         # NEW
│   ├── app.py              # NEW — Starlette ASGI factory
│   ├── routes.py           # NEW — 19 route handlers
│   └── middleware.py       # NEW — Auth + OTel + logging + CORS
├── cli/__init__.py         # MODIFIED — added `qortex serve`
tests/
├── test_service.py         # NEW — 27 tests
├── test_api.py             # NEW — 23 tests
├── test_http_client.py     # NEW — 11 tests (async)
├── test_auth.py            # NEW — 19 tests
pyproject.toml              # MODIFIED — server + http-client extras
```

---

## Improvements

### Architectural
- ASGI middleware that reads request bodies must buffer and replay — bake this pattern into a shared helper
- Config classes should provide explicit-args factory alongside env-var-reading constructor
- Match async characteristics of clients to consumers — don't pay for bridging

### Workflow
- COMMIT EARLY AND OFTEN. Lost files during a rebase because they were untracked. Never again.
- Separate feature branches from CI fix branches. Don't mix concerns in the same worktree.
- Use worktrees (`qortex-track-c`) to isolate parallel work streams

### Domain Knowledge
- Starlette applies middleware bottom-up (first added = outermost)
- httpx.ASGITransport is the correct async alternative to Starlette TestClient
- httpx event hooks must be async for AsyncClient (sync for Client)

---

*Next entry: Phase 2 — Docker deployment + OpenClaw integration*
