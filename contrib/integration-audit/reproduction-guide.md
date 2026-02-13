# Reproduction Guide: qortex Integration Benchmarks

> Everything needed to reproduce the numbers in stat-sheet.md.

---

## Prerequisites

```bash
# Python 3.13+ with uv
brew install uv

# Node.js 18+ with npm
brew install node

# uvx (for MCP server)
pip install uv  # or: brew install uv
```

---

## 1. qortex-track-c Setup

```bash
cd /Users/peleke/Documents/Projects/qortex-track-c

# Install deps (creates .venv automatically)
uv sync
```

### 1a. Agno Eval (graph-enhanced vs vanilla)

```bash
uv run pytest tests/eval_agno_vs_qortex.py -v -s
```

**What it does:** 20-concept auth corpus, 4 queries with ground truth. Compares QortexKnowledge (graph+vec) vs flat cosine similarity. Reports P@5, R@5, distractor counts per query.

**Expected output includes:**
```
AVERAGE   0.55    0.45   0.81   0.65       4       4
```

### 1b. Performance Overhead (batch latency)

```bash
uv run pytest tests/bench_perf.py -v -s
```

**What it does:** 8 queries per method, 20 iterations. Measures median/P95 latency for vanilla search vs qortex retrieve, plus embedding, explore, and feedback latency individually.

**Expected output includes:**
```
Vanilla (embed + cosine)                    40.34ms    49.86ms     5.04ms
Qortex (embed + vec + graph + rules)        40.15ms    50.48ms     5.02ms
Overhead                                     -0.5%
```

### 1c. CrewAI Adapter Benchmark (quality + nDCG + latency)

```bash
uv run pytest tests/bench_crewai_vs_vanilla.py -v -s
```

**What it does:** Same corpus/queries as agno eval, but exercises QortexKnowledgeStorage.search() (the CrewAI adapter path). Reports P@5, R@5, nDCG@5, distractor counts, and per-query latency.

**Expected output includes:**
```
AVERAGE   0.55   0.45   0.81   0.65   0.716   0.628       4       4   28.3    5.0
  Precision delta: +22%
  Recall delta:    +26%
  nDCG delta:      +14%
```

### 1d. AutoGen Adapter Benchmark (quality + nDCG + latency, async)

```bash
uv run pytest tests/bench_autogen_vs_vanilla.py -v -s
```

**What it does:** Same corpus/queries, but exercises QortexMemory.query() (the AutoGen adapter path, all async). Reports P@5, R@5, nDCG@5, distractor counts, and per-query latency.

**Expected output includes:**
```
AVERAGE   0.55   0.45   0.81   0.65   0.716   0.628       4       4   25.8    5.3
  Precision delta: +22%
  Recall delta:    +26%
  nDCG delta:      +14%
```

### 1e. AutoGen Adapter Unit + Integration Tests

```bash
uv run pytest tests/test_autogen_adapter.py -v
```

**What it does:** 18 unit tests (mocked client) + 8 integration tests (real client + InMemoryBackend). Validates all 5 Memory ABC methods, config serialization, feedback loop, and result shapes.

**Expected:** 26/26 pass.

---

## 2. CrewAI's Own Test Suite

```bash
cd /Users/peleke/Documents/Projects/crewAI

# Install (takes ~30s, large dependency tree)
UV_HTTP_TIMEOUT=120 uv sync

# Run knowledge tests
uv run pytest lib/crewai/tests/knowledge/ -v --timeout=60
```

**What it does:** Runs CrewAI's own 49-test knowledge suite. Tests KnowledgeStorage contract, async ops, SearchResult format, Knowledge pipeline, error handling.

**Expected:** 46 pass, 3 fail (missing optional deps: pandas, docling).

---

## 3. Mastra E2E (MCP over stdio)

### 3a. Unit Tests (mock-based, no server needed)

```bash
cd /Users/peleke/Documents/Projects/mastra-qortex
npm install  # if not already done
npx vitest run tests/vector.test.ts
```

**Expected:** 20/20 pass in ~5ms.

### 3b. E2E Tests (real MCP server)

```bash
# Requires uvx qortex to be available
npx vitest run tests/e2e.test.ts
```

**What it does:** Spawns real qortex MCP server via `uvx qortex mcp-serve`, connects QortexVector over stdio, runs all 9 MastraVector methods + dimension validation + full lifecycle test. 29 MCP tool calls total.

**Expected:** 11/11 pass in ~4s.

---

## 4. All Benchmarks (one-shot)

```bash
# From qortex-track-c, run everything:
cd /Users/peleke/Documents/Projects/qortex-track-c
uv run pytest tests/eval_agno_vs_qortex.py tests/bench_perf.py tests/bench_crewai_vs_vanilla.py tests/bench_autogen_vs_vanilla.py tests/test_autogen_adapter.py -v -s

# From mastra-qortex:
cd /Users/peleke/Documents/Projects/mastra-qortex
npx vitest run

# From crewAI:
cd /Users/peleke/Documents/Projects/crewAI
UV_HTTP_TIMEOUT=120 uv run pytest lib/crewai/tests/knowledge/ -v --timeout=60
```

---

## Environment Used for Stat Sheet Numbers

```
Machine:    MacBook Pro (Apple Silicon)
Python:     3.14.0
Node:       18+
uv:         latest
Embedding:  sentence-transformers/all-MiniLM-L6-v2 (384d)
Date:       2026-02-13
```

---

## Corpus Description

All benchmarks use the same 20-concept authentication domain:

**Core concepts (10):** OAuth2, JWT, OpenID Connect, PKCE, Refresh Token, SAML, mTLS, API Key, Session Cookie, CORS

**Distractors (10):** OAuth1, HTTP Basic Auth, Kerberos, LDAP, RADIUS, X.509 Certificate, Digest Authentication, SCRAM, WebAuthn, TOTP

**Graph edges (8):** Typed relationships between core concepts (refines, supports, uses, part_of, similar_to, alternative_to). Distractors have NO edges -- this is the structural signal the graph exploits.

**Rules (5):** Security and architecture rules (e.g., "Always use PKCE for public clients").

**Queries (4):** Cross-cutting auth scenarios with ground-truth expected concepts and known distractors.
