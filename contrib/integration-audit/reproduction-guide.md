# Reproduction Guide

> Personal reference for verifying every claim in stat-sheet.md.
> Each section maps directly to a stat sheet claim. Run the command, check the output, tick the box.

---

## Setup (one-time)

```bash
# 1. qortex-track-c — all Python benchmarks + adapters live here
cd /Users/peleke/Documents/Projects/qortex-track-c
uv sync

# 2. mastra-qortex — TypeScript MCP integration
cd /Users/peleke/Documents/Projects/mastra-qortex
npm install

# 3. crewAI — their test suite (large dep tree, ~30s)
cd /Users/peleke/Documents/Projects/crewAI
UV_HTTP_TIMEOUT=120 uv sync
```

**Env used for original numbers:**
- Machine: MacBook Pro (Apple Silicon)
- Python: 3.14.0
- Node: 18+
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384d, downloaded automatically on first run)

---

## Claim 1: "+22% Precision, +26% Recall, +14% nDCG"

This is the headline. Three independent benchmarks produce it — one per adapter (CrewAI, Agno, AutoGen). They all hit the same engine, so the numbers should be identical.

### 1a. CrewAI adapter path

```bash
cd /Users/peleke/Documents/Projects/qortex-track-c
uv run pytest tests/bench_crewai_vs_vanilla.py -v -s
```

**What to look for in the output:**

```
AVERAGE   0.55   0.45   0.81   0.65   0.716   0.628
  Precision delta: +22%
  Recall delta:    +26%
  nDCG delta:      +14%
```

The columns are: Q-P@5, V-P@5, Q-R@5, V-R@5, Q-nDCG, V-nDCG. Q = qortex, V = vanilla.

**Verify:** The delta percentages match stat-sheet.md headline numbers.

### 1b. Agno adapter path

```bash
uv run pytest tests/eval_agno_vs_qortex.py -v -s
```

**What to look for:**

```
AVERAGE   0.55    0.45   0.81   0.65       4       4
```

Same P@5 and R@5 as CrewAI. (Agno eval doesn't compute nDCG — it was written earlier.)

**Verify:** P@5 0.55 vs 0.45 = +22%. R@5 0.81 vs 0.65 = +25%.

### 1c. AutoGen adapter path (async)

```bash
uv run pytest tests/bench_autogen_vs_vanilla.py -v -s
```

**What to look for:**

```
AVERAGE   0.55   0.45   0.81   0.65   0.716   0.628
  Precision delta: +22%
  Recall delta:    +26%
  nDCG delta:      +14%
```

Identical to CrewAI. This exercises the async `QortexMemory.query()` path.

**Verify:** Same numbers as 1a. This confirms the adapter is a thin passthrough.

---

## Claim 2: "Zero overhead / -0.5% batch overhead"

```bash
uv run pytest tests/bench_perf.py -v -s
```

**What to look for:**

```
Vanilla (embed + cosine)                    40.34ms    49.86ms     5.04ms
Qortex (embed + vec + graph + rules)        40.15ms    50.48ms     5.02ms
Overhead                                     -0.5%
```

And the component breakdown:

```
Embedding only                               3.97ms     5.77ms
Graph explore (depth=2)                      0.02ms     0.03ms
Feedback recording                          <0.01ms     0.01ms
```

**Verify:**
- [ ] Batch overhead is near zero (should be -2% to +2%)
- [ ] Graph explore is sub-millisecond (should be <0.1ms)
- [ ] Embedding dominates total cost (should be >95% of total)

**Why batch is negative:** numpy batch cosine is faster than per-query cosine. The graph traversal cost (0.02ms) is less than the batch optimization saves.

---

## Claim 3: "46/49 CrewAI tests pass"

```bash
cd /Users/peleke/Documents/Projects/crewAI
uv run pytest lib/crewai/tests/knowledge/ -v --timeout=60
```

**What to look for:**

```
46 passed, 3 failed
```

**Verify:**
- [ ] 46+ passed
- [ ] The 3 failures reference `pandas` or `docling` (missing optional deps, not our code)
- [ ] No failures in storage/search/async tests

---

## Claim 4: "26/26 AutoGen adapter tests pass"

```bash
cd /Users/peleke/Documents/Projects/qortex-track-c
uv run pytest tests/test_autogen_adapter.py -v
```

**What to look for:**

```
26 passed
```

**Verify:**
- [ ] All 26 pass (18 unit + 8 integration)
- [ ] No skips, no xfails

---

## Claim 5: "12/12 Agno tests pass"

The agno eval file doubles as the test suite:

```bash
uv run pytest tests/eval_agno_vs_qortex.py -v
```

**What to look for:** `12 passed`

---

## Claim 6: "Mastra — 31/31 tests, 29 MCP calls in 3.94s"

### Unit tests (no server needed)

```bash
cd /Users/peleke/Documents/Projects/mastra-qortex
npx vitest run tests/vector.test.ts
```

**What to look for:** `20 passed`

### E2E tests (spawns real MCP server)

```bash
npx vitest run tests/e2e.test.ts
```

**What to look for:**
- `11 passed`
- Total time ~3-5s
- Look for `29 MCP tool calls` in the log output

**Verify:**
- [ ] 20 + 11 = 31 total
- [ ] E2E time is under 5s (MCP transport is not the bottleneck)

**Requires:** `uvx qortex` to be available (the MCP server). If this fails, check `uvx qortex mcp-serve` runs standalone.

---

## Claim 7: "Cross-cutting queries: +50% precision, +49% recall"

This comes from the per-query breakdown in claims 1a/1c. In the CrewAI or AutoGen benchmark output, look at the individual query rows:

```
Enterprise SSO for corporate apps     0.60   0.40   1.00   0.67   0.712   0.416
M2M microservices auth                0.60   0.40   1.00   0.67   0.906   0.704
```

**Verify:**
- [ ] SSO query: P@5 0.60 vs 0.40 = +50%
- [ ] SSO query: R@5 1.00 vs 0.67 = +49%
- [ ] SSO query: nDCG 0.712 vs 0.416 = +71%
- [ ] These are the queries where graph dominates (typed edge traversal finds SAML, OpenID Connect)

The first two queries (OAuth2 flow, token formats) should show identical scores — graph doesn't help on focused single-concept queries.

---

## Claim 8: "8,700+ lines of integration code"

Not a benchmark — just a line count. If you want to spot-check:

```bash
# Adapter implementations
wc -l /Users/peleke/Documents/Projects/qortex-track-c/src/qortex/adapters/*.py

# Specific repos (rough)
find /Users/peleke/Documents/Projects/langchain-qortex/langchain_qortex -name "*.py" | xargs wc -l
find /Users/peleke/Documents/Projects/mastra-qortex/src -name "*.ts" | xargs wc -l
```

Not critical to verify exactly. The point is "this isn't a toy integration."

---

## Claim 9: "LangChain — 47 tests pass" / "LangChain.js — ~40 tests pass"

```bash
# Python
cd /Users/peleke/Documents/Projects/langchain-qortex
uv run pytest -v

# TypeScript
cd /Users/peleke/Documents/Projects/langchain-qortex-js
npx vitest run
```

These are our test suites (not LangChain's own). They validate the VectorStore interface contract.

---

## Run Everything (one shot)

```bash
# All Python benchmarks + tests from qortex-track-c
cd /Users/peleke/Documents/Projects/qortex-track-c
uv run pytest tests/eval_agno_vs_qortex.py tests/bench_perf.py tests/bench_crewai_vs_vanilla.py tests/bench_autogen_vs_vanilla.py tests/test_autogen_adapter.py -v -s

# Mastra (TS, MCP)
cd /Users/peleke/Documents/Projects/mastra-qortex
npx vitest run

# CrewAI's own suite
cd /Users/peleke/Documents/Projects/crewAI
UV_HTTP_TIMEOUT=120 uv run pytest lib/crewai/tests/knowledge/ -v --timeout=60

# LangChain (Python)
cd /Users/peleke/Documents/Projects/langchain-qortex
uv run pytest -v

# LangChain (TypeScript)
cd /Users/peleke/Documents/Projects/langchain-qortex-js
npx vitest run
```

---

## Corpus Reference

All quality benchmarks use the same 20-concept authentication domain:

**Core (10):** OAuth2, JWT, OpenID Connect, PKCE, Refresh Token, SAML, mTLS, API Key, Session Cookie, CORS

**Distractors (10):** OAuth1, HTTP Basic Auth, Kerberos, LDAP, RADIUS, X.509 Certificate, Digest Auth, SCRAM, WebAuthn, TOTP

**Edges (8):** Typed relationships between core concepts (`refines`, `supports`, `uses`, `part_of`, `similar_to`, `alternative_to`). Distractors have NO edges — this is the structural signal the graph exploits.

**Rules (5):** Security and architecture constraints (e.g., "Always use PKCE for public clients").

**Queries (4):** Cross-cutting auth scenarios with hand-labeled ground truth. Two are focused (graph ties vanilla), two are cross-cutting (graph dominates).

---

## If Something Fails

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: qortex` | Run `uv sync` in qortex-track-c |
| `ModuleNotFoundError: sentence_transformers` | Same — `uv sync` pulls it |
| Embedding download on first run (~90MB) | Normal, one-time. Subsequent runs use cache. |
| CrewAI tests: `ImportError: pandas` | Expected. Those 3 failures are missing optional deps. |
| Mastra E2E: `spawn ENOENT` or `uvx not found` | Install uv globally: `brew install uv` |
| Mastra E2E: timeout | MCP server spawn can be slow first time. Retry once. |
| Numbers differ slightly from stat sheet | Embedding non-determinism. P@5/R@5 should be exact (rank-based). Latency will vary by machine. |
