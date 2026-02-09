# Build Journal: Observability Gap Closure

**Date:** 2026-02-09
**Duration:** ~1 hour
**Status:** Complete

---

## The Goal

Wire the 4 missing event emissions, fill 8 Prometheus handler gaps, and add 7 Grafana dashboard panels so the observability layer actually works end-to-end during dogfood. The structural scaffolding (22 files, 25 event types, 4 subscribers) was done; the problem was that several events were defined but never emitted, metrics had `pass` stubs, and the dashboard had blank panels.

---

## What We Built

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| FactorDriftSnapshot emission | Working | Shannon entropy calc at end of update() |
| EnrichmentCompleted/Fallback emission | Working | Timing + fallback path wired |
| ManifestIngested emission | Working | Both InMemoryBackend and MemgraphBackend |
| Prometheus handlers (8 new) | Working | Counters, histograms, fixed FactorUpdated stub |
| Grafana dashboard panels (7 new) | Working | New "Enrichment & Ingestion" row |
| Tests (12 new) | Working | 57 total observability tests |

---

## Test Results

### Full Suite

**Command:**
```bash
uv run pytest --ignore=tests/integration
```

**Result:** 1366 passed, 34 skipped, 0 failures (12.32s)

### Observability Tests

**Command:**
```bash
uv run pytest tests/test_observability.py -v
```

**Result:** 57 passed (0.13s)

---

## Gauntlet Review

**Iteration 1:** 0 critical, 2 major, 2 minor

### Majors (fixed)
1. Prometheus metric handlers had no tests verifying actual increment behavior. Added `TestPrometheusMetrics` class with 5 tests.
2. MemgraphBackend ManifestIngested emission had no test coverage. Added `test_memgraph_manifest_ingested_emitted` with mocked neo4j driver.

### Minors (fixed)
1. Entropy test used `abs(x) < 1e-10` instead of `pytest.approx`. Fixed.
2. Dashboard PromQL metric names could drift from prometheus.py instruments. Logged as ongoing concern.

---

## Files Changed

```
src/qortex/
├── hippocampus/
│   └── factors.py              # FactorDriftSnapshot emission + entropy calc
├── enrichment/
│   └── pipeline.py             # EnrichmentCompleted/Fallback + logging migration
├── core/
│   ├── memory.py               # ManifestIngested in InMemoryBackend
│   └── backend.py              # ManifestIngested in MemgraphBackend
└── observability/
    └── subscribers/
        └── prometheus.py       # 8 new metrics + fixed stub

docker/grafana/dashboards/
└── qortex.json                 # 7 new panels, 1 new row

tests/
└── test_observability.py       # 12 new tests (57 total)
```

---

## Improvements

### Architectural
- Event emission at end of methods (after the work is done) is the right pattern. Keeps timing accurate and avoids emitting on partial failures.

### Workflow
- Reading all target files upfront before editing saved context switches. 7 files modified with zero regressions on first pass.

### Domain Knowledge
- `prometheus_client` Counter labels must be declared at instrument creation time. Can't dynamically add new label names later.
- Shannon entropy of factor distribution is a good single-number health metric. Dropping entropy = factors converging to degenerate distribution.

---

*Next entry: Investor demo loop strategy*
