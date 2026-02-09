# Build Journal: OTEL Collector + Push-Based Metrics

**Date:** 2026-02-09
**Duration:** ~30 min
**Status:** Complete

---

## The Goal

Make observability work when qortex runs in a remote sandbox (Lima VM) while Grafana/Prometheus run on the host. The Prometheus subscriber is pull-based (scrapes localhost), which breaks across network boundaries. Solution: add an OTEL Collector as a routing layer so push-based OTEL metrics get converted to Prometheus format, keeping the existing Grafana dashboard working unchanged.

---

## What We Built

### Architecture

```
[Sandbox: OpenClaw + qortex]
    │ OTLP push (gRPC :4317)
    ↓
[Host: OTEL Collector]
    ├── traces → Jaeger (:16686)
    └── metrics → Prometheus exporter (:8889)
                     ↓
              Prometheus scrapes collector
                     ↓
              Grafana (:3010) — same 18-panel dashboard
```

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| OTEL Collector config | Working | OTLP→Jaeger traces, OTLP→Prometheus metrics |
| docker-compose update | Working | Collector service, OTLP ports moved from Jaeger |
| Prometheus scrape config | Working | Added otel-collector:8889 target |
| OTEL subscriber rewrite | Working | 18 aligned metrics (was 5), 15 handlers (was 7) |
| Span leak fix | Working | _active_spans capped at 1000 with FIFO eviction |
| Credential externalization | Working | Passwords to .env (gitignored) |
| Prometheus smoke test | Working | configure→emit path verified |

---

## Test Results

### Observability Tests

**Command:**
```bash
uv run pytest tests/test_observability.py -q
```

**Result:** 58 passed (0.13s)

### Full Suite

**Command:**
```bash
uv run pytest --ignore=tests/integration -q
```

**Result:** 1373 passed, 34 skipped (10.65s)

---

## Gauntlet Review

**Iteration 1:** 0 critical, 3 major, 3 minor

### Majors (fixed)
1. `_active_spans` unbounded growth if QueryCompleted events lost. Capped at 1000 with FIFO eviction.
2. Hardcoded credentials in docker-compose.yml. Externalized to `.env` with `${VAR:-default}`.
3. No integration test for Prometheus wiring. Added smoke test: configure→emit→no crash.

### Minors (logged)
- `host.docker.internal` is Docker Desktop specific
- Marketing-style comments in module docstrings

---

## Files Changed

```
docker/
├── otel-collector/
│   └── otel-collector-config.yaml  # NEW: Collector pipeline config
├── .env                             # NEW: Externalized credentials
├── docker-compose.yml               # Collector service + env var refs
└── prometheus/
    └── prometheus.yml               # Added collector scrape target

src/qortex/observability/subscribers/
└── otel.py                          # Full rewrite: 18 metrics, 15 handlers

tests/
└── test_observability.py            # +1 smoke test (58 total)
```

---

## Improvements

### Architectural
- OTEL metric names must drop `_total` suffix for counters (Prometheus exporter adds it). Map carefully.
- `create_gauge()` in opentelemetry-api >= 1.22 supports sync `.set()`, simpler than observable gauge callbacks.

### Domain Knowledge
- Lima VM host: `host.lima.internal` = `192.168.5.2`. Sandbox can push OTLP to `http://host.lima.internal:4317`.
- `StdioClientTransport` inherits parent env vars. No explicit env passing needed.
- OTEL Collector `resource_to_telemetry_conversion: enabled` turns resource attrs into metric labels.

---

*Next: version bump to 0.3.0, merge to main, publish to PyPI, run E2E demo in sandbox*
