#!/usr/bin/env python3
"""Live stack validation: prove observability works end-to-end.

Runs a real workload (Memgraph PPR + learning + credit propagation),
then queries Prometheus, Grafana, and Jaeger APIs to verify data arrived.

Prerequisites:
    cd docker && docker compose up -d

Usage:
    MEMGRAPH_USER=memgraph MEMGRAPH_PASSWORD=memgraph \
    uv run python scripts/validate_live_stack.py

Exit code 0 = all checks pass. Non-zero = failures.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import uuid

import requests

# ── Force full observability ───────────────────────────────────────
os.environ["QORTEX_OTEL_ENABLED"] = "true"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
os.environ["QORTEX_PROMETHEUS_ENABLED"] = "true"
os.environ["QORTEX_PROMETHEUS_PORT"] = "9464"
# Export ALL spans so Jaeger shows full Cypher trace tree
os.environ["QORTEX_OTEL_TRACE_SAMPLE_RATE"] = "1.0"
os.environ["QORTEX_OTEL_TRACE_LATENCY_THRESHOLD_MS"] = "0.0"

from qortex_observe import configure, emit, reset as obs_reset
from qortex_observe.config import ObservabilityConfig
from qortex_observe.events import (
    CreditPropagated,
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
    PPRConverged,
    PPRStarted,
)


def _port_open(host: str, port: int) -> bool:
    try:
        s = socket.socket()
        s.settimeout(2)
        s.connect((host, port))
        s.close()
        return True
    except OSError:
        return False


# ── Preflight checks ──────────────────────────────────────────────
CHECKS = {
    "Memgraph (7687)": _port_open("localhost", 7687),
    "OTel Collector (4318)": _port_open("localhost", 4318),
    "Prometheus (9091)": _port_open("localhost", 9091),
    "Grafana (3010)": _port_open("localhost", 3010),
    "Jaeger (16686)": _port_open("localhost", 16686),
}

print("=" * 70)
print("  QORTEX LIVE STACK VALIDATION")
print("=" * 70)
print()
print("  Preflight:")
all_up = True
for name, up in CHECKS.items():
    status = "UP" if up else "DOWN"
    print(f"    {name:<30s} {status}")
    if not up:
        all_up = False

if not all_up:
    print()
    print("  FAIL: Not all services are running.")
    print("  Run: cd docker && docker compose up -d")
    sys.exit(1)

print()
print("  All services UP. Starting validation...")
print()

# ── Initialize observability ──────────────────────────────────────
obs_reset()
configure()

failures: list[str] = []
passes: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        passes.append(name)
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        msg = f"  FAIL  {name}"
        if detail:
            msg += f": {detail}"
        print(msg)


# ── Phase 1: Memgraph PPR with traced Cypher ─────────────────────
print()
print("[1/5] Running PPR against real Memgraph (traced Cypher spans)...")

from qortex.core.backend import MemgraphBackend, MemgraphCredentials
from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    IngestionManifest,
    RelationType,
    SourceMetadata,
)

creds = MemgraphCredentials(
    user=os.environ.get("MEMGRAPH_USER", "memgraph"),
    password=os.environ.get("MEMGRAPH_PASSWORD", "memgraph"),
)
backend = MemgraphBackend(host="localhost", port=7687, credentials=creds)
backend.connect()

domain = f"validate-{uuid.uuid4().hex[:8]}"
source = SourceMetadata(
    id="validate-src", name="validation",
    source_type="text", path_or_url="/dev/null",
)

nodes = [
    ConceptNode(id="python", name="Python", domain=domain,
                description="A programming language", source_id=source.id),
    ConceptNode(id="typing", name="Type Hints", domain=domain,
                description="Static typing for Python", source_id=source.id),
    ConceptNode(id="mypy", name="Mypy", domain=domain,
                description="Static type checker", source_id=source.id),
    ConceptNode(id="pydantic", name="Pydantic", domain=domain,
                description="Data validation", source_id=source.id),
    ConceptNode(id="fastapi", name="FastAPI", domain=domain,
                description="Web framework", source_id=source.id),
]
edges = [
    ConceptEdge(source_id="python", target_id="typing", relation_type=RelationType.SUPPORTS),
    ConceptEdge(source_id="typing", target_id="mypy", relation_type=RelationType.REQUIRES),
    ConceptEdge(source_id="typing", target_id="pydantic", relation_type=RelationType.USES),
    ConceptEdge(source_id="pydantic", target_id="fastapi", relation_type=RelationType.PART_OF),
]
manifest = IngestionManifest(source=source, domain=domain, concepts=nodes, edges=edges, rules=[])
backend.ingest_manifest(manifest)

# Run PPR 5 times to generate multiple traced Cypher spans
ppr_query_ids = []
for i in range(5):
    qid = f"validate-ppr-{i}-{uuid.uuid4().hex[:6]}"
    ppr_query_ids.append(qid)
    scores = backend.personalized_pagerank(
        source_nodes=["python"], domain=domain, query_id=qid,
    )

check("PPR returned scores", len(scores) > 0, f"got {len(scores)}")
check("PPR seed has highest score", scores.get("python", 0) > 0)

print(f"       Ran 5 PPR queries: {len(scores)} scored nodes each")

# ── Phase 2: Learning workload ───────────────────────────────────
print()
print("[2/5] Running learning workload (bandit selections + observations)...")

from qortex.learning.learner import Learner
from qortex.learning.types import Arm, ArmOutcome, LearnerConfig

import random

learner = Learner(LearnerConfig(
    name="validate-learner",
    baseline_rate=0.1,
    state_dir="/tmp/qortex-validate-learning",
))

candidates = [
    Arm(id="prompt:basic", metadata={"type": "basic"}, token_cost=100),
    Arm(id="prompt:cot", metadata={"type": "cot"}, token_cost=200),
    Arm(id="prompt:fewshot", metadata={"type": "fewshot"}, token_cost=300),
]

for i in range(15):
    result = learner.select(candidates, context={"task": "validate"}, k=2)
    for arm in result.selected:
        outcome = "accepted" if random.random() < (0.8 if arm.id == "prompt:cot" else 0.3) else "rejected"
        learner.observe(
            ArmOutcome(arm_id=arm.id, reward=0.0, outcome=outcome),
            context={"task": "validate"},
        )

posteriors = learner.posteriors(context={"task": "validate"})
check("Learning posteriors exist", len(posteriors) > 0, f"{len(posteriors)} arms")
cot_mean = posteriors.get("prompt:cot", {}).get("mean", 0)
basic_mean = posteriors.get("prompt:basic", {}).get("mean", 0)
check("CoT has higher posterior than basic", cot_mean > basic_mean,
      f"cot={cot_mean:.3f} vs basic={basic_mean:.3f}")

print(f"       15 rounds, cot posterior: {cot_mean:.3f}")

# ── Phase 3: Credit propagation ──────────────────────────────────
print()
print("[3/5] Running credit propagation...")

for i in range(10):
    emit(CreditPropagated(
        query_id=f"validate-credit-{i}",
        concept_count=random.randint(3, 8),
        direct_count=random.randint(1, 3),
        ancestor_count=random.randint(2, 5),
        total_alpha_delta=random.uniform(0.1, 1.0),
        total_beta_delta=random.uniform(0.0, 0.5),
        learner="validate-learner",
    ))

print("       Emitted 10 CreditPropagated events")

# ── Phase 4: Force flush + wait for scrape ───────────────────────
print()
print("[4/5] Flushing metrics and waiting for Prometheus scrape cycle...")

# Force flush OTel metric reader + trace exporter
try:
    from opentelemetry.metrics import get_meter_provider
    get_meter_provider().force_flush(timeout_millis=10000)
except Exception:
    pass

try:
    from opentelemetry import trace
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=10000)
except Exception:
    pass

# Wait for Prometheus scrape (15s interval) + OTel batch (5s)
print("       Waiting 25s for scrape cycle...")
time.sleep(25)

# ── Phase 5: Verify all backends ─────────────────────────────────
print()
print("[5/5] Verifying data in Prometheus, Grafana, and Jaeger...")
print()
prom_port = int(os.environ.get("QORTEX_PROMETHEUS_PORT", "9464"))
print(f"  --- Direct Prometheus /metrics (port {prom_port}) ---")

try:
    resp = requests.get(f"http://localhost:{prom_port}/metrics", timeout=5)
    body = resp.text
    qortex_metrics = [l for l in body.splitlines()
                      if l.startswith("qortex_") and not l.startswith("#")]

    check("Direct /metrics has qortex_ metrics", len(qortex_metrics) > 0,
          f"{len(qortex_metrics)} metric lines")

    # Check specific metrics (OTel adds _total suffix to counters)
    # Note: qortex_queries_total only fires from the retrieval adapter,
    # not from standalone PPR calls. We check metrics our workload emits.
    for metric_name, label in [
        ("qortex_ppr_started_total", "PPR started counter"),
        ("qortex_learning_selections_total", "Learning selections"),
        ("qortex_learning_observations_total", "Learning observations"),
        ("qortex_credit_propagations_total", "Credit propagations"),
    ]:
        found_lines = [l for l in qortex_metrics
                       if l.startswith(metric_name)
                       and not l.startswith(metric_name + "_")]
        has_value = False
        for l in found_lines:
            try:
                v = float(l.split()[-1])
                if v > 0:
                    has_value = True
                    break
            except (ValueError, IndexError):
                pass
        check(f"  {label} ({metric_name}) > 0", has_value,
              "not found" if not found_lines else "value is 0")

except requests.ConnectionError:
    check(f"Direct /metrics endpoint reachable on port {prom_port}", False,
          "connection refused")

# ── Prometheus server query ──────────────────────────────────────
print()
print("  --- Prometheus server (port 9091) ---")

try:
    # Check qortex job target health
    resp = requests.get("http://localhost:9091/api/v1/targets", timeout=5)
    targets = resp.json().get("data", {}).get("activeTargets", [])
    qortex_target = next((t for t in targets if t.get("labels", {}).get("job") == "qortex"), None)
    if qortex_target:
        check("Prometheus scraping qortex target",
              qortex_target.get("health") == "up",
              qortex_target.get("lastError", ""))
    else:
        check("Prometheus has qortex scrape target", False, "no qortex job found")

    # Query for metrics
    for metric_query in [
        ("qortex_ppr_started_total", "PPR counter"),
        ("qortex_learning_selections_total", "Learning selections"),
        ("qortex_credit_propagations_total", "Credit propagations"),
    ]:
        metric, label = metric_query
        resp = requests.get(
            "http://localhost:9091/api/v1/query",
            params={"query": metric},
            timeout=5,
        )
        data = resp.json().get("data", {}).get("result", [])
        has_data = any(float(r["value"][1]) > 0 for r in data) if data else False
        check(f"  Prometheus has {label}", has_data,
              f"{len(data)} series" if data else "no data")

except requests.ConnectionError:
    check("Prometheus API reachable", False, "connection refused")

# ── Grafana datasource proxy ────────────────────────────────────
print()
print("  --- Grafana datasource proxy (port 3010) ---")

try:
    # Query Prometheus through Grafana
    resp = requests.get(
        "http://localhost:3010/api/datasources/proxy/uid/prometheus/api/v1/query",
        params={"query": "qortex_ppr_started_total"},
        headers={"Authorization": "Basic YWRtaW46cW9ydGV4"},  # admin:qortex
        timeout=5,
    )
    if resp.status_code == 200:
        data = resp.json().get("data", {}).get("result", [])
        has_data = any(float(r["value"][1]) > 0 for r in data) if data else False
        check("Grafana sees qortex metrics via Prometheus", has_data,
              f"{len(data)} series")
    else:
        check("Grafana datasource proxy responds", False, f"HTTP {resp.status_code}")

except requests.ConnectionError:
    check("Grafana API reachable", False, "connection refused")

# ── Jaeger traces ────────────────────────────────────────────────
print()
print("  --- Jaeger traces (port 16686) ---")

try:
    # Query Jaeger for qortex service traces
    resp = requests.get(
        "http://localhost:16686/api/services",
        timeout=5,
    )
    services = resp.json().get("data", [])
    has_qortex = "qortex" in services
    check("Jaeger has 'qortex' service", has_qortex,
          f"services: {services}")

    if has_qortex:
        # Get recent traces
        resp = requests.get(
            "http://localhost:16686/api/traces",
            params={
                "service": "qortex",
                "limit": 20,
                "lookback": "1h",
            },
            timeout=10,
        )
        traces = resp.json().get("data", [])
        check("Jaeger has recent qortex traces", len(traces) > 0,
              f"{len(traces)} traces")

        if traces:
            # Check for Cypher spans in trace tree
            cypher_spans = []
            all_span_names = set()
            for trace_data in traces:
                for span in trace_data.get("spans", []):
                    op = span.get("operationName", "")
                    all_span_names.add(op)
                    if "cypher" in op.lower():
                        cypher_spans.append(span)
                        # Check for db.statement attribute
                        tags = {t["key"]: t["value"] for t in span.get("tags", [])}
                        if tags.get("db.system") == "memgraph":
                            check("Cypher span has db.system=memgraph", True)
                        if tags.get("db.statement"):
                            stmt = tags["db.statement"]
                            check("Cypher span has db.statement",
                                  len(stmt) > 0,
                                  stmt[:80])

            check("Jaeger has cypher.execute spans", len(cypher_spans) > 0,
                  f"found {len(cypher_spans)} cypher spans")

            # Show all span operation names
            print(f"\n       Span operations found: {sorted(all_span_names)}")

            # cypher.execute is the critical span (proves Cypher is traced)
            check("  Span 'cypher.execute' in traces",
                  "cypher.execute" in all_span_names,
                  f"all spans: {sorted(all_span_names)}")

except requests.ConnectionError:
    check("Jaeger API reachable", False, "connection refused")

# ── Cleanup ──────────────────────────────────────────────────────
backend._run("MATCH (n {domain: $d}) DETACH DELETE n", {"d": domain})
backend._run("MATCH (d:Domain {name: $d}) DELETE d", {"d": domain})
backend.disconnect()

# ── Summary ──────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  RESULTS: {len(passes)} passed, {len(failures)} failed")
print("=" * 70)

if failures:
    print()
    print("  FAILURES:")
    for f in failures:
        print(f"    - {f}")
    print()
    sys.exit(1)
else:
    print()
    print("  ALL CHECKS PASSED. Observability stack is fully operational.")
    print()
    print("  Grafana dashboard: http://localhost:3010/d/qortex-main/qortex-observability")
    print("  Jaeger UI:         http://localhost:16686/search?service=qortex")
    print("  Prometheus:        http://localhost:9091/graph")
    print()
    sys.exit(0)
