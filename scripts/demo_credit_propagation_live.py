#!/usr/bin/env python3
"""Live demo: watch credit propagation metrics flow into Grafana.

Run this while watching:
  http://localhost:3010/d/qortex-main/qortex-observability

The "Credit Propagation" row at the bottom will light up.

What this does:
  1. Builds a 3-layer causal DAG (root → mid → leaf concepts)
  2. Simulates feedback outcomes (accepted / rejected / partial)
  3. Runs CreditAssigner to propagate credit through the DAG
  4. Feeds posterior deltas into a Learner via apply_credit_deltas()
  5. Emits CreditPropagated events → Prometheus metrics → Grafana
"""

from __future__ import annotations

import os
import random
import sys
import time

# Force Prometheus metrics on port 9090
os.environ["QORTEX_PROMETHEUS_ENABLED"] = "true"
os.environ["QORTEX_PROMETHEUS_PORT"] = "9090"

from qortex.causal.credit import CreditAssigner
from qortex.causal.dag import CausalDAG
from qortex.causal.types import CausalDirection, CausalEdge
from qortex.learning.learner import Learner
from qortex.learning.types import LearnerConfig
from qortex.observability import configure, emit, reset as obs_reset
from qortex.observability.events import CreditPropagated

# ── Setup ─────────────────────────────────────────────────────────
obs_reset()
configure()

print("=" * 64)
print("  QORTEX CREDIT PROPAGATION — LIVE GRAFANA DEMO")
print("=" * 64)
print()
print("  Prometheus metrics at: http://localhost:9090/metrics")
print("  Grafana dashboard at:  http://localhost:3010/d/qortex-main/qortex-observability")
print("  Scroll to: 'Credit Propagation' row")
print()
print("  Press Ctrl+C to stop.")
print("=" * 64)
print()

# Wait for Prometheus to notice us
print("[1/7] Waiting 5s for Prometheus to discover /metrics endpoint...")
time.sleep(5)

# ── Build Causal DAG ──────────────────────────────────────────────
# 3-layer DAG:
#   math:foundations → math:algebra → math:linear-algebra
#   math:foundations → math:calculus → math:linear-algebra
#   math:foundations → math:calculus → math:diff-eq
#   cs:data-structures → cs:algorithms → cs:optimization
#   cs:data-structures → cs:algorithms → cs:ml
#   cs:algorithms → cs:ml

NODE_IDS = [
    "math:foundations",
    "math:algebra",
    "math:calculus",
    "math:linear-algebra",
    "math:diff-eq",
    "cs:data-structures",
    "cs:algorithms",
    "cs:optimization",
    "cs:ml",
]

EDGES = [
    CausalEdge("math:foundations", "math:algebra", "requires", CausalDirection.FORWARD, 0.95),
    CausalEdge("math:foundations", "math:calculus", "requires", CausalDirection.FORWARD, 0.90),
    CausalEdge("math:algebra", "math:linear-algebra", "requires", CausalDirection.FORWARD, 0.85),
    CausalEdge("math:calculus", "math:linear-algebra", "requires", CausalDirection.FORWARD, 0.80),
    CausalEdge("math:calculus", "math:diff-eq", "requires", CausalDirection.FORWARD, 0.75),
    CausalEdge("cs:data-structures", "cs:algorithms", "requires", CausalDirection.FORWARD, 0.90),
    CausalEdge("cs:algorithms", "cs:optimization", "requires", CausalDirection.FORWARD, 0.85),
    CausalEdge("cs:algorithms", "cs:ml", "requires", CausalDirection.FORWARD, 0.80),
]

dag = CausalDAG.from_edges(EDGES, {nid: nid for nid in NODE_IDS})
assigner = CreditAssigner(dag, decay_factor=0.5, min_credit=0.01)

print(f"[2/7] Causal DAG built: {len(NODE_IDS)} nodes, {len(EDGES)} edges")
print("       Math:  foundations → algebra/calculus → linear-algebra/diff-eq")
print("       CS:    data-structures → algorithms → optimization/ml")
print()

# ── Create Learner ────────────────────────────────────────────────
learner = Learner(LearnerConfig(
    name="credit",
    baseline_rate=0.0,
    state_dir="/tmp/qortex-demo-credit",
))

print("[3/7] Learner created: credit (Thompson Sampling)")
print()

# ── Outcome simulation ───────────────────────────────────────────
LEAF_CONCEPTS = ["math:linear-algebra", "math:diff-eq", "cs:optimization", "cs:ml"]
OUTCOMES = ["accepted", "rejected", "partial"]
OUTCOME_REWARD = {"accepted": 1.0, "rejected": -1.0, "partial": 0.3}


def run_propagation(concept_ids: list[str], reward: float, round_num: int) -> dict:
    """Run a single credit propagation cycle."""
    assignments = assigner.assign_credit(concept_ids, reward)
    if not assignments:
        return {"concept_count": 0, "direct_count": 0, "ancestor_count": 0}

    updates = CreditAssigner.to_posterior_updates(assignments)
    learner.apply_credit_deltas(updates)

    direct = sum(1 for a in assignments if a.method == "direct")
    ancestor = sum(1 for a in assignments if a.method == "ancestor")
    total_alpha = sum(u.get("alpha_delta", 0.0) for u in updates.values())
    total_beta = sum(u.get("beta_delta", 0.0) for u in updates.values())

    emit(CreditPropagated(
        query_id=f"demo-q{round_num}",
        concept_count=len(assignments),
        direct_count=direct,
        ancestor_count=ancestor,
        total_alpha_delta=total_alpha,
        total_beta_delta=total_beta,
        learner="credit",
    ))

    return {
        "concept_count": len(assignments),
        "direct_count": direct,
        "ancestor_count": ancestor,
        "alpha_delta": total_alpha,
        "beta_delta": total_beta,
    }


# ── Phase 1: Positive signal — leaf concepts accepted ────────────
print("[4/7] Phase 1: Positive reinforcement — 15 rounds, leaf concepts accepted...")
for i in range(15):
    # Pick 1-2 random leaf concepts, reward them
    k = random.randint(1, 2)
    concepts = random.sample(LEAF_CONCEPTS, k)
    result = run_propagation(concepts, reward=1.0, round_num=i + 1)

    print(
        f"  Round {i+1:2d}: concepts={concepts!s:<45s} "
        f"direct={result['direct_count']} ancestors={result['ancestor_count']} "
        f"α+={result['alpha_delta']:.2f}"
    )
    time.sleep(2)

print()

# ── Phase 2: Mixed signal — some rejected ────────────────────────
print("[5/7] Phase 2: Mixed feedback — 20 rounds, varied outcomes...")
for i in range(20):
    concept = random.choice(LEAF_CONCEPTS)
    outcome = random.choices(OUTCOMES, weights=[0.5, 0.3, 0.2])[0]
    reward = OUTCOME_REWARD[outcome]

    result = run_propagation([concept], reward=reward, round_num=15 + i + 1)

    tag = {"accepted": "✓", "rejected": "✗", "partial": "~"}[outcome]
    delta_str = (
        f"α+={result['alpha_delta']:.2f}"
        if result["alpha_delta"] > 0
        else f"β+={result['beta_delta']:.2f}"
    )
    print(
        f"  Round {i+1:2d}: {tag} {outcome:<8s} concept={concept:<22s} "
        f"direct={result['direct_count']} ancestors={result['ancestor_count']} "
        f"{delta_str}"
    )
    time.sleep(1)

print()

# ── Phase 3: Targeted reinforcement — math wins, cs loses ────────
print("[6/7] Phase 3: Targeted — math accepted, cs rejected, 15 rounds...")
for i in range(15):
    if random.random() < 0.6:
        concept = random.choice(["math:linear-algebra", "math:diff-eq"])
        reward = 1.0
        outcome = "accepted"
    else:
        concept = random.choice(["cs:optimization", "cs:ml"])
        reward = -1.0
        outcome = "rejected"

    result = run_propagation([concept], reward=reward, round_num=35 + i + 1)

    tag = "✓" if outcome == "accepted" else "✗"
    delta_str = (
        f"α+={result['alpha_delta']:.2f}"
        if result["alpha_delta"] > 0
        else f"β+={result['beta_delta']:.2f}"
    )
    print(
        f"  Round {i+1:2d}: {tag} {concept:<22s} {delta_str}"
    )
    time.sleep(1)

print()

# ── Summary ───────────────────────────────────────────────────────
print("[7/7] Final posterior state:")
print()

posteriors = learner.posteriors()
if posteriors:
    # Sort by mean descending
    sorted_arms = sorted(posteriors.items(), key=lambda x: x[1]["mean"], reverse=True)
    print(f"  {'Concept':<25s} {'Mean':>6s} {'Alpha':>7s} {'Beta':>7s} {'Pulls':>6s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
    for arm_id, p in sorted_arms:
        print(
            f"  {arm_id:<25s} {p['mean']:>6.3f} {p['alpha']:>7.2f} "
            f"{p['beta']:>7.2f} {p['pulls']:>6d}"
        )
else:
    print("  (no posteriors yet — this shouldn't happen)")

print()
print("  Math concepts should have higher means (received more positive credit)")
print("  CS concepts should have lower means (received more negative credit)")
print("  Root/mid-layer concepts reflect decayed ancestor credit")
print()
print("=" * 64)
print("  Metrics are still being scraped. Check Grafana!")
print("  http://localhost:3010/d/qortex-main/qortex-observability")
print("  Scroll to: 'Credit Propagation' row")
print("=" * 64)

# Keep the process alive so Prometheus can keep scraping
print()
print("  Keeping metrics server alive (Ctrl+C to exit)...")
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\n  Done.")
