#!/usr/bin/env python3
"""Live demo: watch learning metrics flow into Grafana.

Run this while watching:
  http://localhost:3010/d/qortex-main/qortex-observability

The "Learning & Bandits" row at the bottom will light up.
"""

from __future__ import annotations

import os
import sys
import time

# Force Prometheus metrics on port 9090
os.environ["QORTEX_PROMETHEUS_ENABLED"] = "true"
os.environ["QORTEX_PROMETHEUS_PORT"] = "9090"

from qortex.learning.learner import Learner
from qortex.learning.types import Arm, ArmOutcome, LearnerConfig
from qortex.observability import configure, reset as obs_reset

# ── Setup ─────────────────────────────────────────────────────────
obs_reset()
configure()

print("=" * 60)
print("  QORTEX LEARNING — LIVE GRAFANA DEMO")
print("=" * 60)
print()
print("  Prometheus metrics at: http://localhost:9090/metrics")
print("  Grafana dashboard at:  http://localhost:3010/d/qortex-main/qortex-observability")
print("  Scroll to: 'Learning & Bandits' row")
print()
print("  Press Ctrl+C to stop.")
print("=" * 60)
print()

# Wait for Prometheus to notice us
print("[1/6] Waiting 5s for Prometheus to discover /metrics endpoint...")
time.sleep(5)

# ── Create learner ────────────────────────────────────────────────
learner = Learner(LearnerConfig(
    name="prompt-optimizer",
    baseline_rate=0.1,
    state_dir="/tmp/qortex-demo-learning",
))

candidates = [
    Arm(id="prompt:basic", metadata={"type": "basic"}, token_cost=100),
    Arm(id="prompt:cot", metadata={"type": "chain-of-thought"}, token_cost=200),
    Arm(id="prompt:fewshot", metadata={"type": "few-shot"}, token_cost=300),
]

print("[2/6] Learner created: prompt-optimizer")
print(f"       Arms: {[a.id for a in candidates]}")
print()

# ── Phase 1: Initial exploration (all arms roughly equal) ─────────
print("[3/6] Phase 1: Initial exploration — 10 rounds, random rewards...")
for i in range(10):
    result = learner.select(candidates, context={"task": "type-checking"}, k=2)
    selected = [a.id for a in result.selected]

    for arm in result.selected:
        # Simulate: cot is slightly better even early on
        import random
        if arm.id == "prompt:cot":
            outcome = "accepted" if random.random() < 0.7 else "rejected"
        else:
            outcome = "accepted" if random.random() < 0.4 else "rejected"

        learner.observe(
            ArmOutcome(arm_id=arm.id, reward=0.0, outcome=outcome),
            context={"task": "type-checking"},
        )

    posteriors = learner.posteriors(context={"task": "type-checking"})
    means = {k: f"{v['mean']:.3f}" for k, v in posteriors.items()}
    print(f"  Round {i+1:2d}: selected={selected}  posteriors={means}")
    time.sleep(2)  # Let Prometheus scrape between rounds

print()

# ── Phase 2: Signal strengthens (cot clearly better) ─────────────
print("[4/6] Phase 2: Signal strengthens — 20 rounds, cot dominates...")
for i in range(20):
    result = learner.select(candidates, context={"task": "type-checking"}, k=1)
    selected_arm = result.selected[0]

    # cot always wins now, others always lose
    if selected_arm.id == "prompt:cot":
        learner.observe(
            ArmOutcome(arm_id=selected_arm.id, reward=0.0, outcome="accepted"),
            context={"task": "type-checking"},
        )
    else:
        learner.observe(
            ArmOutcome(arm_id=selected_arm.id, reward=0.0, outcome="rejected"),
            context={"task": "type-checking"},
        )

    posteriors = learner.posteriors(context={"task": "type-checking"})
    means = {k: f"{v['mean']:.3f}" for k, v in posteriors.items()}
    cot_pct = f"cot posterior: {posteriors.get('prompt:cot', {}).get('mean', 0):.3f}"
    print(f"  Round {i+1:2d}: picked={selected_arm.id:<16s} {cot_pct}  all={means}")
    time.sleep(1)

print()

# ── Phase 3: Convergence — cot almost always selected ─────────────
print("[5/6] Phase 3: Convergence test — 30 rounds, watch cot dominate selections...")
cot_wins = 0
for i in range(30):
    result = learner.select(candidates, context={"task": "type-checking"}, k=1)
    selected_arm = result.selected[0]
    is_cot = selected_arm.id == "prompt:cot"
    if is_cot:
        cot_wins += 1

    # Keep reinforcing the signal
    if is_cot:
        learner.observe(
            ArmOutcome(arm_id=selected_arm.id, reward=0.0, outcome="accepted"),
            context={"task": "type-checking"},
        )
    else:
        learner.observe(
            ArmOutcome(arm_id=selected_arm.id, reward=0.0, outcome="rejected"),
            context={"task": "type-checking"},
        )

    tag = "★ COT" if is_cot else "  ---"
    pct = cot_wins / (i + 1) * 100
    print(f"  Round {i+1:2d}: {tag}  picked={selected_arm.id:<16s}  cot_rate={pct:.0f}%  baseline={result.is_baseline}")
    time.sleep(0.5)

print()

# ── Summary ───────────────────────────────────────────────────────
print("[6/6] Final metrics:")
metrics = learner.metrics()
posteriors = learner.posteriors(context={"task": "type-checking"})

print(f"  Total pulls:  {metrics['total_pulls']}")
print(f"  Total reward: {metrics['total_reward']}")
print(f"  Accuracy:     {metrics['accuracy']}")
print(f"  Arm count:    {metrics['arm_count']}")
print()
print("  Final posteriors:")
for arm_id, p in sorted(posteriors.items()):
    print(f"    {arm_id:<20s}  mean={p['mean']:.3f}  alpha={p['alpha']:.1f}  beta={p['beta']:.1f}  pulls={p['pulls']}")

print()
print(f"  Chain-of-thought selected {cot_wins}/30 times ({cot_wins/30*100:.0f}%) in convergence phase")
print()
print("=" * 60)
print("  Metrics are still being scraped. Check Grafana!")
print("  http://localhost:3010/d/qortex-main/qortex-observability")
print("=" * 60)

# Keep the process alive so Prometheus can keep scraping
print()
print("  Keeping metrics server alive (Ctrl+C to exit)...")
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\n  Done.")
