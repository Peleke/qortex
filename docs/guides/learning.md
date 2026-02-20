# Learning Layer Guide

qortex uses Thompson Sampling to learn which retrieval candidates produce good outcomes. This guide covers configuration, observation, and intervention.

For the conceptual overview, see [Adaptive Learning](../getting-started/concepts.md#adaptive-learning) in Core Concepts.

## Overview

The learning layer models context engineering as a multi-armed bandit problem. Each candidate item (a concept, tool, file, or prompt fragment) is an **arm** with a Beta-Bernoulli posterior distribution. The system selects arms by sampling from their posteriors, observes outcomes from user feedback, and updates posteriors accordingly. Over time, it converges on the candidates that work.

The core loop is:

1. **Select** -- sample from posteriors, pick the top-k arms (possibly within a token budget).
2. **Use** -- the agent includes the selected items in its context.
3. **Observe** -- the user provides feedback (accepted, rejected, partial).
4. **Update** -- posteriors shift; credit propagates through the causal DAG.

## Configuration

The `LearnerConfig` dataclass controls all learning behavior:

```python
from qortex.learning import Learner, LearnerConfig

config = LearnerConfig(
    name="my-learner",       # Unique name; also used for state file naming
    baseline_rate=0.1,       # Probability of forced uniform exploration (default 10%)
    seed_boost=2.0,          # Alpha boost applied to seed arms on first use
    seed_arms=[],            # Arm IDs that start with boosted priors
    state_dir="",            # Override for state persistence path (default: ~/.qortex/learning/)
    max_arms=1000,           # Cap on tracked arms
    min_pulls=0,             # Force-include arms with fewer than N observations (cold-start protection)
)

learner = Learner(config)
```

### Configuration Reference

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `name` | `str` | (required) | Identifies the learner. Used as the state file name. |
| `baseline_rate` | `float` | `0.1` | Probability of uniform-random exploration per selection. Higher values explore more, converge slower. |
| `seed_boost` | `float` | `2.0` | Prior alpha for seed arms. A seed arm starts at `Beta(2.0, 1.0)` instead of `Beta(1, 1)`, biasing it toward selection. |
| `seed_arms` | `list[str]` | `[]` | Arm IDs to boost. Only applied when the arm has zero pulls (first creation). |
| `state_dir` | `str` | `""` | Custom directory for state files. Empty string uses `~/.qortex/learning/`. |
| `max_arms` | `int` | `1000` | Maximum tracked arms. Prevents unbounded state growth. |
| `min_pulls` | `int` | `0` | Arms with fewer than this many observations are force-included in every selection, bypassing Thompson Sampling. Useful for cold-start protection. |

### Reward Models

The reward model maps outcome strings to float rewards in `[0, 1]`:

| Model | `accepted` | `partial` | `rejected` | Unknown |
|-------|-----------|-----------|-----------| --------|
| `BinaryReward` | 1.0 | 0.0 | 0.0 | 0.0 |
| `TernaryReward` (default) | 1.0 | 0.5 | 0.0 | 0.0 |

You can also bypass the reward model entirely by passing a `reward` float directly to `observe()`.

### Strategy

The default (and currently only) strategy is `ThompsonSampling`. It implements the `LearningStrategy` protocol:

- **select()**: Samples from each arm's `Beta(alpha, beta)` posterior, ranks by sample, returns top-k within optional token budget.
- **update()**: Applies `alpha += reward`, `beta += (1 - reward)` -- the Beta-Bernoulli conjugate update.

Custom strategies can be plugged in by implementing the `LearningStrategy` protocol and passing them to the `Learner` constructor.

### Persistence Backends

| Backend | Install | Thread-safe | Use case |
|---------|---------|-------------|----------|
| `SqliteLearningStore` (default) | Built-in | Yes (WAL + thread lock) | Production, MCP server |
| `JsonLearningStore` | Built-in | No | Testing, single-process scripts |

Both backends partition state by a deterministic hash of the context dict, so the same arm can have independent posteriors across different task contexts.

State files are stored at `{state_dir}/{learner_name}.db` (SQLite) or `{state_dir}/{learner_name}.json` (JSON).

## Python API

### Basic select-observe loop

```python
from qortex.learning import Learner, LearnerConfig, Arm, ArmOutcome

learner = Learner(LearnerConfig(name="retrieval"))

# Define candidates
candidates = [
    Arm(id="concept:jwt_validation", token_cost=120),
    Arm(id="concept:auth_middleware", token_cost=200),
    Arm(id="concept:password_hashing", token_cost=150),
]

# Select the best 2 within a token budget
result = learner.select(candidates, context={"task": "security-review"}, k=2, token_budget=400)
for arm in result.selected:
    print(f"Selected: {arm.id} (score: {result.scores[arm.id]:.3f})")

# After the agent uses these and gets feedback:
learner.observe(ArmOutcome(arm_id="concept:jwt_validation", outcome="accepted", reward=1.0))
learner.observe(ArmOutcome(arm_id="concept:auth_middleware", outcome="rejected", reward=0.0))
```

### Inspecting posteriors

```python
# All posteriors in a context
posteriors = learner.posteriors(context={"task": "security-review"})
for arm_id, state in posteriors.items():
    print(f"{arm_id}: mean={state['mean']:.3f}, pulls={state['pulls']}")

# Top arms by posterior mean
for arm_id, state in learner.top_arms(k=5):
    print(f"{arm_id}: mean={state.mean:.3f}")
```

### Aggregate metrics

```python
metrics = learner.metrics()
# {
#   "learner": "retrieval",
#   "total_pulls": 47,
#   "total_reward": 31.5,
#   "accuracy": 0.6702,
#   "arm_count": 12,
#   "explore_ratio": 0.1,
# }
```

### Credit propagation

```python
from qortex.causal.credit import CreditAssigner
from qortex.causal.dag import CausalDAG

# Build DAG from knowledge graph edges
dag = CausalDAG.from_edges(edges)
assigner = CreditAssigner(dag=dag, decay_factor=0.5, min_credit=0.01)

# Assign credit when a rule linked to these concepts gets reward +1.0
assignments = assigner.assign_credit(
    rule_concept_ids=["jwt_validation", "auth_middleware"],
    reward=1.0,
)

# Convert to posterior deltas and apply
deltas = CreditAssigner.to_posterior_updates(assignments)
learner.apply_credit_deltas(deltas, context={"task": "security-review"})
```

## MCP Tools

All learning operations are available as MCP tools for use by any agent:

### qortex_learning_select

Select arms from a candidate pool using Thompson Sampling.

```json
{
  "learner": "retrieval",
  "candidates": [
    {"id": "concept:jwt_validation", "token_cost": 120},
    {"id": "concept:auth_middleware", "token_cost": 200}
  ],
  "k": 2,
  "token_budget": 400
}
```

### qortex_learning_observe

Record an outcome for a selected arm.

```json
{
  "learner": "retrieval",
  "arm_id": "concept:jwt_validation",
  "outcome": "accepted"
}
```

### qortex_learning_posteriors

Inspect current posterior distributions. Optionally filter by arm IDs.

```json
{
  "learner": "retrieval",
  "arm_ids": ["concept:jwt_validation", "concept:auth_middleware"]
}
```

### qortex_learning_metrics

Get aggregate learning metrics.

```json
{
  "learner": "retrieval"
}
```

### qortex_learning_reset

Delete learned posteriors. Scoped by arm IDs and/or context.

```json
{
  "learner": "retrieval",
  "arm_ids": ["concept:password_hashing"],
  "context": {"task": "security-review"}
}
```

### qortex_learning_session_start / session_end

Track a named learning session for audit and debugging.

## Observing Learning Dynamics (Grafana)

The learning layer emits three event types that flow through `qortex-observe` into Prometheus and the pre-built Grafana dashboard.

### Events

| Event | When | Key Fields |
|-------|------|------------|
| `LearningSelectionMade` | Every `select()` call | `learner`, `selected_count`, `excluded_count`, `is_baseline`, `token_budget`, `used_tokens` |
| `LearningObservationRecorded` | Every `observe()` call | `learner`, `arm_id`, `reward`, `outcome`, `context_hash` |
| `LearningPosteriorUpdated` | Every posterior change | `learner`, `arm_id`, `alpha`, `beta`, `pulls`, `mean` |
| `CreditPropagated` | Every credit assignment | `query_id`, `concept_count`, `direct_count`, `ancestor_count`, `total_alpha_delta`, `total_beta_delta` |

### Prometheus Metrics

| Metric | Type | Labels |
|--------|------|--------|
| `qortex_learning_selections_total` | Counter | `learner`, `baseline` |
| `qortex_learning_observations_total` | Counter | `learner`, `outcome` |
| `qortex_learning_posterior_mean` | Gauge | `learner`, `arm_id` |
| `qortex_learning_token_budget_used` | Histogram | -- |
| `qortex_credit_propagations_total` | Counter | `learner` |
| `qortex_credit_concepts_per_propagation` | Histogram | -- |
| `qortex_credit_alpha_delta_total` | Counter | -- |
| `qortex_credit_beta_delta_total` | Counter | -- |

### Grafana Dashboard Panels

The `qortex-main` dashboard includes two sections for the learning layer:

**Learning & Bandits:**

- **Selection Rate** -- `rate(qortex_learning_selections_total[5m])` split by `baseline=true/false`. When posteriors converge, the Thompson Sampling line (`baseline=false`) should dominate.
- **Observation Rate** -- `rate(qortex_learning_observations_total[5m])` split by `outcome`. In a converging system, `accepted` trends upward.
- **Posterior Mean (top 10 arms)** -- `topk(10, qortex_learning_posterior_mean)`. This is the learning itself. Mean near 1.0 = confident success. All arms at 0.5 = insufficient data.
- **Token Budget Usage** -- p50/p95 histograms of token budget utilization per selection.

**Credit Propagation:**

- **Credit Propagation Rate** -- propagations/sec through the causal DAG. Should track feedback rate. Zero while feedback flows means the feature flag is off or the DAG is empty.
- **Concepts per Propagation** -- p50/p95 of how many concepts receive credit per event. 3-5 is typical for a well-connected DAG.
- **Total Credit Propagations** -- lifetime count stat panel.

### Starting the Observability Stack

```bash
cd docker && docker compose up -d

# Open the dashboard
open http://localhost:3010/d/qortex-main/qortex-observability
```

Credit propagation requires the feature flag: `QORTEX_CREDIT_PROPAGATION=on`.

## Intervention: Tuning and Resetting

### Decaying stale posteriors

If source data changes and old learned signals are no longer valid, decay an arm's posterior toward the uniform prior:

```python
# Shrink alpha and beta by 10% (preserves mean ratio, weakens confidence)
learner.decay_arm("concept:old_pattern", decay_factor=0.9)
```

This multiplies both `alpha` and `beta` by the decay factor, preserving the mean ratio while reducing confidence. Floors at 0.01 to prevent degenerate distributions.

### Resetting specific arms

Delete learned state entirely for specific arms or contexts:

```python
# Reset one arm in all contexts
learner.reset(arm_ids=["concept:poisoned_data"])

# Reset all arms in a specific context
learner.reset(context={"task": "deprecated-workflow"})

# Full reset -- all arms, all contexts
count = learner.reset()
print(f"Deleted {count} arm states")
```

Via MCP:

```json
{
  "tool": "qortex_learning_reset",
  "arguments": {
    "learner": "retrieval",
    "arm_ids": ["concept:poisoned_data"]
  }
}
```

### Boosting seed arms

When introducing new arms you want the system to explore early, add them as seed arms:

```python
config = LearnerConfig(
    name="retrieval",
    seed_arms=["concept:new_pattern_v2"],
    seed_boost=3.0,  # Start at Beta(3.0, 1.0) instead of Beta(1, 1)
)
```

Seed boosts are applied only when the arm has zero pulls, so they are safe to leave in config permanently.

### Adjusting exploration rate

If the system is exploiting too aggressively (always picking the same arms) or exploring too much (not converging):

```python
# More exploration (useful early or when adding many new arms)
config = LearnerConfig(name="retrieval", baseline_rate=0.2)

# Less exploration (useful once posteriors are well-separated)
config = LearnerConfig(name="retrieval", baseline_rate=0.05)
```

### Cold-start protection

Force-include arms that have not been observed enough times:

```python
# Arms with fewer than 3 observations are always included
config = LearnerConfig(name="retrieval", min_pulls=3)
```

This bypasses Thompson Sampling for under-observed arms, ensuring every arm gets a minimum number of trials before the system can choose to ignore it.

## Architecture Summary

```
User Feedback
    |
    v
RewardModel (outcome -> float)
    |
    v
Learner.observe()
    |
    +---> ArmState update (alpha += r, beta += 1-r)
    |         |
    |         v
    |     LearningStore (SQLite or JSON)
    |
    +---> CreditAssigner (causal DAG)
              |
              v
          Ancestor posteriors updated (decayed credit)
              |
              v
          Learner.apply_credit_deltas()
              |
              v
          LearningStore (persist)

All steps emit events -> qortex-observe -> Prometheus -> Grafana
```

## Next Steps

- [Observability and Grafana Dashboard](observability.md) -- full metrics reference and dashboard panel guide
- [Core Concepts: Adaptive Learning](../getting-started/concepts.md#adaptive-learning) -- conceptual overview
- [Causal Reasoning tutorials](../tutorials/causal-dag/index.md) -- theory behind the causal DAG
- [The Geometry of Learning](../tutorials/fisher-information/index.md) -- information-geometric perspective on posterior dynamics
