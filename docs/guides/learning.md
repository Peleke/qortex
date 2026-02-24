# Learning System

qortex includes a Thompson Sampling bandit system for online learning. It powers adaptive decisions — selecting prompts, ranking strategies, or choosing retrieval modes — and updates beliefs from feedback.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Learner                               │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   Strategy   │  │  Store (SQL  │  │  Event Emitter  │ │
│  │  (Thompson)  │  │  or Postgres)│  │  (metrics/OTel) │ │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘ │
│         │                │                    │          │
│    select(arms)    get/put state        emit events      │
└─────────┼────────────────┼────────────────────┼──────────┘
          │                │                    │
          ▼                ▼                    ▼
     Beta(α, β)      persistence          Prometheus
     sampling        (per arm)            + OTel spans
```

Each **Learner** manages one decision problem. It has:

- A **Strategy** (Thompson Sampling) that samples from Beta posteriors
- A **Store** that persists arm states (SQLite or PostgreSQL)
- An **Event Emitter** that fires metrics and trace events

## Concepts

| Term | Meaning |
|------|---------|
| **Learner** | Named decision-maker (e.g., `"prompt-optimizer"`) |
| **Arm** | One option to choose from (e.g., `"prompt:chain-of-thought"`) |
| **Context** | Situational hash that groups arms (enables contextual bandits) |
| **Selection** | Choosing k arms via Thompson Sampling |
| **Observation** | Recording an outcome (`"accepted"` or `"rejected"`) to update beliefs |
| **Posterior** | Beta(α, β) distribution representing current belief about an arm's reward rate |

## Quick Start

### Python API

```python
from qortex.learning import Learner

# Create a learner with default Thompson Sampling
learner = await Learner.create("prompt-optimizer")

# Define candidates
arms = [
    {"id": "prompt:basic", "token_cost": 100},
    {"id": "prompt:chain-of-thought", "token_cost": 200},
    {"id": "prompt:few-shot", "token_cost": 300},
]

# Select the best arm (Thompson Sampling)
selected = await learner.select(arms, k=1, token_budget=500)
print(f"Selected: {selected[0].id}")

# After observing the outcome
await learner.observe("prompt:chain-of-thought", outcome="accepted")
```

### REST API

```bash
# Select an arm
curl -X POST http://localhost:8400/v1/learning/select \
  -H "Content-Type: application/json" \
  -d '{
    "learner": "prompt-optimizer",
    "candidates": [
      {"id": "prompt:basic", "token_cost": 100},
      {"id": "prompt:chain-of-thought", "token_cost": 200}
    ],
    "k": 1,
    "token_budget": 500
  }'

# Record outcome
curl -X POST http://localhost:8400/v1/learning/observe \
  -H "Content-Type: application/json" \
  -d '{"learner": "prompt-optimizer", "arm_id": "prompt:chain-of-thought", "outcome": "accepted"}'

# View posteriors
curl http://localhost:8400/v1/learning/prompt-optimizer/posteriors

# View metrics (convergence, selection rates)
curl http://localhost:8400/v1/learning/prompt-optimizer/metrics
```

### MCP Tool

```
Use qortex_learning_select to choose between prompt strategies.
Use qortex_learning_observe to record the outcome.
```

## Storage Backends

### SQLite (default)

```bash
# No configuration needed — uses ~/.qortex/learning/<learner>.db
qortex serve
```

Each learner gets its own SQLite database file. Good for local development and single-process deployments.

### PostgreSQL

```bash
QORTEX_STORE=postgres \
PGVECTOR_HOST=localhost \
qortex serve
```

All learners share the `learning_arm_states` table, distinguished by a `learner_name` column. Required for multi-pod deployments where state must be shared.

Schema:

```sql
CREATE TABLE learning_arm_states (
    learner_name TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    arm_id       TEXT NOT NULL,
    alpha        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    beta         DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    pulls        INTEGER NOT NULL DEFAULT 0,
    total_reward DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    last_updated TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (learner_name, context_hash, arm_id)
);
```

## Thompson Sampling

The default strategy uses Beta-Bernoulli Thompson Sampling:

1. **Prior**: Each arm starts with Beta(1, 1) — uniform, no preference
2. **Selection**: Sample θ ~ Beta(α, β) for each arm, pick the highest
3. **Update**: On `"accepted"` → α += 1; on `"rejected"` → β += 1
4. **Token budget**: Arms exceeding the budget are filtered before sampling

This naturally balances exploration (trying uncertain arms) and exploitation (favoring arms with high observed reward rates). Arms with fewer observations have wider posteriors, so they occasionally "win" the sample even against arms with higher means — ensuring they get tried.

### Convergence

After ~50 observations per arm, the posteriors tighten and the system exploits the best arm most of the time. You can monitor convergence via:

```bash
curl http://localhost:8400/v1/learning/prompt-optimizer/metrics
```

The response includes selection rates, reward rates, and posterior statistics per arm.

## Observability

With `QORTEX_PROMETHEUS_ENABLED=true`, the learning system emits:

| Metric | Type | Description |
|--------|------|-------------|
| `qortex_learning_selections_total` | counter | Total arm selections |
| `qortex_learning_observations_total` | counter | Total observations recorded |
| `qortex_learning_selection_duration_seconds` | histogram | Selection latency |
| `qortex_learning_observation_duration_seconds` | histogram | Observation latency |
| `qortex_learning_reward_rate` | gauge | Current reward rate per arm |
| `qortex_learning_posterior_alpha` | gauge | Alpha parameter per arm |
| `qortex_learning_posterior_beta` | gauge | Beta parameter per arm |

All operations are traced via OpenTelemetry (`learning.select`, `learning.observe`, `learning.pg.get`, `learning.pg.put`).

## Next Steps

- [REST API](rest-api.md) — full HTTP endpoint reference
- [PostgreSQL Setup](postgres-setup.md) — configure postgres backends
- [Docker Infrastructure](docker.md) — run the full stack
