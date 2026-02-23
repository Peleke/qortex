# qortex-learning

Bandit-based adaptive learning for [qortex](https://github.com/Peleke/qortex): Thompson Sampling with Beta-Bernoulli posteriors, persistent state via SQLite, and pluggable reward models.

<div align="center">

<!-- Architecture: Adaptive Learning Pipeline -->
<svg viewBox="0 0 620 380" xmlns="http://www.w3.org/2000/svg" aria-label="qortex-learning architecture: candidates flow through selection, observation, and posterior updates in a feedback loop">
  <style>
    .lrn-bg { fill: #0d1117; }
    .lrn-box { fill: #161b22; stroke: #30363d; stroke-width: 1; rx: 6; }
    .lrn-box-accent { fill: #161b22; stroke: rgb(168,85,247); stroke-width: 1.5; rx: 6; filter: url(#lrn-glow); }
    .lrn-label { font-family: 'JetBrains Mono', monospace; font-size: 8px; fill: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .lrn-title { font-family: system-ui, sans-serif; font-size: 13px; fill: #e6edf3; }
    .lrn-subtitle { font-family: system-ui, sans-serif; font-size: 10px; fill: #8b949e; }
    .lrn-flow { stroke: rgb(168,85,247); stroke-width: 1.2; stroke-dasharray: 4 3; fill: none; opacity: 0.5; }
    .lrn-flow-anim { animation: lrn-dash 2s linear infinite; }
    @keyframes lrn-dash { to { stroke-dashoffset: -14; } }
    .lrn-arrow { fill: rgb(168,85,247); opacity: 0.5; }
  </style>
  <defs>
    <filter id="lrn-glow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="620" height="380" class="lrn-bg"/>

  <!-- Candidates (top) -->
  <rect x="180" y="20" width="260" height="50" class="lrn-box"/>
  <text x="195" y="38" class="lrn-label">input</text>
  <text x="195" y="55" class="lrn-title">Arms (prompt variants, configs)</text>

  <!-- Select (accented) -->
  <rect x="180" y="110" width="260" height="55" class="lrn-box-accent"/>
  <text x="195" y="128" class="lrn-label">select</text>
  <text x="195" y="148" class="lrn-title">Thompson Sampling</text>

  <!-- Flow: input -> select -->
  <line x1="310" y1="70" x2="310" y2="110" class="lrn-flow lrn-flow-anim"/>
  <polygon points="310,108 306,100 314,100" class="lrn-arrow"/>

  <!-- Annotations -->
  <text x="460" y="125" class="lrn-subtitle">Beta-Bernoulli posteriors</text>
  <text x="460" y="140" class="lrn-subtitle">context-aware selection</text>
  <text x="460" y="155" class="lrn-subtitle">token cost weighting</text>

  <!-- Observe -->
  <rect x="180" y="205" width="260" height="55" class="lrn-box"/>
  <text x="195" y="223" class="lrn-label">observe</text>
  <text x="195" y="243" class="lrn-title">Reward Model (Binary / Ternary)</text>

  <!-- Flow: select -> observe -->
  <line x1="310" y1="165" x2="310" y2="205" class="lrn-flow lrn-flow-anim"/>
  <polygon points="310,203 306,195 314,195" class="lrn-arrow"/>

  <!-- Store -->
  <rect x="180" y="300" width="260" height="55" class="lrn-box"/>
  <text x="195" y="318" class="lrn-label">persist</text>
  <text x="195" y="338" class="lrn-title">SQLite / JSON store</text>

  <!-- Flow: observe -> store -->
  <line x1="310" y1="260" x2="310" y2="300" class="lrn-flow lrn-flow-anim"/>
  <polygon points="310,298 306,290 314,290" class="lrn-arrow"/>

  <!-- Feedback loop: store -> select -->
  <path d="M 180,327 C 130,327 100,200 100,137 C 100,75 150,137 180,137" class="lrn-flow lrn-flow-anim"/>
  <polygon points="178,141 170,137 178,133" class="lrn-arrow"/>
  <text x="70" y="230" class="lrn-label" transform="rotate(-90, 70, 230)">posteriors</text>
</svg>

</div>

## Install

```bash
pip install qortex-learning
```

## Quick Start

```python
from qortex.learning import Learner, LearnerConfig, Arm, ArmOutcome

# Create a learner with SQLite persistence
learner = await Learner.create(LearnerConfig(name="prompts"))

# Define candidates
candidates = [
    Arm(id="concise-v1", token_cost=10),
    Arm(id="detailed-v2", token_cost=15),
    Arm(id="structured-v3", token_cost=20),
]

# Select the best arm via Thompson Sampling
result = await learner.select(candidates, context={"task": "type-check"}, k=1)
print(f"Selected: {result.selected[0].id}")

# Observe the outcome
await learner.observe(ArmOutcome(
    arm_id="detailed-v2",
    outcome="accepted",
    reward=1.0,
))
```

## What It Does

**qortex-learning** provides a multi-armed bandit framework for adaptive selection. It powers the feedback loop in qortex: when users accept or reject retrieval results, the learning layer updates posterior distributions so future selections improve.

### Learner

The main interface. Manages arm selection, observation, and posterior tracking.

| Method | Purpose |
|--------|---------|
| `select(arms, context, k)` | Choose k arms via Thompson Sampling |
| `observe(outcome)` | Record an accept/reject signal, update posteriors |
| `batch_observe(outcomes)` | Bulk observation for batch feedback |
| `metrics()` | Selection counts, reward rates, posterior summaries |
| `top_arms(k)` | Top-k arms ranked by posterior mean |
| `decay_arm(arm_id, factor)` | Shrink learned signal toward prior |
| `posteriors()` | Raw posterior parameters for all arms |

### Strategies

Pluggable selection strategies via the `LearningStrategy` protocol:

| Strategy | Description |
|----------|-------------|
| `ThompsonSampling` | Beta-Bernoulli Thompson Sampling (default) |

### Reward Models

Convert raw outcomes to numeric rewards:

| Model | Mapping |
|-------|---------|
| `BinaryReward` | accepted=1, everything else=0 |
| `TernaryReward` | accepted=1, partial=0.5, rejected=0 |

### Persistence

State survives restarts via pluggable stores:

| Store | Description |
|-------|-------------|
| `SqliteLearningStore` | SQLite backend (default). Async via aiosqlite. |
| `JsonLearningStore` | JSON file backend. Good for debugging. |

Both implement the `LearningStore` protocol, so custom backends are straightforward.

## How It Fits

qortex-learning is the adaptive layer beneath `qortex_feedback`. When a user accepts or rejects a retrieval result:

1. The MCP server calls `learner.observe()` with the outcome
2. The reward model converts the outcome to a numeric signal
3. Thompson Sampling updates the Beta posterior for that arm
4. Next `learner.select()` samples from updated posteriors
5. The store persists state to SQLite so learning survives restarts

This creates the continuous learning loop shown on the [homepage](../index.md): retrieval gets smarter the more you use it.

## Requirements

- Python 3.11+
- [aiosqlite](https://pypi.org/project/aiosqlite/) (async SQLite)
- [qortex-observe](observe.md) (event emission for metrics/traces)

## License

MIT
