# Core Concepts

qortex is built around a knowledge graph that stores concepts and typed relationships. The graph also records feedback signals, so retrieval improves over time. Rules are one output format; the graph itself is the persistent layer.

## The Knowledge Graph

![subgraph-domain-domain-error-h](../images/diagrams/concepts-1-subgraph-domain-domain-error-h.svg)

## Domains

A **domain** is an isolated subgraph, like a schema in a database. Each domain contains concepts, edges, and rules that belong together semantically.

```python
backend.create_domain("error_handling", "Error handling patterns")
backend.create_domain("testing", "Testing strategies")
```

!!! tip "Domain Isolation"
    Concepts in different domains don't interact by default. This allows you to ingest multiple books or sources without cross-contamination.

## Concepts (Nodes)

A **concept** represents an idea, pattern, or entity extracted from source material.

```python
from qortex.core.models import ConceptNode

node = ConceptNode(
    id="circuit_breaker",           # Unique within the graph
    name="Circuit Breaker",         # Human-readable name
    description="Pattern that...",  # Full description
    domain="error_handling",        # Which domain it belongs to
    source_id="patterns_book",      # Provenance
    source_location="Chapter 3",    # Optional: where in source
    confidence=0.95,                # Extraction confidence
)
```

## Edges (Relationships)

An **edge** connects two concepts with a semantic relationship type.

```python
from qortex.core.models import ConceptEdge, RelationType

edge = ConceptEdge(
    source_id="circuit_breaker",
    target_id="timeout",
    relation_type=RelationType.REQUIRES,
    confidence=0.9,
)
```

### Relation Types

qortex supports 10 semantic relation types:

| Type | Meaning | Example |
|------|---------|---------|
| `REQUIRES` | A needs B to work | Circuit Breaker requires Timeout |
| `CONTRADICTS` | A and B are mutually exclusive | Retry contradicts Fail Fast |
| `REFINES` | A is a specific form of B | JWT refines Authentication |
| `IMPLEMENTS` | A is a concrete form of B | Redis implements Cache |
| `PART_OF` | A is a component of B | Handler is part of Middleware |
| `USES` | A depends on B | Service uses Database |
| `SIMILAR_TO` | A and B are analogous | Saga similar to Transaction |
| `ALTERNATIVE_TO` | A can substitute for B | gRPC alternative to REST |
| `SUPPORTS` | A provides evidence for B | Benchmark supports Optimization |
| `CHALLENGES` | A provides counter-evidence for B | Edge Case challenges Assumption |

## Rules

Rules are actionable guidelines extracted or derived from the knowledge graph.

### Explicit Rules

Rules stated directly in source material:

```python
from qortex.core.models import ExplicitRule

rule = ExplicitRule(
    id="rule:timeout",
    text="Always configure timeouts for external calls",
    domain="error_handling",
    source_id="patterns_book",
    concept_ids=["timeout", "circuit_breaker"],  # Related concepts
    category="architectural",
    confidence=1.0,
)
```

### Derived Rules

Rules generated from edges using templates. qortex has 30 built-in templates (3 variants x 10 relation types):

| Variant | Style | Example |
|---------|-------|---------|
| `imperative` | Direct command | "Ensure Circuit Breaker has Timeout configured" |
| `conditional` | When/then | "When using Circuit Breaker, ensure Timeout is available" |
| `warning` | Caution | "Using Circuit Breaker without Timeout may cause issues" |

## The Projection Pipeline

![kg-knowledge-graph-source](../images/diagrams/concepts-2-kg-knowledge-graph-source.svg)

1. **Source**: Extracts rules from the graph (explicit + derived)
2. **Enricher**: Adds context, antipatterns, rationale, tags
3. **Target**: Serializes to output format

See [Projecting Rules](../guides/projecting-rules.md) for details.

## Universal Rule Set Schema

All projected rules follow a universal schema that any consumer can validate:

```yaml
persona: my_rules        # Flat string identifier
version: 1               # Integer version
rules:
  - rule: "The rule text"
    category: architectural
    context: "When this applies"
    antipattern: "What violating looks like"
    rationale: "Why this matters"
    tags: [error_handling, patterns]
    provenance:
      id: rule:timeout
      domain: error_handling
      derivation: explicit  # or "derived"
      confidence: 0.95
metadata:
  source: qortex
  rule_count: 10
```

See [Interop Schema](../reference/interop-schema.md) for the full JSON Schema definition.

## Vector Indexing

qortex maintains a vector layer independent of the graph layer. Two implementations are available:

- **NumpyVectorIndex**: In-memory cosine similarity. Fast, no dependencies beyond numpy. Data is lost on restart.
- **SqliteVecIndex**: SQLite-backed persistence via `sqlite-vec`. Vectors survive restarts. Install with `pip install qortex[vec-sqlite]`.

The MCP server also exposes a separate registry of **named vector indexes** (the `qortex_vector_*` tools) for raw-vector consumers like MastraVector. These are independent of the text-level index used by `qortex_query`.

## Adaptive Learning

qortex includes a Thompson Sampling-based learning module that improves retrieval quality over time. This is the key differentiator: instead of returning the same results regardless of feedback, qortex models every candidate as a statistical belief and updates that belief from experience.

### What the Learning Layer Does

The learning layer treats context engineering as a multi-armed bandit problem. Every candidate item that could appear in a prompt -- a retrieved concept, a tool, a file, a prompt component -- is an **arm**. The system's goal is to learn which arms produce good outcomes and select them more often, while still exploring uncertain alternatives.

### Arms and Posteriors

Each arm carries a Beta(alpha, beta) posterior distribution that encodes the system's belief about that arm's success probability.

| Term | Meaning |
|------|---------|
| **Arm** | A candidate action: a concept to retrieve, a tool to invoke, a prompt fragment to include. Identified by a hierarchical ID like `"tool:search:v2"`. |
| **ArmState** | The posterior belief: `alpha` (pseudo-successes), `beta` (pseudo-failures), `pulls` (observation count), `total_reward`. |
| **Posterior mean** | `alpha / (alpha + beta)` -- the expected success rate. Starts at 0.5 (uniform prior) and shifts toward 0 or 1 with observations. |
| **Token cost** | Each arm can declare its token cost, enabling budget-aware selection that fills the context window optimally. |

A new arm starts with `Beta(1, 1)`, the uniform distribution -- maximum uncertainty. Every observation sharpens the belief.

### Selection

When qortex needs to choose which items to include in a prompt, it uses **Thompson Sampling**:

1. For each candidate arm, sample a value from its `Beta(alpha, beta)` posterior.
2. Rank candidates by sampled value (descending).
3. Select the top-k, optionally respecting a token budget.

Arms that are uncertain get sampled high sometimes and low sometimes -- this is how the system explores. Arms with strong track records get consistently high samples -- this is how it exploits.

A `baseline_rate` (default 10%) forces uniform-random selection to guarantee ongoing exploration even after posteriors have converged. Arms below `min_pulls` are force-included regardless of their posterior to protect against cold-start bias.

### Observation

After the agent uses the selected context and the user provides feedback, the system records an observation:

```python
learner.observe(ArmOutcome(arm_id="concept:jwt_validation", reward=1.0, outcome="accepted"))
```

The reward model maps outcomes to floats:

| Reward Model | accepted | partial | rejected |
|-------------|----------|---------|----------|
| `BinaryReward` | 1.0 | 0.0 | 0.0 |
| `TernaryReward` | 1.0 | 0.5 | 0.0 |

The posterior updates: `alpha += reward`, `beta += (1 - reward)`. This is the standard Beta-Bernoulli conjugate update -- exact, closed-form, and computationally trivial.

### Credit Propagation Through the Causal DAG

Feedback does not stop at the directly-used item. The `CreditAssigner` builds a causal DAG from the knowledge graph's typed edges and propagates credit backward to ancestor concepts.

**How it works:**

1. A rule linked to concepts `["jwt_validation", "auth_middleware"]` receives reward `+1.0`.
2. Direct concepts receive full credit: `alpha_delta = +1.0`.
3. The DAG is traversed upward. Each ancestor receives `credit * decay_factor * edge_weight`, where `decay_factor` defaults to 0.5 and `edge_weight` comes from the graph edge strength.
4. Propagation stops when credit falls below `min_credit` (default 0.01) or `max_depth` (default 50) is reached.
5. The resulting `alpha_delta` / `beta_delta` values are applied directly to each concept's posterior via `learner.apply_credit_deltas()`.

This means accepting a result about "JWT Validation" also strengthens "Authentication" (via REFINES), "Security Middleware" (via PART_OF), and other upstream concepts proportional to their causal distance.

### Connection to the Knowledge Graph

The learning layer and the knowledge graph reinforce each other:

- **Graph structure defines causal paths.** Typed edges (REQUIRES, REFINES, USES, etc.) determine how credit flows between concepts. A well-connected graph produces richer credit propagation.
- **Learning updates bias retrieval.** Personalized PageRank scores are adjusted by posterior means, so concepts the system has learned to trust rank higher.
- **Feedback closes the loop.** Every `qortex_feedback` call simultaneously updates the learning layer's posteriors and optionally propagates credit through the causal DAG.

### Persistence

Two backends store arm states:

- **`SqliteLearningStore`** (default): ACID-safe, concurrent-safe with WAL mode and thread locking. File layout: `~/.qortex/learning/{learner_name}.db`.
- **`JsonLearningStore`**: Simple JSON files, no locking. Suitable for single-process testing. File layout: `~/.qortex/learning/{learner_name}.json`.

Both partition state by context hash, so the same arm can have different posteriors in different contexts (e.g., different task types).

### MCP Tools

The learning layer is fully accessible via MCP:

| Tool | Purpose |
|------|---------|
| `qortex_learning_select` | Select arms from candidates using Thompson Sampling |
| `qortex_learning_observe` | Record an outcome and update posteriors |
| `qortex_learning_posteriors` | Inspect current posterior distributions |
| `qortex_learning_metrics` | Get aggregate metrics (pulls, reward, accuracy) |
| `qortex_learning_session_start` | Start a named session for tracking |
| `qortex_learning_session_end` | End a session and get a summary |
| `qortex_learning_reset` | Delete learned posteriors (full or scoped) |

See the [Learning Layer Guide](../guides/learning.md) for configuration, observability, and intervention options.

## Observability (qortex-observe)

`qortex-observe` is a workspace package that provides structured logging, event emission, and optional OpenTelemetry integration. It is a core dependency of qortex.

- **Structured logging**: `get_logger(__name__)` returns a logger with structured key-value output.
- **Event system**: `emit(event)` publishes typed events (e.g., `CreditPropagated`, `QueryServed`) to configurable subscribers.
- **Subscribers**: JSONL file sink, stdout sink, structlog, OpenTelemetry span exporter, and alert rules.
- **MCP tracing**: `mcp_trace_middleware` wraps each MCP tool call with distributed trace context.
- **Carbon tracking**: Optional GHG Protocol-aligned carbon footprint estimation for LLM calls.

Install `pip install qortex[observability]` for OpenTelemetry exporters. The base logging and event system works without extras.
