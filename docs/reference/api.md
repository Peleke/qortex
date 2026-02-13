# API Reference

This page provides a quick reference for the main qortex APIs. For detailed documentation, see the source code docstrings.

## QortexClient

The consumer-facing protocol. All framework adapters and MCP tools target this interface.

### Protocol

```python
from qortex.client import QortexClient

class QortexClient(Protocol):
    def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 10,
        min_confidence: float = 0.0,
    ) -> QueryResult: ...

    def explore(self, node_id: str, depth: int = 1) -> ExploreResult | None: ...

    def rules(
        self,
        domains: list[str] | None = None,
        concept_ids: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> RulesResult: ...

    def feedback(
        self,
        query_id: str,
        outcomes: dict[str, str],
        source: str = "unknown",
    ) -> None: ...

    def status(self) -> dict: ...
    def domains(self) -> list[dict]: ...
    def ingest(self, source_path: str, domain: str) -> dict: ...
```

### LocalQortexClient

In-process implementation.

```python
from qortex.client import LocalQortexClient
from qortex.core.memory import InMemoryBackend
from qortex.vec.index import NumpyVectorIndex

vector_index = NumpyVectorIndex(dimensions=384)
backend = InMemoryBackend(vector_index=vector_index)
backend.connect()

client = LocalQortexClient(
    vector_index=vector_index,
    backend=backend,
    embedding_model=my_embedding,
    mode="graph",  # or "vec"
)

# Query
result = client.query("OAuth2 authorization", domains=["security"], top_k=5)

# Explore
explore = client.explore(result.items[0].node_id)

# Rules
rules = client.rules(concept_ids=[item.node_id for item in result.items])

# Feedback
client.feedback(result.query_id, {result.items[0].id: "accepted"})

# Add concepts (for adapters)
ids = client.add_concepts(texts=["Zero-trust architecture"], domain="security")

# Get nodes by ID (for adapters)
nodes = client.get_nodes(["sec:oauth", "sec:jwt"])
```

### Result Types

```python
from qortex.client import (
    QueryResult,    # query() return type
    QueryItem,      # Individual query result
    ExploreResult,  # explore() return type
    RulesResult,    # rules() return type
    NodeItem,       # Node in explore results
    EdgeItem,       # Edge in explore results
    RuleItem,       # Rule in any result
)
```

#### QueryResult

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | `str` | Unique ID for feedback |
| `items` | `list[QueryItem]` | Ranked results |
| `rules` | `list[RuleItem]` | Auto-surfaced linked rules |

#### QueryItem

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique item ID |
| `content` | `str` | `"Name: Description"` text |
| `score` | `float` | 0.0–1.0 relevance |
| `domain` | `str` | Source domain |
| `node_id` | `str` | Graph node ID (use for explore) |
| `metadata` | `dict` | Additional metadata |

#### ExploreResult

| Field | Type | Description |
|-------|------|-------------|
| `node` | `NodeItem` | The explored node |
| `edges` | `list[EdgeItem]` | Typed edges |
| `neighbors` | `list[NodeItem]` | Connected nodes |
| `rules` | `list[RuleItem]` | Linked rules |

#### RulesResult

| Field | Type | Description |
|-------|------|-------------|
| `rules` | `list[RuleItem]` | Matching rules |
| `domain_count` | `int` | Distinct domains |
| `projection` | `str` | Always `"rules"` |

#### NodeItem

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Node ID |
| `name` | `str` | Human-readable name |
| `description` | `str` | Full description |
| `domain` | `str` | Source domain |
| `confidence` | `float` | Extraction confidence |
| `properties` | `dict` | Additional properties |

#### EdgeItem

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | Source node ID |
| `target_id` | `str` | Target node ID |
| `relation_type` | `str` | e.g. `"REQUIRES"`, `"USES"` |
| `confidence` | `float` | Edge confidence |
| `properties` | `dict` | Additional properties |

#### RuleItem

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Rule ID |
| `text` | `str` | Rule text |
| `domain` | `str` | Source domain |
| `category` | `str` | e.g. `"security"`, `"architectural"` |
| `confidence` | `float` | Rule confidence |
| `relevance` | `float` | Relevance to query/context |
| `source_concepts` | `list[str]` | Linked concept IDs |
| `metadata` | `dict` | Additional metadata |

## MCP Tools

qortex runs as an MCP server exposing all client operations over JSON-RPC.

### qortex_query

Search the knowledge graph.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `context` | `str` | Yes | — |
| `domains` | `list[str]` | No | all |
| `top_k` | `int` | No | `10` |
| `min_confidence` | `float` | No | `0.0` |

Returns: `{query_id, items: [{id, content, score, domain, node_id, metadata}], rules: [{id, text, ...}]}`

### qortex_explore

Traverse the graph from a node.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `node_id` | `str` | Yes | — |
| `depth` | `int` | No | `1` |

Returns: `{node, edges, neighbors, rules}` or `null` if node not found.

### qortex_rules

Get projected rules.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `domains` | `list[str]` | No | all |
| `concept_ids` | `list[str]` | No | all |
| `categories` | `list[str]` | No | all |

Returns: `{rules: [...], domain_count, projection: "rules"}`

### qortex_feedback

Report outcomes for a query.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `query_id` | `str` | Yes | — |
| `outcomes` | `dict[str, str]` | Yes | — |
| `source` | `str` | No | `"unknown"` |

Returns: `{status: "recorded", processed: N}`

### qortex_status

Health check.

Returns: `{status: "ok", vector_search: bool, graph_backend: str, domains: int}`

### qortex_domains

List domains.

Returns: `{domains: [{name, description, node_count, edge_count, rule_count}]}`

### qortex_ingest

Ingest content.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `source_path` | `str` | Yes | — |
| `domain` | `str` | Yes | — |

Returns: `{status, domain, concepts, edges, rules}`

### qortex_ingest_text

Ingest raw text directly (no file needed).

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `text` | `str` | Yes | -- |
| `domain` | `str` | Yes | -- |
| `title` | `str` | No | `""` |

Returns: `{status, domain, concepts, edges, rules}`

### qortex_ingest_structured

Ingest structured JSON data.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `data` | `dict` | Yes | -- |
| `domain` | `str` | Yes | -- |

Returns: `{status, domain, concepts, edges, rules}`

### qortex_compare

Compare graph-enhanced retrieval against cosine-only retrieval on the same query. Use this to see what the graph adds.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `context` | `str` | Yes | -- |
| `domains` | `list[str]` | No | all |
| `top_k` | `int` | No | `5` |

Returns:
```json
{
  "query": "...",
  "vec_only": {"method": "Cosine similarity", "items": [...]},
  "graph_enhanced": {"method": "Graph-enhanced", "items": [...], "rules_surfaced": N, "rules": [...]},
  "diff": {
    "graph_found_that_cosine_missed": [...],
    "cosine_found_that_graph_dropped": [...],
    "rank_changes": [{"id": "...", "vec_rank": 3, "graph_rank": 1, "delta": 2}],
    "overlap": N
  },
  "summary": "Graph-enhanced retrieval found 2 item(s) that cosine missed, surfaced 1 rule(s)."
}
```

### qortex_stats

Knowledge coverage, learning progress, query activity, and persistence info.

Returns:
```json
{
  "knowledge": {"domains": N, "concepts": N, "edges": N, "rules": N, "domain_breakdown": {...}},
  "learning": {"learners": N, "total_observations": N, "learner_breakdown": {...}},
  "activity": {"queries_served": N, "feedback_given": N, "feedback_rate": 0.5, "outcomes": {...}},
  "health": {"backend": "InMemoryBackend", "vector_index": "NumpyVectorIndex", "persistence": {...}}
}
```

### Learning Tools

#### qortex_learning_select

Select items using adaptive learning (Thompson Sampling).

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `learner` | `str` | Yes | -- |
| `candidates` | `list[dict]` | Yes | -- |
| `context` | `dict` | No | `{}` |
| `k` | `int` | No | `1` |

Returns: `{selected_arms: [...], excluded_arms: [...], is_baseline: bool}`

#### qortex_learning_observe

Record whether a selected item worked well.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `learner` | `str` | Yes | -- |
| `arm_id` | `str` | Yes | -- |
| `outcome` | `str` | No | -- |
| `reward` | `float` | No | -- |
| `context` | `dict` | No | `{}` |

Returns: `{alpha, beta, pulls, total_reward, mean}`

#### qortex_learning_posteriors

Get posterior distributions for arms.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `learner` | `str` | Yes | -- |
| `context` | `dict` | No | `{}` |
| `arm_ids` | `list[str]` | No | all |

Returns: `{posteriors: {arm_id: {alpha, beta, pulls, mean, ...}}}`

#### qortex_learning_metrics

Get aggregate learning metrics.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `learner` | `str` | Yes | -- |

Returns: `{learner, total_pulls, total_reward, accuracy, arm_count, explore_ratio}`

#### qortex_learning_session_start

Start a learning session for tracking.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `learner` | `str` | Yes | -- |
| `session_name` | `str` | Yes | -- |

Returns: `{session_id: "..."}`

#### qortex_learning_session_end

End a learning session.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `learner` | `str` | Yes | -- |
| `session_id` | `str` | Yes | -- |

Returns: `{session_id, learner, selected_arms, outcomes, started_at, ended_at}`

## Learner (Python API)

Direct Python API for adaptive learning. Used by framework adapters (buildlog `QortexLearner`, etc.).

```python
from qortex.learning.learner import Learner
from qortex.learning.types import Arm, ArmOutcome, LearnerConfig

learner = Learner(LearnerConfig(name="my-learner", state_dir="/tmp/state"))
```

### Methods

| Method | Description |
|--------|-------------|
| `select(candidates, context, k)` | Select k arms from candidates using Thompson Sampling |
| `observe(outcome, context)` | Record a single observation and update posterior |
| `batch_observe(outcomes, context)` | Record multiple observations in one call |
| `top_arms(context, k)` | Return top-k arms by posterior mean, descending |
| `decay_arm(arm_id, decay_factor, context)` | Shrink an arm's learned signal toward the prior |
| `posteriors(context, arm_ids)` | Get current posteriors for arms |
| `metrics()` | Compute aggregate learning metrics |
| `apply_credit_deltas(deltas, context)` | Apply causal credit deltas to posteriors |
| `session_start(name)` | Start a named learning session |
| `session_end(session_id)` | End a session and return summary |

### Types

#### ArmState

| Field | Type | Description |
|-------|------|-------------|
| `alpha` | `float` | Beta distribution alpha (successes + prior) |
| `beta` | `float` | Beta distribution beta (failures + prior) |
| `pulls` | `int` | Total observations |
| `total_reward` | `float` | Cumulative reward |
| `last_updated` | `str` | ISO timestamp |
| `mean` | `float` | Posterior mean: alpha / (alpha + beta) |

#### ArmOutcome

| Field | Type | Description |
|-------|------|-------------|
| `arm_id` | `str` | Which arm was observed |
| `reward` | `float` | 0.0 to 1.0 (optional if outcome provided) |
| `outcome` | `str` | "accepted", "rejected", "partial", or custom |
| `context` | `dict` | Context for partitioned state |

#### LearnerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | -- | Learner identifier |
| `baseline_rate` | `float` | `0.1` | Probability of uniform random exploration |
| `seed_boost` | `float` | `2.0` | Alpha prior for seed arms |
| `seed_arms` | `list[str]` | `[]` | Arms with boosted priors |
| `state_dir` | `str` | `""` | Override for state persistence path |
| `max_arms` | `int` | `1000` | Cap on tracked arms |
| `min_pulls` | `int` | `0` | Force-include arms with fewer than N observations |

## Backends

### GraphBackend Protocol

```python
from qortex.core.backend import GraphBackend

class GraphBackend(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

    # Domains
    def create_domain(self, name: str, description: str | None = None) -> Domain: ...
    def get_domain(self, name: str) -> Domain | None: ...
    def list_domains(self) -> list[Domain]: ...
    def delete_domain(self, name: str) -> None: ...

    # Nodes
    def add_node(self, node: ConceptNode) -> None: ...
    def get_node(self, node_id: str) -> ConceptNode | None: ...
    def find_nodes(self, domain: str | None = None, pattern: str | None = None) -> list[ConceptNode]: ...

    # Edges
    def add_edge(self, edge: ConceptEdge) -> None: ...
    def get_edges(self, source_id: str | None = None, target_id: str | None = None) -> list[ConceptEdge]: ...

    # Rules
    def add_rule(self, rule: ExplicitRule) -> None: ...
    def get_rules(self, domain: str | None = None) -> list[ExplicitRule]: ...

    # Bulk operations
    def ingest_manifest(self, manifest: IngestionManifest) -> None: ...
```

### InMemoryBackend

In-memory implementation for testing.

```python
from qortex.core.memory import InMemoryBackend

backend = InMemoryBackend()
backend.connect()

# Use the backend...
backend.create_domain("test", "Test domain")
backend.add_node(node)
backend.add_edge(edge)

# Query
nodes = backend.find_nodes(domain="test")
edges = backend.get_edges(source_id="concept_a")
```

### MemgraphBackend

Production backend using Memgraph.

```python
from qortex.core.backend import MemgraphBackend

backend = MemgraphBackend(
    host="localhost",
    port=7687,
    username=None,
    password=None,
)
backend.connect()

# Additional Memgraph-specific methods
results = backend.query_cypher("MATCH (n) RETURN n LIMIT 10")
scores = backend.personalized_pagerank(source_ids=["concept_a"])
checkpoint_id = backend.checkpoint()
backend.restore(checkpoint_id)
```

## Projection Pipeline

### ProjectionSource Protocol

```python
from qortex.projectors.base import ProjectionSource
from qortex.core.models import Rule
from qortex.projectors.models import ProjectionFilter

class ProjectionSource(Protocol):
    def derive(
        self,
        domains: list[str] | None = None,
        filters: ProjectionFilter | None = None,
    ) -> list[Rule]: ...
```

### FlatRuleSource

Default source that extracts explicit and derived rules.

```python
from qortex.projectors.sources.flat import FlatRuleSource

source = FlatRuleSource(backend=backend)
rules = source.derive(domains=["error_handling"])
```

### Enricher Protocol

```python
from qortex.projectors.base import Enricher
from qortex.projectors.models import EnrichedRule

class Enricher(Protocol):
    def enrich(self, rules: list[Rule]) -> list[EnrichedRule]: ...
```

### TemplateEnricher

Fast, deterministic enrichment.

```python
from qortex.projectors.enrichers.template import TemplateEnricher

enricher = TemplateEnricher(domain="error_handling")
enriched = enricher.enrich(rules)
```

### ProjectionTarget Protocol

```python
from qortex.projectors.base import ProjectionTarget
from qortex.projectors.models import EnrichedRule

class ProjectionTarget(Protocol[T]):
    def serialize(self, rules: list[EnrichedRule]) -> T: ...
```

### BuildlogSeedTarget

Target for the universal schema format.

```python
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

target = BuildlogSeedTarget(
    persona_name="my_rules",
    version=1,
    graph_version="2026-02-05T12:00:00Z",  # Optional
)
result = target.serialize(enriched_rules)
# Returns dict in universal schema format
```

### Projection

Composes source, enricher, and target.

```python
from qortex.projectors.projection import Projection

projection = Projection(
    source=FlatRuleSource(backend=backend),
    enricher=TemplateEnricher(domain="error_handling"),  # Optional
    target=BuildlogSeedTarget(persona_name="rules"),
)

result = projection.project(
    domains=["error_handling"],  # Optional: all domains if None
    filters=ProjectionFilter(...),  # Optional
)
```

## Enrichment

### EnrichmentBackend Protocol

```python
from qortex.enrichment.base import EnrichmentBackend
from qortex.projectors.models import RuleEnrichment

class EnrichmentBackend(Protocol):
    def enrich_batch(
        self,
        rules: list[Rule],
        domain: str,
    ) -> list[RuleEnrichment]: ...
```

### AnthropicEnrichmentBackend

LLM-powered enrichment using Claude.

```python
from qortex.enrichment.anthropic import AnthropicEnrichmentBackend

backend = AnthropicEnrichmentBackend(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
)
enrichments = backend.enrich_batch(rules, domain="error_handling")
```

### EnrichmentPipeline

Orchestrates enrichment with fallback.

```python
from qortex.enrichment.pipeline import EnrichmentPipeline

pipeline = EnrichmentPipeline(
    backend=AnthropicEnrichmentBackend(),  # Optional
    # Falls back to TemplateEnrichmentFallback if backend fails
)

enriched = pipeline.enrich(rules, domain="error_handling")

# Check stats
print(pipeline.stats.succeeded)
print(pipeline.stats.failed)
```

## Templates

### EdgeRuleTemplate

Templates for deriving rules from edges.

```python
from qortex.core.templates import EdgeRuleTemplate, get_template, select_template

# Get specific template
template = get_template("requires:imperative")
text = template.render(source_name="A", target_name="B")

# Select template for an edge
from qortex.core.models import ConceptEdge, RelationType

edge = ConceptEdge(
    source_id="a",
    target_id="b",
    relation_type=RelationType.REQUIRES,
)
template = select_template(edge, variant="imperative")
```

### Template Registry

```python
from qortex.core.templates import TEMPLATE_REGISTRY, reset_template_registry

# List all templates
for template_id in TEMPLATE_REGISTRY:
    print(template_id)

# Reset for test isolation
reset_template_registry()
```

## Interop

### Configuration

```python
from qortex.interop import (
    InteropConfig,
    SeedsConfig,
    SignalsConfig,
    get_interop_config,
    write_config,
)

# Load config (from ~/.claude/qortex-consumers.yaml or defaults)
config = get_interop_config()

# Custom config
config = InteropConfig(
    seeds=SeedsConfig(
        pending=Path("~/.qortex/seeds/pending"),
        processed=Path("~/.qortex/seeds/processed"),
        failed=Path("~/.qortex/seeds/failed"),
    ),
    signals=SignalsConfig(
        projections=Path("~/.qortex/signals/projections.jsonl"),
    ),
)

# Write config
write_config(config)
```

### Seed Operations

```python
from qortex.interop import (
    write_seed_to_pending,
    list_pending_seeds,
    list_processed_seeds,
    list_failed_seeds,
    generate_seed_filename,
)

# Write a seed
path = write_seed_to_pending(
    seed_data=result,
    persona="my_rules",
    domain="error_handling",
    emit_signal=True,
)

# List seeds
pending = list_pending_seeds()
processed = list_processed_seeds()
failed = list_failed_seeds()  # Returns [(path, error_msg), ...]
```

### Signal Operations

```python
from qortex.interop import (
    ProjectionEvent,
    append_signal,
    read_signals,
)
from datetime import datetime, timezone

# Emit event
event = ProjectionEvent(
    event="projection_complete",
    persona="my_rules",
    domain="error_handling",
    path="/path/to/seed.yaml",
    rule_count=5,
    ts=datetime.now(timezone.utc).isoformat(),
)
append_signal(event)

# Read events
events = read_signals(
    since=datetime.now(timezone.utc),
    event_types=["projection_complete"],
)
```

### Schema Validation

```python
from qortex.interop_schemas import (
    validate_seed,
    validate_event,
    export_schemas,
    SEED_SCHEMA,
    EVENT_SCHEMA,
)

# Validate
errors = validate_seed(seed_dict)
errors = validate_event(event_dict)

# Export
seed_path, event_path = export_schemas("./schemas/")

# Get raw schemas
schema = SEED_SCHEMA.copy()
```

## Serialization

### serialize_ruleset

Low-level serialization to universal schema.

```python
from qortex.projectors.targets._serialize import serialize_ruleset

seed = serialize_ruleset(
    enriched_rules=rules,
    persona_name="my_rules",
    version=1,
    graph_version="2026-02-05T12:00:00Z",
    source_version="0.1.0",
)
```

### rule_to_dict

Convert a single rule to dict.

```python
from qortex.projectors.targets._serialize import rule_to_dict

rule_dict = rule_to_dict(enriched_rule)
```

## Exceptions

```python
from qortex.core.exceptions import (
    QortexError,           # Base exception
    DomainNotFoundError,   # Domain doesn't exist
    NodeNotFoundError,     # Node doesn't exist
    ConnectionError,       # Backend connection failed
)

try:
    backend.get_domain("nonexistent")
except DomainNotFoundError:
    print("Domain not found")
```
