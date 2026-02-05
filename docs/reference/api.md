# API Reference

This page provides a quick reference for the main qortex APIs. For detailed documentation, see the source code docstrings.

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
