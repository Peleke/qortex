# Data Models

qortex uses a rich data model to capture knowledge graphs with full provenance.

## Core Models

### ConceptNode

A concept extracted from source material.

```python
from qortex.core.models import ConceptNode

node = ConceptNode(
    id="circuit_breaker",           # Unique identifier
    name="Circuit Breaker",         # Human-readable name
    description="Pattern that...",  # Full description
    domain="error_handling",        # Domain it belongs to
    source_id="patterns_book",      # Where it came from
    source_location="Chapter 3",    # Optional: specific location
    confidence=0.95,                # Extraction confidence
    properties={},                  # Extensible metadata
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | str | Yes | Unique identifier within the graph |
| `name` | str | Yes | Human-readable name |
| `description` | str | Yes | Full description |
| `domain` | str | Yes | Domain this belongs to |
| `source_id` | str | Yes | Source identifier for provenance |
| `source_location` | str | No | Location within source (chapter, page) |
| `confidence` | float | No | Extraction confidence (0-1), default 1.0 |
| `properties` | dict | No | Extensible metadata |

### ConceptEdge

A relationship between two concepts.

```python
from qortex.core.models import ConceptEdge, RelationType

edge = ConceptEdge(
    source_id="circuit_breaker",
    target_id="timeout",
    relation_type=RelationType.REQUIRES,
    confidence=0.9,
    bidirectional=False,
    properties={},
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_id` | str | Yes | Source concept ID |
| `target_id` | str | Yes | Target concept ID |
| `relation_type` | RelationType | Yes | Semantic relationship type |
| `confidence` | float | No | Edge confidence (0-1), default 1.0 |
| `bidirectional` | bool | No | Whether edge goes both ways |
| `properties` | dict | No | Extensible metadata |

### RelationType

Semantic relationship types between concepts.

```python
from qortex.core.models import RelationType

# Logical relationships
RelationType.CONTRADICTS   # A and B are mutually exclusive
RelationType.REQUIRES      # A requires B to be true/present
RelationType.REFINES       # A is a more specific form of B
RelationType.IMPLEMENTS    # A is a concrete implementation of B

# Compositional relationships
RelationType.PART_OF       # A is a component of B
RelationType.USES          # A uses/depends on B

# Similarity relationships
RelationType.SIMILAR_TO    # A and B are related/analogous
RelationType.ALTERNATIVE_TO  # A can substitute for B

# Epistemic relationships
RelationType.SUPPORTS      # A provides evidence for B
RelationType.CHALLENGES    # A provides counter-evidence for B
```

### ExplicitRule

A rule explicitly stated in source material.

```python
from qortex.core.models import ExplicitRule

rule = ExplicitRule(
    id="rule:timeout",
    text="Always configure timeouts for external calls",
    domain="error_handling",
    source_id="patterns_book",
    concept_ids=["timeout", "circuit_breaker"],
    source_location="Chapter 3, page 45",
    category="architectural",
    confidence=1.0,
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | str | Yes | Unique rule identifier |
| `text` | str | Yes | The rule text |
| `domain` | str | Yes | Domain this belongs to |
| `source_id` | str | Yes | Source identifier |
| `concept_ids` | list[str] | No | Related concept IDs |
| `source_location` | str | No | Location within source |
| `category` | str | No | Rule category (architectural, testing, etc.) |
| `confidence` | float | No | Rule confidence (0-1), default 1.0 |

### Rule

A rule as returned to consumers (includes both explicit and derived).

```python
from qortex.core.models import Rule

rule = Rule(
    id="rule:timeout",
    text="Always configure timeouts",
    domain="error_handling",
    derivation="explicit",  # or "derived"
    source_concepts=["timeout"],
    confidence=1.0,
    relevance=0.0,
    category="architectural",
    metadata={},  # Template info for derived rules
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | str | Yes | Unique rule identifier |
| `text` | str | Yes | The rule text |
| `domain` | str | Yes | Domain this belongs to |
| `derivation` | str | Yes | "explicit" or "derived" |
| `source_concepts` | list[str] | Yes | Concept IDs it came from |
| `confidence` | float | Yes | Rule confidence (0-1) |
| `relevance` | float | No | Relevance score from retrieval |
| `category` | str | No | Rule category |
| `metadata` | dict | No | Template info for derived rules |

### Domain

An isolated subgraph (like a database schema).

```python
from qortex.core.models import Domain

domain = Domain(
    name="error_handling",
    description="Error handling patterns and strategies",
    source_ids=["patterns_book"],
    concept_count=15,
    edge_count=23,
    rule_count=8,
)
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | str | Yes | Domain identifier |
| `description` | str | No | Human-readable description |
| `source_ids` | list[str] | No | Sources that contributed |
| `concept_count` | int | No | Number of concepts |
| `edge_count` | int | No | Number of edges |
| `rule_count` | int | No | Number of rules |
| `created_at` | datetime | No | Creation timestamp |
| `updated_at` | datetime | No | Last update timestamp |

## Ingestion Models

### IngestionManifest

The output of an ingestor, consumed by the KG.

```python
from qortex.core.models import IngestionManifest, SourceMetadata

manifest = IngestionManifest(
    source=SourceMetadata(...),
    domain="error_handling",
    concepts=[...],
    edges=[...],
    rules=[...],
    extraction_confidence=0.95,
    warnings=["Low confidence on some extractions"],
)
```

This is the contract between ingestion and storage. The KG doesn't know about PDFs or Markdown - it just knows manifests.

### SourceMetadata

Metadata about an ingested source.

```python
from qortex.core.models import SourceMetadata

source = SourceMetadata(
    id="patterns_book",
    name="Design Patterns Book",
    source_type="pdf",
    path_or_url="/path/to/book.pdf",
    content_hash="sha256:...",
    chunk_count=45,
    concept_count=23,
    rule_count=8,
)
```

## Projection Models

### EnrichedRule

A rule paired with its enrichment.

```python
from qortex.projectors.models import EnrichedRule, RuleEnrichment

enriched = EnrichedRule(
    rule=rule,
    enrichment=RuleEnrichment(
        context="When making external calls",
        antipattern="No timeout limits",
        rationale="Prevents cascading failures",
        tags=["timeout", "resilience"],
        enrichment_version=1,
        enriched_at=datetime.now(timezone.utc),
        enrichment_source="anthropic",
    ),
)
```

### RuleEnrichment

Enrichment data for a rule.

```python
from qortex.projectors.models import RuleEnrichment

enrichment = RuleEnrichment(
    context="When this rule applies",
    antipattern="What violating looks like",
    rationale="Why this matters",
    tags=["tag1", "tag2"],
    enrichment_version=1,
    enriched_at=datetime.now(timezone.utc),
    enrichment_source="template",  # or "anthropic"
    source_contexts=[],  # Re-enrichment history
)
```

### ProjectionFilter

Filter options for rule projection.

```python
from qortex.projectors.models import ProjectionFilter

filters = ProjectionFilter(
    min_confidence=0.8,
    categories=["architectural"],
    exclude_derived=False,
)
```

## Edge Rule Templates

qortex includes 30 templates for deriving rules from edges (3 variants x 10 relation types).

```python
from qortex.core.templates import EdgeRuleTemplate, get_template, TEMPLATE_REGISTRY

# Get a specific template
template = get_template("requires:imperative")

# Generate rule text
text = template.render(
    source_name="Circuit Breaker",
    target_name="Timeout",
)
# -> "Ensure Circuit Breaker has Timeout configured"

# List all templates
for template_id in TEMPLATE_REGISTRY:
    print(template_id)
```

Template variants:

| Variant | Style | Example |
|---------|-------|---------|
| `imperative` | Direct command | "Ensure A has B" |
| `conditional` | When/then | "When using A, ensure B" |
| `warning` | Caution | "Using A without B may cause issues" |

## Interop Models

### InteropConfig

Configuration for the consumer interop protocol.

```python
from qortex.interop import InteropConfig, SeedsConfig, SignalsConfig

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
```

### ProjectionEvent

An event in the signal log.

```python
from qortex.interop import ProjectionEvent

event = ProjectionEvent(
    event="projection_complete",
    persona="error_rules",
    domain="error_handling",
    path="/path/to/seed.yaml",
    rule_count=5,
    ts="2026-02-05T12:00:00Z",
    source="qortex",
    source_version="0.1.0",
)
```

## Type Hints

All models use Python dataclasses with full type hints:

```python
from qortex.core.models import ConceptNode, ConceptEdge, RelationType
from typing import reveal_type

node = ConceptNode(...)
reveal_type(node.confidence)  # float
reveal_type(node.properties)  # dict[str, Any]
```

## Serialization

Models serialize cleanly to dict/JSON:

```python
import dataclasses
import json

node = ConceptNode(...)
node_dict = dataclasses.asdict(node)
node_json = json.dumps(node_dict)
```

For the universal schema format, use `serialize_ruleset()`:

```python
from qortex.projectors.targets._serialize import serialize_ruleset

seed = serialize_ruleset(
    enriched_rules=rules,
    persona_name="my_rules",
    version=1,
)
```
