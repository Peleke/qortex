# Core Concepts

qortex is built around a knowledge graph model that captures concepts, their relationships, and actionable rules.

## The Knowledge Graph

```mermaid
graph TB
    subgraph Domain["Domain: error_handling"]
        CB[Circuit Breaker]
        TO[Timeout]
        RT[Retry]
        FF[Fail Fast]

        CB -->|REQUIRES| TO
        RT -->|CONTRADICTS| FF
    end

    subgraph Rules
        R1[Explicit Rule]
        R2[Derived Rule]
    end

    CB -.->|source| R2
    TO -.->|source| R2
```

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

Rules generated from edges using templates. qortex has 30 built-in templates (3 variants × 10 relation types):

| Variant | Style | Example |
|---------|-------|---------|
| `imperative` | Direct command | "Ensure Circuit Breaker has Timeout configured" |
| `conditional` | When/then | "When using Circuit Breaker, ensure Timeout is available" |
| `warning` | Caution | "Using Circuit Breaker without Timeout may cause issues" |

## The Projection Pipeline

```mermaid
flowchart LR
    KG[(Knowledge Graph)] --> Source
    Source --> Enricher
    Enricher --> Target
    Target --> Output[YAML/JSON/Seed]

    subgraph Pipeline
        Source[FlatRuleSource]
        Enricher[TemplateEnricher]
        Target[BuildlogSeedTarget]
    end
```

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
