# Architecture Overview

qortex is a knowledge graph ingestion engine that transforms unstructured content into actionable rules for AI agents.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph Sources["Source Material"]
        PDF[PDF]
        MD[Markdown]
        Text[Text]
    end

    subgraph Ingest["Ingestion Layer"]
        Parser[Parsers]
        Extract[Concept Extraction]
        Manifest[IngestionManifest]
    end

    subgraph KG["Knowledge Graph"]
        Backend[GraphBackend]
        InMem[InMemoryBackend]
        Memgraph[MemgraphBackend]
    end

    subgraph Project["Projection Layer"]
        Source[ProjectionSource]
        Enricher[Enricher]
        Target[ProjectionTarget]
    end

    subgraph Interop["Interop Layer"]
        Pending[pending/]
        Signals[signals.jsonl]
        Schema[JSON Schema]
    end

    subgraph Consumers["Consumers"]
        Buildlog[buildlog]
        MCP[MCP Server]
        Agent[AI Agents]
    end

    PDF --> Parser
    MD --> Parser
    Text --> Parser
    Parser --> Extract --> Manifest

    Manifest --> Backend
    Backend --> InMem
    Backend --> Memgraph

    Backend --> Source
    Source --> Enricher --> Target

    Target --> Pending
    Target --> Signals
    Schema --> Consumers

    Pending --> Buildlog
    Pending --> MCP
    Pending --> Agent
```

## Design Principles

### 1. Separation of Concerns

Each layer has a single responsibility:

| Layer | Responsibility |
|-------|----------------|
| Ingestion | Parse sources, extract concepts, produce manifests |
| Knowledge Graph | Store concepts/edges/rules, provide queries |
| Projection | Transform KG into rules, enrich, serialize |
| Interop | Distribute seeds, coordinate consumers |

### 2. Protocol-Driven Design

Core abstractions are Python protocols, enabling:

- Multiple backend implementations (InMemory, Memgraph, future: Neo4j)
- Pluggable enrichment backends (Template, Anthropic, future: custom)
- Extensible projection targets (Buildlog, flat, JSON)

```python
class GraphBackend(Protocol):
    def connect(self) -> None: ...
    def add_node(self, node: ConceptNode) -> None: ...
    # ...

class ProjectionSource(Protocol):
    def derive(self, domains: list[str] | None) -> list[Rule]: ...

class Enricher(Protocol):
    def enrich(self, rules: list[Rule]) -> list[EnrichedRule]: ...
```

### 3. Manifest as Boundary

The `IngestionManifest` is the contract between ingestion and storage:

```python
@dataclass
class IngestionManifest:
    source: SourceMetadata
    domain: str
    concepts: list[ConceptNode]
    edges: list[ConceptEdge]
    rules: list[ExplicitRule]
```

This allows:
- Ingestors to be developed independently
- KG to be agnostic of source formats
- Future: ingestion as a separate microservice

### 4. Universal Schema

All outputs follow a single schema that any consumer can validate:

```yaml
persona: <flat string>
version: <integer>
rules:
  - rule: <text>
    category: <string>
    provenance: {...}
metadata: {...}
```

Benefits:
- Language-agnostic validation via JSON Schema
- Forward compatibility (additive changes only)
- Clear contract between qortex and consumers

## Package Structure

```
qortex/
├── core/
│   ├── models.py          # ConceptNode, ConceptEdge, Rule, etc.
│   ├── backend.py         # GraphBackend protocol, MemgraphBackend
│   ├── memory.py          # InMemoryBackend
│   └── templates.py       # 30 edge rule templates
├── projectors/
│   ├── base.py            # ProjectionSource, Enricher, ProjectionTarget
│   ├── models.py          # EnrichedRule, ProjectionFilter
│   ├── projection.py      # Projection orchestrator
│   ├── sources/
│   │   └── flat.py        # FlatRuleSource
│   ├── enrichers/
│   │   └── template.py    # TemplateEnricher
│   └── targets/
│       ├── buildlog_seed.py  # BuildlogSeedTarget
│       └── _serialize.py     # serialize_ruleset()
├── enrichment/
│   ├── base.py            # EnrichmentBackend protocol
│   ├── anthropic.py       # AnthropicEnrichmentBackend
│   └── pipeline.py        # EnrichmentPipeline
├── interop.py             # Consumer interop protocol
├── interop_schemas.py     # JSON Schema definitions
└── cli/
    ├── __init__.py        # Typer app
    ├── infra.py           # qortex infra
    ├── ingest.py          # qortex ingest
    ├── project.py         # qortex project
    ├── inspect_cmd.py     # qortex inspect
    ├── viz.py             # qortex viz
    └── interop_cmd.py     # qortex interop
```

## Data Flow

### Ingestion Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Parser
    participant Extractor
    participant Backend

    User->>CLI: qortex ingest book.pdf --domain patterns
    CLI->>Parser: parse(book.pdf)
    Parser-->>CLI: chunks[]
    CLI->>Extractor: extract(chunks)
    Extractor-->>CLI: IngestionManifest
    CLI->>Backend: ingest_manifest(manifest)
    Backend-->>CLI: success
    CLI-->>User: Ingested 45 concepts, 67 edges, 23 rules
```

### Projection Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Source
    participant Enricher
    participant Target
    participant Interop

    User->>CLI: qortex project buildlog --domain patterns --pending
    CLI->>Source: derive(domains=["patterns"])
    Source-->>CLI: rules[]
    CLI->>Enricher: enrich(rules)
    Enricher-->>CLI: enriched_rules[]
    CLI->>Target: serialize(enriched_rules)
    Target-->>CLI: seed_dict
    CLI->>Interop: write_seed_to_pending(seed_dict)
    Interop-->>CLI: /path/to/seed.yaml
    CLI-->>User: Projected 35 rules to pending/
```

## Key Abstractions

### GraphBackend

The storage abstraction with two implementations:

| Implementation | Use Case | Features |
|----------------|----------|----------|
| `InMemoryBackend` | Testing, development | Fast, no setup |
| `MemgraphBackend` | Production | Persistence, Cypher, MAGE algorithms |

### Projection Pipeline

Composable pipeline: Source → Enricher → Target

```python
projection = Projection(
    source=FlatRuleSource(backend),
    enricher=TemplateEnricher(domain),  # Optional
    target=BuildlogSeedTarget(persona),
)
result = projection.project(domains=["patterns"])
```

### Edge Rule Templates

30 templates (3 variants × 10 relation types) for deriving rules from edges:

| Relation | imperative | conditional | warning |
|----------|------------|-------------|---------|
| REQUIRES | "Ensure A has B" | "When using A, ensure B" | "A without B may fail" |
| CONTRADICTS | "Choose A or B, not both" | "If using A, avoid B" | "A and B conflict" |
| ... | ... | ... | ... |

### Consumer Interop

Hybrid pull/push model for any consumer:

- **Pull**: Consumers scan `pending/` directory
- **Push**: Consumers tail `signals/projections.jsonl`
- **Validation**: JSON Schema for any-language validation

## Extension Points

### Custom Backend

```python
class MyBackend:
    def connect(self) -> None: ...
    def add_node(self, node: ConceptNode) -> None: ...
    # Implement GraphBackend protocol
```

### Custom Enrichment

```python
class MyEnrichmentBackend:
    def enrich_batch(self, rules: list[Rule], domain: str) -> list[RuleEnrichment]:
        # Your enrichment logic
```

### Custom Projection Target

```python
class MyTarget:
    def serialize(self, rules: list[EnrichedRule]) -> MyOutputFormat:
        # Your serialization logic
```

## Roadmap

### Phase 2: HippoRAG-Style Retrieval

Cross-domain retrieval using Personalized PageRank for pattern completion:

```mermaid
flowchart LR
    Query[Query] --> PPR[Personalized PageRank]
    PPR --> Concepts[Relevant Concepts]
    Concepts --> Rules[Related Rules]
    Rules --> Context[Augmented Context]
```

The scaffolding exists in `src/qortex/hippocampus/` but is not yet fully implemented.

### Phase 3: Causal DAG

Confidence feedback loops from reward events:

```mermaid
flowchart LR
    Rewards[Reward Events] --> DAG[Causal DAG]
    DAG --> Confidence[Updated Confidence]
    Confidence --> Rules[Better Rules]
```

See [GitHub Issues](https://github.com/Peleke/qortex/issues) for detailed roadmap.

## Next Steps

- [Projection Pipeline](projection-pipeline.md) - Deep dive into projection
- [Data Models](../reference/models.md) - Model reference
- [Consumer Integration](../guides/consumer-integration.md) - Interop details
