# Quick Start

This guide walks you through projecting rules from a knowledge graph in under 5 minutes.

## 1. Create a Backend

qortex uses backends to store the knowledge graph. For development, use `InMemoryBackend`:

```python
from qortex.core.memory import InMemoryBackend

backend = InMemoryBackend()
backend.connect()
```

## 2. Add Some Data

Create a domain and add concepts with edges:

```python
from qortex.core.models import ConceptNode, ConceptEdge, ExplicitRule, RelationType

# Create a domain
backend.create_domain("error_handling", "Error handling patterns")

# Add concepts
backend.add_node(ConceptNode(
    id="circuit_breaker",
    name="Circuit Breaker",
    description="Pattern that prevents cascading failures",
    domain="error_handling",
    source_id="patterns_book",
))

backend.add_node(ConceptNode(
    id="timeout",
    name="Timeout",
    description="Time limit for operations",
    domain="error_handling",
    source_id="patterns_book",
))

# Add an edge (circuit breaker REQUIRES timeout)
backend.add_edge(ConceptEdge(
    source_id="circuit_breaker",
    target_id="timeout",
    relation_type=RelationType.REQUIRES,
))

# Add an explicit rule
backend.add_rule(ExplicitRule(
    id="rule:timeout",
    text="Always configure timeouts for external calls",
    domain="error_handling",
    source_id="patterns_book",
))
```

## 3. Project Rules

Use the projection pipeline to generate rules:

```python
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.enrichers.template import TemplateEnricher
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget
from qortex.projectors.projection import Projection

projection = Projection(
    source=FlatRuleSource(backend=backend),
    enricher=TemplateEnricher(domain="error_handling"),
    target=BuildlogSeedTarget(persona_name="error_handling_rules"),
)

result = projection.project(domains=["error_handling"])
print(f"Generated {result['metadata']['rule_count']} rules")
```

This produces both:

- **Explicit rules**: Rules directly from your source material
- **Derived rules**: Rules generated from edge templates (e.g., "Circuit Breaker requires Timeout to function correctly")

## 4. Output the Result

The result is a dict in the universal rule set schema:

```python
import yaml
print(yaml.dump(result, default_flow_style=False))
```

Output:

```yaml
persona: error_handling_rules
version: 1
rules:
  - rule: Always configure timeouts for external calls
    category: error_handling
    provenance:
      id: rule:timeout
      domain: error_handling
      derivation: explicit
      confidence: 1.0
  - rule: Circuit Breaker requires Timeout to function correctly
    category: dependency
    provenance:
      id: derived:circuit_breaker->timeout:imperative
      domain: error_handling
      derivation: derived
      confidence: 1.0
      relation_type: requires
      template_id: requires:imperative
metadata:
  source: qortex
  rule_count: 2
```

## Using the CLI

The same workflow via CLI:

```bash
# Project to stdout
qortex project buildlog --domain error_handling

# Project to file
qortex project buildlog --domain error_handling -o rules.yaml

# Project to interop pending directory (for buildlog integration)
qortex project buildlog --domain error_handling --pending
```

## 5. Query the Graph

Use QortexClient to search, explore, and learn:

```python
from qortex.client import LocalQortexClient
from qortex.vec.index import NumpyVectorIndex

# Set up vector search
vector_index = NumpyVectorIndex(dimensions=384)
# ... (add embeddings to vector_index)

client = LocalQortexClient(
    vector_index=vector_index,
    backend=backend,
    embedding_model=my_embedding,
    mode="graph",
)

# Search
result = client.query("circuit breaker patterns", domains=["error_handling"], top_k=5)
for item in result.items:
    print(f"{item.score:.2f}: {item.content}")

# Rules auto-surfaced in results
for rule in result.rules:
    print(f"Rule: {rule.text}")

# Explore graph from a result
explore = client.explore(result.items[0].node_id)
for edge in explore.edges:
    print(f"{edge.source_id} --{edge.relation_type}--> {edge.target_id}")

# Feedback — close the learning loop
client.feedback(result.query_id, {result.items[0].id: "accepted"})
```

Or use it as a LangChain VectorStore:

```python
from qortex.adapters.langchain_vectorstore import QortexVectorStore

vs = QortexVectorStore(client=client, domain="error_handling")
docs = vs.similarity_search("timeout patterns", k=5)
retriever = vs.as_retriever()
```

See [Querying Guide](../guides/querying.md) for the full query pipeline.

## Next Steps

- [Core Concepts](concepts.md) - Deep dive into domains, concepts, edges, and rules
- [Querying](../guides/querying.md) - Search, explore, and learn from your graph
- [Projecting Rules](../guides/projecting-rules.md) - Master the projection pipeline
- [Consumer Integration](../guides/consumer-integration.md) - Connect to buildlog and other tools
- [Case Studies](../tutorials/case-studies/index.md) - LangChain, Mastra, CrewAI, Agno integrations
