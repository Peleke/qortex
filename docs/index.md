# qortex

**Knowledge that learns.**

Your AI assistant forgets everything between conversations. qortex adds a knowledge graph that learns from every interaction. One command to install. Zero config.

![qortex pipeline](images/diagrams/pipeline.svg)

## Features

- **Graph-Enhanced Retrieval**: Combines vector similarity with structural graph traversal. Related concepts get promoted even if they don't share keywords.
- **Adaptive Learning**: Every feedback call updates retrieval weights. The system gets smarter the more you use it.
- **Compare and Prove**: `qortex_compare` runs the same query through cosine-only and graph-enhanced retrieval so you can see the difference on your own data.
- **Activity Tracking**: `qortex_stats` shows knowledge coverage, learning progress, and query activity at a glance.
- **Auto-Ingest**: Feed it docs, specs, or code. LLM extraction builds concepts, typed edges, and rules automatically.
- **Persistent by Default**: SQLite stores the knowledge graph, vector index, and learning state across restarts.
- **Framework Adapters**: Drop-in for LangChain VectorStore, Mastra MastraVector, and agno KnowledgeProtocol.
- **Projection Pipeline**: Source, Enricher, Target architecture for rule generation.
- **Multiple Backends**: InMemory (testing), Memgraph (production), SQLite (default persistent).

## Quick Example

### Search, explore, learn

```python
from qortex.client import LocalQortexClient

client = LocalQortexClient(vector_index, backend, embedding, mode="graph")

# Search: vec + graph combined scoring, rules auto-surfaced
result = client.query("OAuth2 authorization", domains=["security"], top_k=5)

# Explore: traverse typed edges from any result
explore = client.explore(result.items[0].node_id)
for edge in explore.edges:
    print(f"{edge.source_id} --{edge.relation_type}--> {edge.target_id}")

# Feedback: close the learning loop
client.feedback(result.query_id, {result.items[0].id: "accepted"})
```

### LangChain VectorStore (drop-in)

```python
from qortex.adapters.langchain_vectorstore import QortexVectorStore

vs = QortexVectorStore.from_texts(texts, embedding, domain="security")
docs = vs.similarity_search("authentication", k=5)
retriever = vs.as_retriever()
```

### Project rules

```python
from qortex.projectors.projection import Projection
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

projection = Projection(
    source=FlatRuleSource(backend=backend),
    target=BuildlogSeedTarget(persona_name="my_rules"),
)
result = projection.project(domains=["my_domain"])
```

Or use the CLI:

```bash
qortex project buildlog --domain my_domain --pending
```

## Installation

```bash
pip install qortex

# With Memgraph support
pip install qortex[memgraph]

# With all optional dependencies
pip install qortex[all]
```

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Getting Started**

    ---

    Install qortex and project your first rules in under 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-magnify:{ .lg .middle } **Querying**

    ---

    Graph-enhanced search with exploration and feedback

    [:octicons-arrow-right-24: Querying Guide](guides/querying.md)

-   :material-graph:{ .lg .middle } **Core Concepts**

    ---

    Understand domains, concepts, edges, and rules

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

-   :material-connection:{ .lg .middle } **Framework Adapters**

    ---

    Drop-in for LangChain, Mastra, CrewAI, and Agno

    [:octicons-arrow-right-24: Case Studies](tutorials/case-studies/index.md)

</div>

## License

MIT License. See [LICENSE](https://github.com/Peleke/qortex/blob/main/LICENSE) for details.
