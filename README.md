# qortex

**Knowledge graph ingestion engine for automated rule generation.**

qortex transforms unstructured content (books, docs, PDFs) into a knowledge graph, then projects actionable rules for AI agents, buildlog, and other consumers.

## Features

- **Graph-Enhanced Retrieval**: Vector similarity + Personalized PageRank over typed edges
- **Explore and Navigate**: Traverse typed edges, discover neighbors and linked rules from any search result
- **Rules Auto-Surfaced**: Query results include linked rules with relevance scores, zero consumer effort
- **Feedback-Driven Learning**: Consumer outcomes adjust PPR teleportation factors; results improve over time
- **Framework Adapters**: Drop-in for [LangChain VectorStore](https://github.com/Peleke/langchain-qortex), Mastra MCP, CrewAI, Agno
- **Flexible Ingestion**: PDF, Markdown, and text sources into a unified knowledge graph
- **Rich Type System**: 10 semantic relation types with 30 edge rule templates
- **Projection Pipeline**: Source, Enricher, Target architecture for rule generation
- **Universal Schema**: JSON Schema artifacts for any-language validation
- **Multiple Backends**: InMemory (testing), Memgraph (production with MAGE algorithms)

## Quick Start

```bash
pip install qortex

# With optional dependencies
pip install qortex[vec]       # numpy + sentence-transformers
pip install qortex[mcp]       # MCP server
pip install qortex[memgraph]  # Memgraph backend
pip install qortex[all]       # Everything
```

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

### LangChain VectorStore

```python
from langchain_qortex import QortexVectorStore

vs = QortexVectorStore.from_texts(texts, embedding, domain="security")
docs = vs.similarity_search("authentication", k=5)
retriever = vs.as_retriever()
```

See [langchain-qortex](https://github.com/Peleke/langchain-qortex) for the standalone package.

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

### MCP Server

Configure qortex as an MCP server for cross-language consumers:

```json
{
  "mcpServers": {
    "qortex": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/qortex", "qortex-mcp"]
    }
  }
}
```

Tools: `qortex_query`, `qortex_explore`, `qortex_rules`, `qortex_feedback`, `qortex_status`, `qortex_domains`, `qortex_ingest`.

## Framework Adapters

| Framework | Package / Adapter | What It Augments |
|-----------|------------------|-----------------|
| LangChain | [langchain-qortex](https://github.com/Peleke/langchain-qortex) | Any VectorStore (Chroma, FAISS, Pinecone) |
| Mastra | MCP tools | Any MastraVector implementation |
| CrewAI | `QortexKnowledgeStorage` | KnowledgeStorage (ChromaDB default) |
| Agno | `QortexKnowledge` | Any KnowledgeProtocol implementation |

All adapters expose the same qortex extras: `explore()`, `rules()`, and `feedback()`.

## Documentation

Full documentation: https://peleke.github.io/qortex/

## License

MIT
