# qortex

Graph-enhanced retrieval engine. Transforms unstructured content into a knowledge graph with typed edges, then uses Personalized PageRank for structurally-aware search with feedback-driven learning.

## Features

- **Graph-enhanced retrieval**: Vector similarity + PPR over typed edges (REQUIRES, REFINES, USES...)
- **Explore and navigate**: Traverse typed edges, discover neighbors and linked rules from any search result
- **Rules auto-surfaced**: Query results include linked rules with relevance scores, zero consumer effort
- **Feedback-driven learning**: Consumer outcomes adjust PPR teleportation factors; results improve over time
- **Framework adapters**: Drop-in for [LangChain](https://github.com/Peleke/langchain-qortex), [Mastra](https://github.com/Peleke/mastra-qortex), and any MCP client
- **Flexible ingestion**: PDF, Markdown, and text sources into a unified knowledge graph
- **Projection pipeline**: Source, Enricher, Target architecture for rule generation
- **Multiple backends**: InMemory (testing), Memgraph (production with MAGE algorithms)

## Install

```bash
pip install qortex            # Core: numpy, fastmcp, typer, pyyaml
pip install qortex[vec]       # + sentence-transformers (text-level search)
pip install qortex[vec-sqlite] # + sqlite-vec (persistent vector index)
pip install qortex[memgraph]  # + neo4j driver (production graph backend)
pip install qortex[all]       # Everything
```

### What's included where

| Capability | Install | What you get |
|-----------|---------|-------------|
| Vector-level MCP tools (`qortex_vector_*`) | `pip install qortex` | Create indexes, upsert/query raw vectors, metadata filters. Consumers provide their own embeddings. |
| Text-level search (`qortex_query`) | `pip install qortex[vec]` | qortex embeds your text with sentence-transformers. Adds ~2GB for PyTorch + model weights. |
| Persistent vectors | `pip install qortex[vec-sqlite]` | SqliteVec index survives restarts. Without this, vectors are in-memory only. |
| Production graph | `pip install qortex[memgraph]` | Memgraph backend with MAGE algorithms for real PPR. Default is in-memory. |

**For MCP consumers** (Mastra, Claude Desktop, etc.) that provide their own embeddings: the base `pip install qortex` is sufficient. The `[vec]` extra is only needed if you want qortex to embed text for you.

## Quick start

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

### Mastra MastraVector

```typescript
import { QortexVector } from "@peleke.s/mastra-qortex";

const qortex = new QortexVector({ id: "qortex" });
await qortex.createIndex({ indexName: "docs", dimension: 384 });
await qortex.upsert({ indexName: "docs", vectors: embeddings, metadata });
const results = await qortex.query({ indexName: "docs", queryVector: q, topK: 10 });
```

See [@peleke.s/mastra-qortex](https://github.com/Peleke/mastra-qortex) for the standalone package.

### MCP server

```bash
qortex mcp-serve  # stdio transport, works with Claude Desktop / any MCP client
```

Tools: `qortex_query`, `qortex_explore`, `qortex_rules`, `qortex_feedback`, `qortex_ingest`, `qortex_domains`, `qortex_status`, plus 9 vector-level tools (`qortex_vector_*`).

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

## Framework adapters

| Framework | Package | Language | Interface |
|-----------|---------|----------|-----------|
| LangChain | [`langchain-qortex`](https://github.com/Peleke/langchain-qortex) | Python | `VectorStore` ABC |
| Mastra | [`@peleke.s/mastra-qortex`](https://github.com/Peleke/mastra-qortex) | TypeScript | `MastraVector` abstract class |
| Any MCP client | Built-in MCP server | Any | MCP tools (JSON-RPC) |

## Architecture

```
Sources (PDF/MD/Text)
  -> Ingestion (LLM extraction -> concepts + typed edges + rules)
    -> Knowledge Graph (InMemoryBackend | Memgraph)
      -> VectorIndex (NumpyVectorIndex | SqliteVecIndex)
        -> Retrieval (PPR + cosine -> combined scoring)
          -> Consumers (LangChain, Mastra, MCP, buildlog, agents)
```

## Documentation

- [Full docs](https://peleke.github.io/qortex/)
- [LangChain integration](https://github.com/Peleke/langchain-qortex)
- [Mastra integration](https://github.com/Peleke/mastra-qortex)

## License

MIT
