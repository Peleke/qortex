# qortex

Graph-enhanced retrieval engine. Transforms unstructured content into a knowledge graph with typed edges, then uses Personalized PageRank for structurally-aware search with feedback-driven learning.

## Features

- **Graph-enhanced retrieval** — PPR over typed edges (REQUIRES, REFINES, USES...), not just cosine similarity
- **Feedback-driven learning** — accepted/rejected signals adjust teleportation factors for future queries
- **Rule surfacing** — domain rules linked to concepts appear automatically in results
- **Graph exploration** — navigate typed edges and neighbors from any search result
- **Framework adapters** — drop-in integrations for LangChain, Mastra, and MCP clients
- **Projection pipeline** — project rules to any consumer (buildlog, CI, agents)

## Quick start

### QortexClient (Python)

```python
from qortex.client import LocalQortexClient

client = LocalQortexClient(vector_index, backend, embedding_model, mode="graph")

result = client.query("authentication protocols", domains=["security"], top_k=5)
for item in result.items:
    print(f"{item.score:.2f}: {item.content}")

# Graph exploration from any result
explored = client.explore(result.items[0].node_id)
print(explored.edges)      # Typed relationships
print(explored.neighbors)  # Connected concepts
print(explored.rules)      # Linked rules

# Feedback loop
client.feedback(result.query_id, {result.items[0].id: "accepted"})
```

### MCP server

```bash
pip install qortex
qortex mcp-serve  # stdio transport, works with Claude Desktop / any MCP client
```

Tools: `qortex_query`, `qortex_explore`, `qortex_rules`, `qortex_feedback`, `qortex_ingest`, `qortex_domains`, `qortex_status`, plus 9 vector-level tools (`qortex_vector_*`).

## Framework adapters

| Framework | Package | Language | Interface |
|-----------|---------|----------|-----------|
| LangChain | [`langchain-qortex`](https://github.com/Peleke/langchain-qortex) | Python | `VectorStore` ABC |
| Mastra | [`@peleke/mastra-qortex`](https://github.com/Peleke/mastra-qortex) | TypeScript | `MastraVector` abstract class |
| Any MCP client | Built-in MCP server | Any | MCP tools (JSON-RPC) |

## Architecture

```
Sources (PDF/MD/Text)
  → Ingestion (LLM extraction → concepts + typed edges + rules)
    → Knowledge Graph (InMemoryBackend | Memgraph)
      → VectorIndex (NumpyVectorIndex | SqliteVecIndex)
        → Retrieval (PPR + cosine → combined scoring)
          → Consumers (LangChain, Mastra, MCP, buildlog, agents)
```

## Documentation

- [Full docs](https://peleke.github.io/qortex/)
- [LangChain integration](https://github.com/Peleke/langchain-qortex)
- [Mastra integration](https://github.com/Peleke/mastra-qortex)
- [Case studies](docs/tutorials/case-studies/)

## License

MIT
