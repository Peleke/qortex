# qortex

**Knowledge that learns.**

Your AI assistant forgets everything between conversations. qortex adds a knowledge graph that learns from every interaction. One command to install. Zero config.

![qortex pipeline](images/diagrams/pipeline.svg)

## Install

**Claude Code**
```bash
claude mcp add qortex -- uvx qortex mcp-serve
```

**Cursor / Windsurf**
```bash
uvx qortex mcp-serve  # add as stdio MCP server in settings
```

**Any MCP client**
```bash
pip install qortex[all] && qortex mcp-serve
```

Once installed, your assistant automatically:

1. **Searches** the knowledge graph before answering architecture questions
2. **Retrieves** relevant concepts, relationships, and rules -- not just similar text
3. **Learns** from your feedback: accepted results get boosted, rejected ones get suppressed
4. **Persists** everything to SQLite so knowledge survives restarts

No config files. No API keys for the knowledge layer. Just start asking questions.

## The difference

| | Vanilla RAG | qortex |
|---|---|---|
| **Retrieval** | Cosine similarity (what's textually similar) | Graph-enhanced (what's structurally relevant) |
| **Context** | Flat chunks | Concepts + typed edges + rules |
| **Learning** | Static | Adapts from every accept/reject signal |
| **Cross-references** | None | Traverses REQUIRES, REFINES, USES edges |

## Prove it

Call `qortex_compare` to see the difference on your own data:

```json
{
  "summary": "Graph-enhanced retrieval found 2 item(s) that cosine missed, surfaced 1 rule(s), replaced 1 distractor(s).",
  "diff": {
    "graph_found_that_cosine_missed": [
      {"rank": 3, "id": "security:JWTValidation", "score": 0.72}
    ],
    "cosine_found_that_graph_dropped": [
      {"rank": 4, "id": "security:PasswordHashing", "score": 0.68}
    ],
    "rank_changes": [
      {"id": "security:AuthMiddleware", "vec_rank": 3, "graph_rank": 1, "delta": 2}
    ]
  }
}
```

Graph retrieval promotes structurally connected concepts (AuthMiddleware depends on JWTValidation) and demotes textually similar but unrelated results.

## How it works

- **Graph-enhanced retrieval**: Queries combine vector similarity with structural graph traversal. Related concepts get promoted even if they don't share keywords.
- **Adaptive learning**: Every `qortex_feedback` call updates retrieval weights via Thompson Sampling. The system gets smarter the more you use it.
- **Auto-ingest**: Feed it docs, specs, or code. LLM extraction builds concepts, typed edges, and rules automatically.
- **Persistent by default**: SQLite stores the knowledge graph, vector index, and learning state across restarts.
- **Activity tracking**: `qortex_stats` shows knowledge coverage, learning progress, and query activity at a glance.
- **Projection pipeline**: Source, Enricher, Target architecture for projecting knowledge into any format -- buildlog seeds, flat YAML, JSON, or custom targets.
- **Multiple backends**: InMemory (testing), Memgraph (production with MAGE algorithms), SQLite (default persistent).

## Quick example

### Search, explore, learn

```python
from qortex.client import LocalQortexClient

client = LocalQortexClient(vector_index, backend, embedding, mode="graph")

# Search: vec + graph combined scoring
result = client.query("OAuth2 authorization", domains=["security"], top_k=5)

# Explore: traverse typed edges from any result
explore = client.explore(result.items[0].node_id)
for edge in explore.edges:
    print(f"{edge.source_id} --{edge.relation_type}--> {edge.target_id}")

# Feedback: close the learning loop
client.feedback(result.query_id, {result.items[0].id: "accepted"})
```

## Framework adapters

Drop-in adapters for the frameworks you already use.

### agno KnowledgeProtocol

```python
from qortex.adapters.agno import QortexKnowledgeSource

knowledge = QortexKnowledgeSource(domains=["security"])
agent = Agent(knowledge=knowledge)
```

### LangChain VectorStore

```python
from langchain_qortex import QortexVectorStore

vs = QortexVectorStore.from_texts(texts, embedding, domain="security")
retriever = vs.as_retriever()
```

See [langchain-qortex](https://github.com/Peleke/langchain-qortex) for the standalone package.

### Mastra MastraVector

```typescript
import { QortexVector } from "@peleke.s/mastra-qortex";

const qortex = new QortexVector({ id: "qortex" });
await qortex.createIndex({ indexName: "docs", dimension: 384 });
const results = await qortex.query({ indexName: "docs", queryVector: q, topK: 10 });
```

See [@peleke.s/mastra-qortex](https://github.com/Peleke/mastra-qortex) for the standalone package.

| Framework | Package | Language | Interface |
|-----------|---------|----------|-----------|
| agno | Built-in adapter | Python | `KnowledgeProtocol` |
| LangChain | [`langchain-qortex`](https://github.com/Peleke/langchain-qortex) | Python | `VectorStore` ABC |
| Mastra | [`@peleke.s/mastra-qortex`](https://github.com/Peleke/mastra-qortex) | TypeScript | `MastraVector` abstract class |
| CrewAI | Built-in adapter | Python | `KnowledgeStorage` |
| AutoGen | Built-in adapter | Python | Async tool interface |
| Any MCP client | Built-in MCP server | Any | MCP tools (JSON-RPC) |

## Install extras

| Capability | Install | What you get |
|-----------|---------|-------------|
| Core + MCP tools | `pip install qortex` | Knowledge graph, MCP server, vector-level tools. Consumers provide embeddings. |
| Text-level search | `pip install qortex[vec]` | qortex embeds text with sentence-transformers. Adds ~2GB for PyTorch + model weights. |
| Persistent vectors | `pip install qortex[vec-sqlite]` | SQLite-backed vector index. Without this, vectors are in-memory only. |
| Production graph | `pip install qortex[memgraph]` | Memgraph backend for production-scale graph operations. |
| Everything | `pip install qortex[all]` | All of the above. |

## Next steps

- **[Quick Start](getting-started/quickstart.md)** -- Install qortex and run your first query in under 5 minutes
- **[Querying Guide](guides/querying.md)** -- Graph-enhanced search with exploration and feedback
- **[Core Concepts](getting-started/concepts.md)** -- Domains, concepts, typed edges, and the learning loop
- **[Framework Adapters](tutorials/case-studies/index.md)** -- Drop-in for LangChain, Mastra, CrewAI, Agno, and AutoGen
- **[Theory](tutorials/index.md)** -- Why knowledge graphs, causal reasoning, and the geometry of learning

## License

MIT License. See [LICENSE](https://github.com/Peleke/qortex/blob/main/LICENSE) for details.
