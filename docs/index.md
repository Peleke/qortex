# qortex

<p style="font-style: italic; color: #888; margin-bottom: 0.5em;">
  Part of the <strong>qlawbox</strong> stack
  &nbsp;&middot;&nbsp; <a href="https://peleke.github.io/openclaw/">vindler</a>
  &nbsp;&middot;&nbsp; <a href="https://peleke.github.io/openclaw-sandbox/">bilrost</a>
  &nbsp;&middot;&nbsp; <a href="https://pypi.org/project/qortex/">PyPI</a>
</p>

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

### The Learning Layer

Most retrieval systems are static: they return the same results no matter how many times you correct them. qortex treats context engineering as a **learning problem**.

Every candidate that could appear in a prompt -- a retrieved concept, a tool, a file, a prompt component -- is modeled as an **arm** in a multi-armed bandit. Each arm carries a Beta(alpha, beta) posterior that encodes the system's belief about how useful that candidate is.

**Selection.** When qortex needs to fill a context window, it samples from each arm's posterior using Beta-Bernoulli Thompson Sampling. Arms with strong track records are exploited; uncertain arms are explored. A configurable `baseline_rate` (default 10%) forces uniform-random exploration to prevent premature convergence.

**Observation.** After the agent uses the selected context and the user provides feedback (`accepted`, `rejected`, or `partial`), a reward model maps that outcome to a float. The arm's posterior updates: `alpha += reward`, `beta += (1 - reward)`. Over time, good candidates rise and bad ones sink.

**Credit propagation.** Feedback does not stop at the item that was directly used. qortex builds a causal DAG from the knowledge graph's typed edges. When a rule receives a reward signal, credit flows backward through the DAG to ancestor concepts, decaying by hop distance and edge strength. This means a successful retrieval of "JWT Validation" also strengthens "Authentication", the concept it refines.

**Observability.** Every selection, observation, and posterior update emits a structured event. The `qortex-observe` package routes these to Prometheus metrics, and the pre-built Grafana dashboard (`qortex-main`) visualizes posterior means, selection rates, observation outcomes, credit propagation depth, and token budget usage in real time.

See the [Learning Layer Guide](guides/learning.md) for configuration, intervention options, and observability details.

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
- **[Learning Layer](guides/learning.md)** -- Thompson Sampling, credit propagation, and observability
- **[Framework Adapters](tutorials/case-studies/index.md)** -- Drop-in for LangChain, Mastra, CrewAI, Agno, and AutoGen
- **[Theory](tutorials/index.md)** -- Why knowledge graphs, causal reasoning, and the geometry of learning

## The qlawbox stack

qortex is the knowledge layer. The full stack:

| Component | Role | Docs |
|-----------|------|------|
| **[vindler](https://peleke.github.io/openclaw/)** | Agent runtime (OpenClaw fork) | [Docs](https://peleke.github.io/openclaw/) |
| **[bilrost](https://peleke.github.io/openclaw-sandbox/)** | Hardened VM isolation | [PyPI](https://pypi.org/project/bilrost/) |
| **qortex** | Knowledge graph with adaptive learning (this project) | [PyPI](https://pypi.org/project/qortex/) |

## License

MIT License. See [LICENSE](https://github.com/Peleke/qortex/blob/main/LICENSE) for details.
