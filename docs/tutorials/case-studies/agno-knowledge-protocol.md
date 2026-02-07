# Case Study: Replacing Agno's Knowledge Backend

> **Status**: E2E in progress — `tests/test_e2e_agno.py`
>
> **The hook**: Agno uses a duck-typed `KnowledgeProtocol`. No inheritance required. Implement 3 methods, pass to any Agno agent, and your knowledge source is live. qortex fits like a glove.

## What Agno Has

Agno provides a `KnowledgeProtocol` with flexible knowledge backends (ChromaDB, PgVector, LanceDB, etc.) and 19 embedder options. Agents get a `search_knowledge_base` tool automatically when knowledge is attached.

Their architecture is clean: `Agent(knowledge=your_thing)` — done. The agent calls `.search()` or `.retrieve()` at runtime.

| Agno Interface | qortex Implementation |
|---------------|----------------------|
| `knowledge.search(query)` | `QortexKnowledge.search(query)` → QortexClient.query() |
| `knowledge.retrieve(query)` | `QortexKnowledge.retrieve(query)` → list[Document] |
| `knowledge.build_context()` | System prompt instructions for the agent |
| Custom retriever callable | `QortexClient.query()` wrapped as callable |
| — (no equivalent) | `QortexKnowledge.feedback(outcomes)` |

## The Swap

<!-- TODO: Before/after showing ChromaDB → qortex swap -->
<!-- TODO: Show agent code stays IDENTICAL -->

### Current (Agno + ChromaDB)

```python
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.chroma import ChromaDb

knowledge = PDFKnowledgeBase(
    path="docs/",
    vector_db=ChromaDb(collection="my-docs"),
)

agent = Agent(knowledge=knowledge, search_knowledge=True)
agent.print_response("What is OAuth2?")
```

### After (Agno + qortex)

```python
from agno.agent import Agent
from qortex.adapters.agno import QortexKnowledge
from qortex.client import LocalQortexClient

client = LocalQortexClient(vector_index, backend, embedding_model)
knowledge = QortexKnowledge(client=client, domains=["security"])

agent = Agent(knowledge=knowledge, search_knowledge=True)
agent.print_response("What is OAuth2?")
# Same question. Same interface. Graph-enhanced answers.
```

Two lines changed. The agent doesn't know.

## What We Prove

**Protocol compliance**: `QortexKnowledge` satisfies Agno's `KnowledgeProtocol` duck type — no inheritance, no registration.

**Real embeddings**: sentence-transformers/all-MiniLM-L6-v2. Not mocked.

**Retrieve returns real Documents**: Agno `Document(content, id, name, meta_data, reranking_score)` — constructed from qortex QueryItems.

**build_context works**: Agent gets system prompt instructions for knowledge search.

**Feedback closes the loop**: After agent retrieves, outcomes feed back into qortex for future improvement.

## Why Agno + qortex > Agno + ChromaDB

| Dimension | Agno + ChromaDB | Agno + qortex |
|-----------|----------------|---------------|
| Search | Cosine similarity | Vec + graph-enhanced PPR |
| Learning | None | Feedback-driven teleportation factors |
| Cross-agent | Each agent has its own ChromaDB | Shared qortex backend, all agents benefit |
| Persistence | ChromaDB collection | SqliteVec + Memgraph |
| Insight | Black box | kg_coverage metric, promotion rate, convergence tracking |

## Next Steps

<!-- TODO: Demo with Ollama (fully local, zero API keys) -->
<!-- TODO: Show custom_retriever pattern (agno's knowledge_retriever param) -->
<!-- TODO: Benchmark: qortex vs ChromaDB on same Agno agent task -->
<!-- TODO: Multi-agent demo: Agent A gives feedback, Agent B retrieval improves -->
