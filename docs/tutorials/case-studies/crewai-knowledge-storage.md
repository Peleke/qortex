# Case Study: Replacing CrewAI's KnowledgeStorage

> **Status**: Compat proven — `tests/test_framework_compat.py`, `tests/test_dropin_dogfood.py`
>
> **The hook**: CrewAI defaults to ChromaDB for knowledge storage. Swap in qortex and every crew gets graph retrieval + learning — without touching agent or task code.

## What CrewAI Has

CrewAI's Knowledge system uses `BaseKnowledgeStorage` (abstract base class) backed by ChromaDB. Agents get knowledge injected into their context automatically during task execution.

| CrewAI Interface | qortex Implementation |
|-----------------|----------------------|
| `storage.search(query, limit, score_threshold)` | `QortexKnowledgeStorage.search()` → QortexClient.query() |
| `storage.save(documents)` | No-op (qortex uses file-based ingestion) |
| `storage.reset()` | No-op (qortex storage is persistent) |
| `SearchResult` TypedDict | `{id, content, metadata, score}` — exact match |
| — (no equivalent) | `.feedback(outcomes)` |

**Key detail**: CrewAI passes `query` as `list[str]`, not `str`. We found this bug during dogfooding and fixed it.

## The Swap

<!-- TODO: Before/after showing ChromaDB → qortex in a real Crew -->
<!-- TODO: Show Crew + Agent + Task code stays IDENTICAL -->

### Current (CrewAI + ChromaDB)

```python
from crewai import Agent, Crew, Task
from crewai.knowledge import Knowledge, TextKnowledgeSource

knowledge = Knowledge(
    collection_name="security-docs",
    sources=[TextKnowledgeSource(content="OAuth2 is...")],
)

agent = Agent(role="Security Analyst", knowledge_sources=[knowledge])
```

### After (CrewAI + qortex)

```python
from crewai import Agent, Crew, Task
from qortex.adapters.crewai import QortexKnowledgeStorage
from qortex.client import LocalQortexClient

client = LocalQortexClient(vector_index, backend, embedding_model)
storage = QortexKnowledgeStorage(client=client, domains=["security"])

# Wire into CrewAI's knowledge system
# Agent code stays exactly the same
```

## What We Proved

**SearchResult shape match**: Verified against the actual `SearchResult` TypedDict from crewAI source at `crewai/rag/types.py`. Exact key and type match.

**list[str] query handling**: CrewAI passes queries as `list[str]`. Our adapter accepts both `list[str]` and `str`, joining list elements with space (matching crewAI's internal behavior).

**save/reset graceful no-ops**: CrewAI expects these methods to exist. They're no-ops in qortex (persistent storage, file-based ingestion).

**Cross-framework consistency**: Same query through crewAI adapter returns identical IDs and scores as Mastra, LangChain, and Agno adapters.

## Next Steps

<!-- TODO: Full Crew E2E with real tasks (needs LLM API key) -->
<!-- TODO: Show knowledge_sources wiring into Crew -->
<!-- TODO: Demo: CrewAI crew with feedback loop improving over multiple task runs -->
<!-- TODO: Benchmark: qortex vs ChromaDB on CrewAI RAG task -->
