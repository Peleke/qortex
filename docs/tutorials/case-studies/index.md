# Case Studies: Drop-In Replacements

qortex slots into existing agent frameworks as a drop-in replacement for their memory/knowledge/vector backends. Same API, zero code changes — plus graph-enhanced retrieval and feedback-driven learning that the originals don't have.

Each case study is a self-contained E2E demonstration backed by real tests in the repo.

## The Lineup

| Framework | What We Replace | Protocol | Test File |
|-----------|----------------|----------|-----------|
| [Mastra](mastra-mcp-vector-store.md) | Any `MastraVector` impl (PgVector, Chroma, Pinecone...) | MCP (cross-language) | `tests/test_e2e_mastra.py` |
| [Agno](agno-knowledge-protocol.md) | Any `KnowledgeProtocol` impl | Python protocol (duck-typed) | `tests/test_e2e_agno.py` |
| [CrewAI](crewai-knowledge-storage.md) | `KnowledgeStorage` (ChromaDB default) | Python ABC | `tests/test_framework_compat.py` |
| [LangChain](langchain-base-retriever.md) | Any `BaseRetriever` | Python ABC (Pydantic) | `tests/test_framework_compat.py` |

## The Pitch

**Before** (framework-locked):
```
Your Agent → Framework's Vector Store → Cosine Similarity → Static Results
```

**After** (qortex drop-in):
```
Your Agent → qortex Adapter → Vec + Graph + Feedback → Results That Improve
```

One import swap. No code changes. The agent doesn't know or care.

## What Makes This Different

Every framework above does cosine similarity search. That's table stakes. qortex adds:

1. **Graph-enhanced retrieval** (Level 1) — typed edges between concepts, PPR over knowledge graph
2. **Feedback-driven learning** (Level 2) — consumer outcomes adjust retrieval weights over time
3. **Cross-framework consistency** — same backend, same results, regardless of which adapter you use
4. **Cross-language via MCP** — TypeScript consumers (Mastra) talk to qortex over JSON-RPC

The consumer starts at Level 0 (vec parity) and gets Levels 1-2 for free as the system matures.
