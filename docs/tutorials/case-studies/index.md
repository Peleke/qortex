# Case Studies: Drop-In Replacements

qortex augments existing agent frameworks by implementing their memory/knowledge/vector interfaces. Same API, zero code changes, plus graph-enhanced retrieval and feedback-driven learning layered on top of what these excellent frameworks already provide.

Each case study is a self-contained E2E demonstration backed by real tests in the repo.

## The Lineup

| Framework | What We Replace | Protocol | Test File |
|-----------|----------------|----------|-----------|
| [LangChain VectorStore](langchain-vectorstore.md) | Chroma, FAISS, Pinecone, any `VectorStore` | Python ABC | `tests/test_langchain_e2e_dogfood.py` |
| [Mastra](mastra-mcp-vector-store.md) | Any `MastraVector` impl (PgVector, Chroma, Pinecone...) | MCP (cross-language) | `tests/test_mastra_mcp_dogfood.py` |
| [Agno](agno-knowledge-protocol.md) | Any `KnowledgeProtocol` impl | Python protocol (duck-typed) | `tests/test_e2e_agno.py` |
| [CrewAI](crewai-knowledge-storage.md) | `KnowledgeStorage` (ChromaDB default) | Python ABC | `tests/test_framework_compat.py` |
| [LangChain Retriever](langchain-base-retriever.md) | Any `BaseRetriever` | Python ABC (Pydantic) | `tests/test_framework_compat.py` |

## The Pitch

**Before** (great, but vector-only):
```
Your Agent → Framework's Vector Store → Cosine Similarity → Results
```

**After** (augmented with qortex):
```
Your Agent → qortex Adapter → Vec + Graph + Feedback → Results That Improve
```

Swap one import and the agent gets graph structure and continuous learning without any other code changes.

## What Makes This Different

Every framework above provides excellent cosine similarity search. qortex builds on that foundation and adds:

1. **Graph-enhanced retrieval** (Level 1): typed edges between concepts, PPR over knowledge graph
2. **Rules auto-surfaced in results**: linked rules appear in query responses, zero consumer effort
3. **Graph exploration**: `explore(node_id)` traverses typed edges, surfaces neighbors and rules
4. **Feedback-driven learning** (Level 2): consumer outcomes adjust retrieval weights over time
5. **Cross-framework consistency**: same backend, same results, regardless of which adapter you use
6. **Cross-language via MCP**: TypeScript consumers (Mastra) talk to qortex over JSON-RPC

The consumer starts at Level 0 (vec parity) and gets Levels 1-2 for free as the system matures.
