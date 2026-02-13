# qortex Competitive Positioning

> Generated: 2026-02-13
> Purpose: Raw material for blog posts, pitch conversations, and positioning decisions.
> Not a marketing document. This is an honest assessment.

---

## SWOT Analysis

### Strengths

1. **Retrieval quality, benchmarked** — +22% precision, +26% recall, +14% nDCG vs vanilla vector search. Consistent across 3 adapter benchmarks (CrewAI, Agno, AutoGen). Same engine, same numbers. Not cherry-picked.

2. **Zero overhead** — Graph + rules layer adds -0.5% batch overhead. Graph traversal: 0.02ms median. Feedback recording: <0.01ms. The embedding step dominates everything. The graph is free.

3. **7 framework adapters** — CrewAI, Agno, LangChain (Py), LangChain (JS), Mastra, OpenClaw, AutoGen. Drop-in replacements that implement each framework's native interface. 8,700+ lines of tested integration code.

4. **Deterministic, auditable** — Graph traversal + rule projection = same query, same results. No LLM calls in the retrieval path. Full provenance from source material through graph to projected context.

5. **Feedback loop** — Every query can trigger accepted/rejected/ignored signals. Thompson sampling adjusts edge weights. Next retrieval automatically benefits. No competitor has this at the retrieval layer.

6. **Cross-language via MCP** — Same Python engine serves TypeScript frameworks over MCP stdio. Mastra, LangChain-JS, and OpenClaw prove this works (31/31, ~40, and production respectively).

7. **Embeddable, local-first** — No cloud dependency. No API keys for retrieval. Runs in-process or over MCP. Suitable for air-gapped, edge, and privacy-sensitive deployments.

8. **Knowledge structure** — Typed edges (10 relation types), explicit rules, domain partitioning. Cross-cutting queries see +50% precision because the graph follows relationships cosine can't.

### Weaknesses

1. **No cloud offering** — Local-only. No managed service. No hosted dashboard. Enterprise buyers may need this.

2. **No entity extraction from conversation** — qortex ingests structured knowledge. It doesn't watch chat and extract entities automatically (mem0's core feature).

3. **Small community** — Early-stage. No Discord. No 21K GitHub stars. Integration quality is high but visibility is low.

4. **No agent runtime** — qortex is a memory layer, not an orchestration platform. It needs a framework to be useful.

5. **Ingestion requires structure** — Best results come from structured ingestion (concepts, edges, rules). Unstructured text ingestion exists but the graph advantage depends on having typed relationships.

6. **No conversation state management** — Not designed for multi-turn conversation memory, session persistence, or persona management.

### Opportunities

1. **Complementary positioning** — Use both: qortex for domain knowledge, mem0 for user preferences, MemGPT/Letta for conversation state. The market is big enough.

2. **Enterprise structured knowledge** — The sweet spot: organizations with domain rules, compliance requirements, institutional knowledge that needs to be injected into agents deterministically.

3. **Letta archival memory replacement** — Letta's archival memory is flat vector search. Their architecture is pluggable. qortex could slot in as the archival backend, adding +22% precision with zero LLM cost.

4. **AutoGen ecosystem** — We now have the 2nd Memory implementation after ChromaDB. AutoGen is growing. Being there early matters.

5. **Feedback loop as differentiation** — No competitor has retrieval that learns from outcomes. This compounds over time. The longer you use it, the better it gets.

6. **Open-source credibility** — 8,700+ lines of integration code across 7 frameworks. Benchmarks are reproducible. Numbers are real. This builds trust.

7. **CrewAI upstream contribution** — PR ready for CrewAI's knowledge system. If accepted, qortex becomes a first-class memory option in the most popular agent framework. Plus the LTM score-ordering bug fix.

8. **Mastra / Vercel AI ecosystem** — TypeScript-native via MCP. Mastra is Vercel-adjacent and growing fast. Being the structured memory option for the JS/TS agent ecosystem is wide open.

### Threats

1. **mem0 adds graph features** — mem0 already has entity extraction into a graph. If they add typed relationships and rule projection, the gap narrows.

2. **ChromaDB adds learning** — If ChromaDB adds feedback-based reranking, the overhead story becomes less compelling (though graph traversal would still be different).

3. **Frameworks build their own memory** — CrewAI, AutoGen, etc. could build graph-enhanced memory natively. The adapter model assumes frameworks want pluggable backends.

4. **MemGPT/Letta absorbs the use case** — If Letta's self-editing memory becomes reliable enough with cheaper models, the deterministic approach may seem less necessary.

5. **Vector databases add graph layers** — Weaviate, Qdrant, etc. could add graph-enhanced retrieval. They have the distribution and community.

6. **LangChain/LangGraph builds deeper memory primitives** — LangGraph already has checkpointing and state management. If LangChain builds a graph-enhanced retrieval layer natively, the langchain-qortex adapter becomes less necessary.

---

## Framework Landscape

### How Each Framework Handles Memory Today (and Where qortex Fits)

#### CrewAI

**Their approach:** ChromaDB-backed `KnowledgeStorage`. Flat vector search. No graph. No feedback. LTM (Long-Term Memory) exists but has a score-ordering bug (`ASC` instead of `DESC` — we filed a fix).

**qortex fit:** Drop-in replacement. `QortexKnowledgeStorage` implements `KnowledgeStorage` exactly. 46/49 of their own tests pass. Same interface, +22% precision.

**Integration depth:** 109-line adapter + 288-line contrib version + 555-line test suite + benchmark. PR ready.

**Their moat:** Default choice. Every CrewAI tutorial uses ChromaDB. Inertia is real.

**Our angle:** Better numbers, zero migration effort. The upgrade is one import statement. The LTM bug fix also demonstrates we know their codebase.

#### Agno

**Their approach:** `KnowledgeProtocol` — a 5-method interface (`retrieve`, `build_context`, `get_tools`, `aretrieve`, `aget_tools`). Ships with vector-based knowledge classes (PDF, CSV, URL, etc.). No built-in graph layer.

**qortex fit:** `QortexKnowledge` implements `KnowledgeProtocol` fully (375 lines). 12/12 eval tests pass. Includes `build_context` which injects knowledge + rules into agent context automatically.

**Integration depth:** Full adapter in qortex-track-c + tests in agno repo + cookbook example. Benchmark: same +22% P@5.

**Their moat:** Clean protocol design makes it easy for others to implement too. Good docs. Growing community.

**Our angle:** We're one of the first non-trivial `KnowledgeProtocol` implementations. The eval script (`eval_agno_vs_qortex.py`) is also a good template for anyone benchmarking knowledge backends.

#### AutoGen (AG2)

**Their approach:** `Memory` ABC in `autogen-core` with 5 async methods (`update_context`, `query`, `add`, `clear`, `close`). Ships with `ChromaDBVectorMemory` (~460 lines) and `ListMemory`. Component serialization via Pydantic.

**qortex fit:** `QortexMemory` implements all 5 methods (240 lines). 26/26 tests pass. Benchmark: same +22% P@5. Async wrapping via `asyncio.to_thread()` for the sync client. No autogen dependency required for basic usage — falls back to compatible duck-typed returns when autogen-core isn't installed.

**Integration depth:** Adapter + 26 tests (18 unit, 8 integration) + benchmark. 2nd Memory implementation in the ecosystem after ChromaDB.

**Their moat:** Microsoft backing. AutoGen is where enterprise agent development happens.

**Our angle:** Being the 2nd Memory option matters. When people look for alternatives to ChromaDB in AutoGen, we're there. The async interface is also cleaner than CrewAI's sync-only approach.

#### LangChain (Python)

**Their approach:** `VectorStore` ABC — the most mature and widely-used retrieval interface. `BaseRetriever` for search-only. Dozens of vector store integrations (Chroma, Pinecone, Qdrant, FAISS, pgvector, etc.). LangGraph adds stateful workflows with checkpointing.

**qortex fit:** Two adapters:
- `QortexRetriever` (BaseRetriever, ~90 lines) — search-only, lightweight
- `QortexVectorStore` (VectorStore, ~287 lines) — full integration with `similarity_search`, `add_texts`, `from_texts`, `as_retriever()`, graph exploration, rules, feedback

Both in qortex-track-c. Separate repo (`langchain-qortex`) with 47 tests for the full VectorStore.

**Integration depth:** The VectorStore adapter is the deepest integration — it supports static factories (`from_texts`, `from_documents`), embedding wrappers, and the full retriever pipeline.

**Their moat:** Massive ecosystem. Every RAG tutorial uses LangChain. Hundreds of integrations. LangGraph is becoming the standard for stateful agents.

**Our angle:** We're one of many VectorStore backends, but we're the only one with a graph layer, rules, and feedback. The `from_texts` factory makes it zero-config — same DX as Chroma but with graph enhancement. LangGraph's state management and qortex's knowledge retrieval are complementary, not competitive.

#### LangChain.js (TypeScript)

**Their approach:** Same VectorStore interface, TypeScript. Growing but smaller ecosystem than Python.

**qortex fit:** `langchain-qortex-js` — VectorStore over MCP stdio. Same Python engine, TypeScript interface. ~40 tests.

**Integration depth:** Full VectorStore implementation via MCP transport. Proves cross-language works.

**Their moat:** JavaScript/TypeScript is where web agents live. Vercel AI SDK, Next.js, etc.

**Our angle:** Cross-language without a rewrite. Same knowledge graph, TypeScript interface. This is rare — most Python ML tools don't have JS equivalents.

#### Mastra

**Their approach:** `MastraVector` — 9 abstract methods for vector operations (createIndex, listIndexes, describeIndex, deleteIndex, upsert, query, updateVector, deleteVector, deleteVectors). TypeScript-native. Vercel AI SDK ecosystem.

**qortex fit:** `mastra-qortex` — full `MastraVector` implementation over MCP stdio. Own repo. 31/31 tests (20 unit + 11 e2e). 29 MCP tool calls in 3.94s.

**Integration depth:** The most complete MCP integration. Exercises all 9 MastraVector methods + dimension validation + full lifecycle. The e2e tests spawn a real qortex MCP server.

**Their moat:** Growing fast in the TypeScript agent space. Vercel adjacent. Good DX.

**Our angle:** Mastra doesn't have built-in graph memory. We fill that gap completely. The MCP transport means zero Mastra-side dependencies — it just talks to qortex over stdio.

#### OpenClaw

**Their approach:** Custom agent runtime with learning + memory over MCP. Uses qortex on every agent turn in production.

**qortex fit:** This IS the production deployment. OpenClaw uses qortex's MCP server for knowledge retrieval, tool selection (bandits), and memory ingestion. It's dogfood.

**Integration depth:** ~500+ lines. Production use. Every agent turn hits qortex.

**Our angle:** This is the "we use it ourselves" story. Not a demo integration — a shipping product where qortex is the memory layer.

---

## Memory Competitor Landscape

### vs ChromaDB

| | qortex | ChromaDB |
|---|---|---|
| **Retrieval** | Vec + graph PPR + rules projection | Cosine similarity only |
| **Quality** | +22% precision, +26% recall (benchmarked) | Baseline |
| **Learning** | Feedback loop adjusts edge weights | Static index |
| **Structure** | Knowledge graph with typed edges | Flat vectors + metadata |
| **Overhead** | -0.5% batch overhead | Baseline |
| **Framework integration** | Drop-in replacement (46/49 CrewAI tests pass) | Native default in CrewAI, AutoGen |

**Positioning:** Same interface, better retrieval. The upgrade path is one import statement.

### vs Pinecone / Qdrant / Weaviate

| | qortex | Managed vector DBs |
|---|---|---|
| **Scale** | Embeddable, local-first | Massive cloud scale |
| **Architecture** | Graph layer ON TOP of vectors | Vector storage |
| **Integration** | Could use Pinecone as vec backend with qortex graph on top | Direct API |
| **Use case** | Agent memory with structure | General vector search |

**Positioning:** Not competing on scale. Complementary layer. You could use both — qortex's graph sits on top of any vector backend.

### vs mem0

| | qortex | mem0 |
|---|---|---|
| **Core purpose** | Structured domain knowledge + rules | Conversational memory + entity extraction |
| **Data model** | Knowledge graph (typed edges, explicit rules) | Entity graph (extracted from conversation) |
| **Learning signal** | Explicit feedback (accepted/rejected) | Implicit (entity recurrence in chat) |
| **Ingestion** | Structured data (concepts, edges, rules) | Unstructured chat messages |
| **Retrieval** | Vec + graph PPR + rules projection | Vec + entity matching |
| **Transport** | Direct + MCP (cross-language) | Direct Python API |
| **Framework support** | 7 (CrewAI, Agno, Mastra, LangChain x2, OpenClaw, AutoGen) | AutoGen (native), LangGraph, CrewAI |
| **Scale target** | Embeddable, local-first | Cloud + local |
| **Open source** | Yes | Yes (with cloud offering) |

**Positioning:** They're NOT competitors. An agent uses mem0 to remember "user likes dark mode" and qortex to know "OAuth2 requires PKCE for mobile clients." Different data, different signals, different retrieval paths.

- mem0 = "what users said" (conversational memory)
- qortex = "what the system knows" (domain knowledge)

### vs MemGPT / Letta

| | qortex | MemGPT/Letta |
|---|---|---|
| **Architecture** | Memory backend (plugs into any framework) | Agent runtime (IS the framework) |
| **Memory management** | Deterministic graph traversal | LLM self-editing (stochastic) |
| **Overhead** | Zero LLM calls for retrieval | ~16,900 tokens per memory management cycle |
| **Retrieval quality** | +22% P@5 vs vanilla (benchmarked) | No published retrieval benchmarks |
| **Archival memory** | Graph-enhanced (typed edges, rules, PPR) | Flat vector similarity (pgvector/ChromaDB) |
| **Conversation memory** | Not designed for this | Purpose-built (core/recall/archival tiers) |
| **Agent persona** | Not designed for this | Self-editing persona blocks |
| **Framework coverage** | 7 frameworks as drop-in backend | Is its own framework |
| **Cost model** | Zero LLM cost for retrieval | LLM calls for every memory operation |
| **Provenance** | Full thread from source to context | None (LLM-written blocks) |
| **Community** | Early-stage | 21K stars, $10M funding, 11K Discord |
| **Reliability** | Deterministic, same input = same output | Model-dependent; ~90% stacktrace rate reported |

**Positioning:** Different layers of the stack entirely.

- Letta = agent runtime (orchestration, conversation state, persona)
- qortex = knowledge memory layer (structured retrieval, domain knowledge, rules, feedback)

**The integration opportunity:** Letta's archival memory is explicitly swappable. Their architecture is tool-based and pluggable. qortex could replace Letta's flat vector archival store with graph-enhanced retrieval — giving Letta agents +22% precision at zero additional LLM cost.

**The cost narrative:** Letta uses LLM calls to decide what to remember. qortex uses graph structure and explicit feedback. One costs tokens on every interaction. The other costs 0.02ms of graph traversal. For cost-sensitive deployments, this matters enormously.

### vs Microsoft GraphRAG

| | qortex | Microsoft GraphRAG |
|---|---|---|
| **Graph construction** | Structured ingestion + feedback learning | Heavy offline LLM pipeline (entity extraction, community detection) |
| **Runtime cost** | Zero LLM calls for retrieval | LLM-intensive offline processing |
| **Use case** | Agent retrieval + learning | Corpus summarization + global queries |
| **Scale** | Embeddable, local-first | Large corpus processing |
| **Feedback** | Runtime feedback loop (Thompson sampling) | Static after construction |

**Positioning:** GraphRAG is for understanding a large corpus offline. qortex is for teaching agents domain knowledge that improves over time. Different problems.

### vs LangGraph Memory / Checkpointing

| | qortex | LangGraph |
|---|---|---|
| **Memory type** | Knowledge retrieval (domain facts, rules) | Workflow state (checkpoints, branching, human-in-the-loop) |
| **Persistence** | Knowledge graph (concepts, edges, rules) | Thread-level state snapshots |
| **Retrieval** | Semantic search + graph traversal + rule projection | State recall by thread/checkpoint ID |
| **Learning** | Feedback loop on retrieval quality | No retrieval learning |
| **Scope** | Cross-session domain knowledge | Per-workflow execution state |

**Positioning:** Complementary. LangGraph manages workflow state ("where am I in this process?"). qortex manages domain knowledge ("what do I need to know to do this step?"). A LangGraph workflow can use qortex as its retrieval backend via the LangChain VectorStore adapter.

---

## Full Landscape Summary

| System | What it stores | How it learns | Our relationship |
|--------|---------------|---------------|------------------|
| **ChromaDB** | Vectors | Doesn't | We replace it (same interface, +22% precision) |
| **Pinecone/Qdrant/Weaviate** | Vectors at scale | Doesn't | We sit on top (graph layer) |
| **mem0** | User preferences + entities | Entity extraction from chat | Complementary (different data) |
| **MemGPT/Letta** | Agent state + conversation | LLM self-editing | Complementary (different layer). Could replace their archival store. |
| **Microsoft GraphRAG** | Corpus summaries | Offline LLM extraction | Different problem (offline vs runtime) |
| **LangGraph** | Workflow state | Checkpoints | Complementary. We're the retrieval backend. |
| **CrewAI Knowledge** | Task context | Doesn't | We're a better backend (PR ready) |
| **Agno Knowledge** | Agent context | Doesn't | We implement their protocol |
| **AutoGen Memory** | Agent context | Doesn't (ChromaDB default) | We're the 2nd Memory impl |
| **Mastra Vector** | Vector operations | Doesn't | We implement MastraVector over MCP |

---

## Key Narratives

### For technical audiences
> "We added a knowledge graph, typed relationships, domain rules, AND a feedback loop... and it costs less than nothing. -0.5% batch overhead. The graph is free."

### For positioning conversations
> "mem0 remembers what users say. MemGPT manages what agents think. qortex teaches agents what they need to know. Different layers, complementary."

### For enterprise
> "When your agent needs to know that OAuth2 requires PKCE for mobile clients — not because someone said it in chat, but because it's a security rule — that's qortex. Deterministic, auditable, provenance-tracked."

### For cost-conscious buyers
> "MemGPT uses LLM calls to manage memory. Every interaction costs tokens. qortex uses graph structure and feedback. Every interaction costs 0.02ms of graph traversal. At scale, this is the difference between viable and not."

### For framework maintainers
> "We implement your interface. Your tests pass. The upgrade path for your users is one import statement. We don't compete with your framework — we make your memory layer better."
