# Qortex Integration Audit Summary

> Audit date: 2026-02-13
> Scope: All qortex framework integrations across sibling repos
> Output: `/Users/peleke/Documents/Projects/qortex/contrib/integration-audit/`

## Related Documents

| File | Description |
|------|-------------|
| [stat-sheet.md](./stat-sheet.md) | Comprehensive benchmark data with all numbers |
| [pitch-deck-integration.md](./pitch-deck-integration.md) | Narrative source material for technical content |
| [reproduction-guide.md](./reproduction-guide.md) | How to reproduce every benchmark |

---

## Benchmark Results

### Quality: qortex (graph+vec) vs Vanilla (vec-only)

Tested via both CrewAI adapter (`bench_crewai_vs_vanilla.py`) and Agno adapter (`eval_agno_vs_qortex.py`). Same corpus, same queries, same results -- proving the engine is adapter-agnostic.

| Metric | qortex | Vanilla | Delta |
|--------|--------|---------|-------|
| Precision@5 | 0.55 | 0.45 | **+22%** |
| Recall@5 | 0.81 | 0.65 | **+26%** |
| nDCG@5 | 0.716 | 0.628 | **+14%** |

**Where graph dominates:** Cross-cutting queries (SSO, M2M auth) see **+50% precision, +49% recall, +71% nDCG** because the graph follows typed edges that cosine similarity can't see.

### Performance: Effectively Zero Overhead

| Method | Median | P95 | Per Query |
|--------|--------|-----|-----------|
| Vanilla (embed + cosine) | 40.34ms | 49.86ms | 5.04ms |
| Qortex (embed + vec + graph + rules) | 40.15ms | 50.48ms | 5.02ms |
| **Overhead** | **-0.5%** | +1.2% | **-0.02ms** |

### Framework Test Suite Compliance

| Framework | Their Tests | Result |
|-----------|-------------|--------|
| CrewAI | 49 knowledge tests | **46 pass** (3 = missing optional deps) |
| Agno | 12 eval tests | **12/12 pass** |
| Mastra (unit) | 20 vector tests | **20/20 pass** |
| Mastra (e2e) | 11 MCP tests | **11/11 pass** |

### MCP Transport (Cross-Language)

Mastra e2e: 29 MCP tool calls over real stdio in 3.94s. Server spawn ~400ms one-time.

---

## Integration Matrix

| Integration | Lang | Transport | Interface | Tests | Benchmarks | Merged/Shipped |
|-------------|------|-----------|-----------|-------|------------|----------------|
| langchain-qortex | Python | Direct (LocalQortexClient) | VectorStore | 47 | -- | [Own repo](https://github.com/Peleke/langchain-qortex) |
| langchain-qortex-js | TypeScript | MCP (stdio) | VectorStore | ~40 | -- | [Own repo](https://github.com/Peleke/langchain-qortex-js) |
| mastra-qortex | TypeScript | MCP (stdio) | MastraVector (9/9) | 31 (20+11) | MCP e2e | [Own repo](https://github.com/Peleke/mastra-qortex) |
| CrewAI | Python | Direct | KnowledgeStorage | 45 | P@5 +22%, R@5 +26%, nDCG +14% | contrib/ (PR ready) |
| Agno | Python | Direct | KnowledgeProtocol | 50+ | P@5 +22%, R@5 +25% | In agno repo |
| OpenClaw | TypeScript | MCP (stdio) | Learning + Memory | Yes | Dogfood scripts | Shipping in prod |
| AutoGen | Python | Direct (LocalQortexClient) | Memory (5 async) | 26 | P@5 +22%, R@5 +26%, nDCG +14% | In qortex-track-c |

## Core Adapters in qortex-track-c

All 6 adapters live in `qortex-track-c/src/qortex/adapters/`:

| File | Class | Framework | Lines |
|------|-------|-----------|-------|
| `langchain.py` | `QortexRetriever` | LangChain (retriever) | ~90 |
| `langchain_vectorstore.py` | `QortexVectorStore` | LangChain (vectorstore) | ~287 |
| `mastra.py` | `QortexVectorStore` | Mastra | ~192 |
| `crewai.py` | `QortexKnowledgeStorage` | CrewAI | ~109 |
| `agno.py` | `QortexKnowledge` | Agno | ~375 |
| `autogen.py` | `QortexMemory` | AutoGen (AG2) | ~240 |

## CrewAI Status

**Adapter:** COMPLETE (109 lines in qortex-track-c)
**Contribution:** COMPLETE in `qortex/contrib/crewai/`
- `qortex_knowledge_storage.py` (288 lines) -- Full implementation with SHA256 dedup
- `qortex_knowledge_storage_test.py` (555 lines) -- 45 tests across 7 classes
- `benchmark_crewai_vs_qortex.py` -- Needs rebuild (lost file, only `__pycache__/`)
- `bench_crewai_vs_vanilla.py` (qortex-track-c) -- Working benchmark with real numbers
- `PR_DESCRIPTION.md` -- Ready-to-submit PR spec

**LTM Fix:** `qortex/contrib/crewai-ltm-fix/` -- Bug fix for LTM `score ASC` -> `score DESC` (2-line fix with test + patch)

**Benchmark results:** P@5 +22%, R@5 +26%, nDCG +14% vs vanilla vector search.

## Agno Status

**Adapter:** COMPLETE (375 lines, full KnowledgeProtocol)
**Tests in agno repo:**
- `test_qortex_knowledge.py` (359 lines, 50+ tests)
- `test_qortex_vs_vanilla.py` (321 lines, head-to-head comparison)

**Cookbook:** `agno/cookbook/07_knowledge/protocol/qortex_knowledge.py` (132 lines)

**Eval:** `qortex-track-c/tests/eval_agno_vs_qortex.py` (445 lines) -- 20 concepts, 4 queries with ground truth, recall/precision/nDCG metrics

**Benchmark results:** P@5 +22%, R@5 +25% vs vanilla. Same as CrewAI (same engine).

## qortex-track-c Branch Status

**1 commit ahead of main:**
```
ca6dbd6 feat: numpy batch cosine + full agno KnowledgeProtocol integration
```

**Changes:** 11 files, +1671 lines
- Agno adapter expansion (375 lines)
- Eval script (445 lines)
- Performance benchmark (221 lines)
- CrewAI benchmark (new)
- Updated tests (e2e, client, dogfood, framework compat)
- Numpy batch cosine optimization
- Buildlog journal entry

**Action needed: Merge to main.**

## Feature Coverage Comparison

| Feature | langchain (Py) | langchain (JS) | mastra | crewai | agno | openclaw | autogen |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Similarity search | x | x | x | x | x | x | x |
| Graph-enhanced search | x | x | x | -- | x | x | x |
| Graph exploration | x | x | x | -- | x | -- | -- |
| Rules projection | x | x | x | -- | x | x | x |
| Feedback loop | x | x | x | x | x | x | x |
| Index management | -- | x | x | -- | -- | -- | -- |
| Vector CRUD | -- | x | x | -- | -- | -- | -- |
| Static factories | x | x | -- | -- | -- | -- | -- |
| Embedding wrappers | x | x | -- | -- | -- | -- | -- |
| Tool selection (bandits) | -- | -- | -- | -- | -- | x | -- |
| Memory ingestion | -- | -- | -- | -- | -- | x | x |
| Context injection | -- | -- | -- | -- | x | -- | x |

## Competitive Positioning

| | qortex | ChromaDB | Pinecone | mem0 | MemGPT |
|---|---|---|---|---|---|
| Structure | Knowledge graph (typed edges) | Flat vectors | Flat + metadata | Entity graph (from chat) | Tiered memory (core/archival/recall) |
| Learning | Feedback loop (runtime) | None | None | Entity extraction | LLM self-editing |
| Retrieval | Vec + graph + rules | Cosine only | Cosine + filter | Semantic search | LLM-managed paging |
| Scale | Embeddable, local-first | Medium | Massive | Per-user | Per-agent |
| Agent integration | 7 frameworks | CrewAI default | API only | AutoGen, LangGraph, CrewAI | Own agent runtime |

**Key positioning:**
- mem0 is complementary (conversational memory), not competitive. An agent uses both.
- MemGPT/Letta uses LLM calls to manage memory tiers — expensive, non-deterministic. qortex uses graph structure + feedback learning — cheap, interpretable, improves over time.
- GraphRAG (Microsoft) is heavy offline; qortex is lightweight runtime.

## Gaps & Action Items

### High Priority

1. **Merge qortex-track-c to main** -- 1671+ lines of tested work sitting unmerged
2. **Submit CrewAI PR** -- All artifacts ready in `contrib/crewai/`
3. **Submit CrewAI LTM fix** -- Trivial 2-line fix in `contrib/crewai-ltm-fix/`

### Medium Priority

4. **LangChain benchmark** -- Tutorial-style integration (not head-to-head; shows RAG pipeline)
5. **E2E tests for JS packages** -- Currently mock-only; e2e scripts exist but need running qortex server
6. **Python mastra adapter parity** -- Has `NotImplementedError` for upsert/vector_query; TS version is complete
7. **Rebuild CrewAI benchmark script** in contrib/ (lost file, only `__pycache__/`)

### Low Priority

8. **Async parity for agno** -- `aretrieve`/`aget_tools` just wrap sync calls
9. **NPM publish for langchain-qortex-js** -- CI configured with OIDC trusted publishing
10. **PyPI publish for langchain-qortex** -- CI workflow added recently
11. ~~**AutoGen adapter** -- Scoped for next sprint~~ — DONE (QortexMemory, 26/26 tests, benchmarked)

## Lines of Code Summary

| Component | Implementation | Tests | Benchmarks/Eval | Total |
|-----------|---------------|-------|-----------------|-------|
| langchain-qortex (Py) | ~350 | ~500 | -- | ~850 |
| langchain-qortex-js | ~510 | ~860 | -- | ~1,370 |
| mastra-qortex | ~510 | ~670 | -- | ~1,180 |
| CrewAI contrib | ~400 | ~555 | ~610 | ~1,565 |
| Agno adapter + eval | ~375 | ~680 | ~445 | ~1,500 |
| OpenClaw integration | ~300+ | ~200+ | -- | ~500+ |
| AutoGen adapter + bench | ~240 | ~260 | ~180 | ~680 |
| Core adapters (6) | ~1,290 | -- | -- | ~1,290 |
| **Total** | **~3,740** | **~3,725** | **~1,235** | **~8,700+** |
