# Case Study: Replacing Mastra's Vector Store via MCP

> **Status**: E2E proven — `tests/test_e2e_mastra.py` (13 tests, real embeddings)
>
> **The hook**: Mastra is TypeScript. qortex is Python. MCP bridges the gap. One config change gives any Mastra app graph-enhanced retrieval with a feedback loop — something Mastra's own GraphRAG can't do.

## What Mastra Has

Mastra provides `MastraVector`, an abstract class with 22 implementations (PgVector, Chroma, Pinecone, Qdrant, etc.). Their `GraphRAG` builds an in-memory cosine similarity graph at query time — O(N²), no persistence, no learning.

| MastraVector Method | qortex MCP Equivalent |
|--------------------|----------------------|
| `query(indexName, queryVector, topK)` | `qortex_query(context, domains, top_k)` |
| `listIndexes()` | `qortex_domains()` |
| `describeIndex(name)` | `qortex_domains()` + filter |
| `upsert(indexName, vectors)` | `qortex_ingest(source_path, domain)` |
| `createIndex(name, dimension)` | Auto-created on ingest |
| — (nothing) | `qortex_feedback(query_id, outcomes)` |

The last row is the point. Mastra has no feedback mechanism.

## The Swap

<!-- TODO: Show the TypeScript side — @qortex/mastra MastraVector implementation -->
<!-- TODO: Show the MCP server config in claude_desktop_config.json or equivalent -->
<!-- TODO: Before/after code comparison -->

### Python Side (proven)

```python
from qortex.adapters.mastra import QortexVectorStore
from qortex.client import LocalQortexClient

client = LocalQortexClient(vector_index, backend, embedding_model)
store = QortexVectorStore(client=client)

# Exact same API as any MastraVector
results = store.query(index_name="security", query_text="OAuth2 auth", top_k=3)
# → [{"id": "...", "score": 0.94, "metadata": {...}, "document": "..."}]

# The upgrade Mastra can't do
store.feedback({results[0]["id"]: "accepted"})
```

### MCP Server (proven)

The MCP server exposes the same operations over JSON-RPC. A TypeScript Mastra client calls:

```json
{"method": "tools/call", "params": {"name": "qortex_query", "arguments": {"context": "OAuth2 auth", "domains": ["security"], "top_k": 3}}}
```

Response maps 1:1 to Mastra's `QueryResult[]` shape.

## What We Proved

**Real embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions). Not mocks.

**Real ingestion**: Text document → concept extraction → embedding generation → vector index + graph storage.

**Real semantic search**: Query "How does OAuth2 work?" → top result contains OAuth2 content (verified).

**Real feedback loop**: Query → accept/reject outcomes → recorded for future PPR weight adjustment.

**JSON serializable**: Every MCP response round-trips through `json.dumps()`/`json.loads()`.

## Where qortex Is Strictly Better

| Dimension | Mastra GraphRAG | qortex |
|-----------|----------------|--------|
| Graph edges | Cosine sim threshold (one type) | Typed edges (REQUIRES, REFINES...) + cosine fallback |
| Graph construction | O(N²) all-pairs, in-memory, per query | Persistent KG + online gen for gaps |
| Walk algorithm | Monte Carlo random walk (stochastic) | PPR power iteration (deterministic) |
| Persistence | None — rebuilt every query | SqliteVec + Memgraph |
| Learning | None | Teleportation factors from feedback |
| Cross-session | Nothing carries over | Graph + factors accumulate |

## Next Steps

<!-- TODO: Build @qortex/mastra TypeScript package -->
<!-- TODO: Publish to npm -->
<!-- TODO: Record demo video: Mastra app → swap config → show improved retrieval -->
<!-- TODO: Benchmark: qortex PPR vs Mastra Monte Carlo random walks on same dataset -->
