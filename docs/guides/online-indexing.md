# Online Indexing

Online indexing adds conversation turns to the knowledge graph in real time via the `qortex_ingest_message` and `qortex_ingest_tool_result` MCP tools. No LLM is needed -- chunking and embedding run locally.

## MCP Tools

### qortex_ingest_message

Index a session message into the vector layer and graph backend.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | (required) | The message content. |
| `session_id` | string | (required) | Session identifier for grouping. |
| `role` | string | `"user"` | Message role: `user`, `assistant`, `system`, `tool`. |
| `domain` | string | `"session"` | Knowledge domain for graph partitioning. |

Returns:

```json
{
  "session_id": "s1",
  "chunks": 3,
  "concepts": 3,
  "edges": 2,
  "latency_ms": 12.5
}
```

### qortex_ingest_tool_result

Index a tool's output. Same pipeline, different event type for observability.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_name` | string | (required) | Name of the tool (truncated to 64 chars). |
| `result_text` | string | (required) | The tool's text output. |
| `session_id` | string | (required) | Session identifier. |
| `domain` | string | `"session"` | Knowledge domain. |

## Pipeline

Both tools share the same `_online_index_pipeline`:

```
Text input
  │
  ▼
1. Chunking (SentenceBoundaryChunker)
  │  - Splits at sentence boundaries (regex: (?<=[.!?\n])\s+)
  │  - max_tokens=256 (~1024 chars), overlap_tokens=32 (~128 chars)
  │  - Deterministic chunk IDs: SHA-256 of "{source_id}:{index}:{text[:64]}"
  │  - Returns frozen Chunk dataclass (id, text, index)
  ▼
2. Embedding
  │  - EmbeddingModel.embed(texts) on all chunks in one batch
  │  - Uses whatever model the server initialized (sentence-transformers default)
  ▼
3. Vector index
  │  - VectorIndex.add(ids, embeddings)
  │  - IDs prefixed with session_id: "{session_id}:{chunk_id}"
  ▼
4. Graph nodes
  │  - One ConceptNode per chunk
  │  - name = first 80 chars of text
  │  - description = full chunk text
  │  - domain = provided domain parameter
  │  - source_id = "{session_id}:{role}" or "{session_id}:tool:{tool_name}"
  ▼
5. Co-occurrence edges
  │  - Consecutive chunks get REQUIRES edges (confidence=0.8)
  │  - Properties: {"origin": "co_occurrence"}
  │  - N chunks produce N-1 edges
  ▼
6. Observability events
     - GraphNodesCreated(count, domain, origin="online_index")
     - GraphEdgesCreated(count, domain, origin="co_occurrence")
     - MessageIngested or ToolResultIngested (per-call event)
```

### Empty input handling

Empty or whitespace-only text returns immediately with zero counts. No chunks, nodes, or edges are created.

### Role clamping

Message roles are clamped to an allowlist (`user`, `assistant`, `system`, `tool`, `unknown`) to prevent metric label cardinality explosion. Invalid roles map to `"unknown"`.

## Domains

Domains partition the knowledge graph. The convention for consumer integrations is `memory/{agentId}`:

```
memory/main       ← default agent
memory/work       ← work agent
project/acme      ← project-specific domain
session           ← default when no domain specified
```

The `domain` parameter controls which graph partition receives the nodes and edges. Queries can search across multiple domains, but ingest always targets a single domain.

## Pluggable chunking

The chunking strategy is swappable at runtime:

```python
from qortex.mcp.server import set_chunking_strategy
from qortex.online.chunker import ChunkingStrategy, Chunk

class MyChunker:
    def __call__(self, text, max_tokens=256, overlap_tokens=32, source_id=""):
        # Your logic here
        return [Chunk(id="...", text="...", index=0)]

set_chunking_strategy(MyChunker())
```

Any callable matching the `ChunkingStrategy` protocol works. The default `SentenceBoundaryChunker` uses a `1 token ~ 4 chars` approximation with no external dependencies.

To reset to the default:

```python
set_chunking_strategy(None)
```

## M3 Roadmap: Concept Extraction

The current implementation stores raw text chunks as ConceptNodes with co-occurrence edges. M3 will add:

- **Named entity extraction**: identify people, tools, concepts from text.
- **Typed relationships**: CAUSES, REQUIRES, RELATES_TO edges (not just co-occurrence).
- **Cross-session merging**: deduplicate concepts across sessions.
- **Temporal decay**: reduce weight of older concepts over time.

Until then, retrieval quality relies on vector similarity + PPR over co-occurrence structure + any manually ingested knowledge.

## Observability

### Events

| Event | Emitter | Fields |
|-------|---------|--------|
| `MessageIngested` | `_ingest_message_impl` | `session_id`, `role`, `domain`, `chunk_count`, `concept_count`, `edge_count`, `latency_ms` |
| `ToolResultIngested` | `_ingest_tool_result_impl` | `tool_name`, `session_id`, `domain`, `concept_count`, `edge_count`, `latency_ms` |
| `GraphNodesCreated` | `_online_index_pipeline` | `count`, `domain`, `origin` |
| `GraphEdgesCreated` | `_online_index_pipeline` | `count`, `domain`, `origin` |

### Metrics (Prometheus/Grafana)

| Metric | Type | Source Event |
|--------|------|--------------|
| `qortex_graph_nodes_created_total` | Counter | `GraphNodesCreated` |
| `qortex_graph_edges_created_total` | Counter | `GraphEdgesCreated` |

### Grafana panels

The **KG Growth** section of the `qortex-main` dashboard shows:

- **Total Nodes / Total Edges**: lifetime stat panels.
- **Nodes vs Edges over time**: time series showing growth rate.
- **By Origin**: breakdown of `online_index` vs `manifest` vs `co_occurrence`.

### Jaeger traces

When OTel is enabled (`QORTEX_OTEL_ENABLED=true`), each ingest call produces a trace tree:

```
qortex_ingest_message
  ├─ chunker (SentenceBoundaryChunker)
  ├─ vec.embed.sentence_transformer
  ├─ vec.add
  ├─ memgraph.add_node (x N)
  └─ memgraph.add_edge (x N-1)
```

## Next steps

- [Observability](observability.md) -- full metrics and tracing reference
- [Querying](querying.md) -- how retrieval uses online-indexed content
- [Consumer Integration](consumer-integration.md) -- how external systems consume projected knowledge
