# Online Indexing

Online indexing adds conversation turns to the knowledge graph in real time via the `qortex_ingest_message` and `qortex_ingest_tool_result` MCP tools. Concept extraction runs locally via spaCy (default) or optionally via LLM. No external API key is required for the default configuration.

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
  "concepts": 8,
  "edges": 5,
  "extracted_concepts": 5,
  "latency_ms": 42.3
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
4. Per-chunk processing (for each chunk):
  │
  │  4a. Chunk node (vec bridge)
  │  │  - ConceptNode with name = first 80 chars of text
  │  │  - description = full chunk text, domain and source_id from params
  │  │  - This node bridges vec search → graph traversal
  │  │
  │  4b. Concept extraction
  │  │  - Runs the active ExtractionStrategy on chunk text
  │  │  - Returns ExtractedConcepts (named entities, noun phrases)
  │  │  - Returns ExtractedRelations (typed relationships)
  │  │
  │  4c. Concept nodes + CONTAINS edges
  │  │  - One ConceptNode per extracted concept (ID: "{prefix}:concept:{slug}")
  │  │  - CONTAINS edge from chunk node → each concept node
  │  │  - Properties: {"origin": "extraction"}
  │  │
  │  4d. Typed relation edges
  │     - Edges between concept nodes from extraction results
  │     - Types: USES, REQUIRES, CONTAINS, IMPLEMENTS, REFINES, SIMILAR_TO
  │     - Confidence from extraction strategy (0.5-0.9)
  │
  ▼
5. Co-occurrence edges
  │  - Consecutive chunks get REQUIRES edges (confidence=0.8)
  │  - Properties: {"origin": "co_occurrence"}
  │  - N chunks produce N-1 edges
  ▼
6. Observability events
     - ConceptsExtracted (per-chunk: concept_count, relation_count, strategy, latency_ms)
     - ExtractionPipelineCompleted (aggregate: total_concepts, total_relations, strategy)
     - GraphNodesCreated(count, domain, origin="online_index")
     - GraphEdgesCreated(count, domain, origin="online_index")
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

## Concept Extraction

The pipeline extracts named concepts and typed relationships from each chunk using a pluggable `ExtractionStrategy`. This produces a richer graph than raw text nodes alone: PPR traverses from chunk nodes through CONTAINS edges to named concept nodes, then follows typed relation edges to discover related concepts.

### Extraction strategies

Set via `QORTEX_EXTRACTION` environment variable:

| Value | Strategy | Description |
|-------|----------|-------------|
| `spacy` (default) | `SpaCyExtractor` | NER + noun chunks + dependency-parse relation inference. Fast, local, no API key. |
| `llm` | `LLMExtractor` | Wraps the qortex-ingest `LLMBackend` (Anthropic or Ollama). Higher quality, requires API key or local model. |
| `none` | `NullExtractor` | No extraction. Chunk nodes use raw text[:80] as names. |

### SpaCy extractor (default)

The default strategy uses spaCy's `en_core_web_sm` model:

1. **Named entity recognition**: Extracts PERSON, ORG, PRODUCT, GPE, WORK_OF_ART, EVENT, FAC, LAW, LANGUAGE, NORP entities (confidence 0.9).
2. **Noun chunk extraction**: Extracts noun phrases not already covered by NER (confidence 0.7).
3. **Span deduplication**: Prefers NER entities over noun chunks when spans overlap.
4. **Relation inference** from dependency parse:
   - Subject-verb-object with "use/call/invoke" verbs → USES
   - Subject-verb-object with "require/need/depend/import" → REQUIRES
   - "contain/include/have" → CONTAINS
   - "implement/extend/inherit" → IMPLEMENTS
   - "refine/specialize/customize" → REFINES
   - Coordination patterns ("X and Y") → SIMILAR_TO

The spaCy model is downloaded eagerly on first use. If spaCy is not installed, the extractor returns empty results and the pipeline falls back to raw text behavior.

Install spaCy support:

```bash
pip install 'qortex[nlp]'
```

### LLM extractor (opt-in)

Wraps the existing qortex-ingest `LLMBackend`:

```bash
export QORTEX_EXTRACTION=llm
```

Calls `extract_concepts()` and `extract_relations()` on the configured backend (Anthropic or Ollama). Higher quality extraction but incurs API costs or requires a local model.

### Custom strategies

Any callable matching the `ExtractionStrategy` protocol works:

```python
from qortex.mcp.server import set_extraction_strategy
from qortex.online.extractor import ExtractionResult, ExtractedConcept

class MyExtractor:
    def __call__(self, text: str, domain: str = "") -> ExtractionResult:
        # Your logic here
        return ExtractionResult(concepts=[...], relations=[...])

set_extraction_strategy(MyExtractor(), name="custom")
```

### Graph structure

For a message like "The auth module handles JWT validation. It requires the crypto library.":

```
chunk:abc123 (name="The auth module handles JWT validation...")
  ├── CONTAINS → concept:auth_module (name="Auth Module")
  ├── CONTAINS → concept:jwt_validation (name="Jwt Validation")
  └── CONTAINS → concept:crypto_library (name="Crypto Library")

concept:auth_module ──USES──→ concept:jwt_validation
concept:auth_module ──REQUIRES──→ concept:crypto_library
```

Chunk nodes remain as the bridge between vector search and the concept graph. PPR traverses: vec search → chunk node → CONTAINS → concept nodes → typed edges → more concepts.

## Roadmap

- **Cross-session concept merging**: Deduplicate concepts across sessions.
- **Temporal decay**: Reduce weight of older concepts over time.
- **Hybrid extraction**: Combine spaCy speed with LLM precision for high-value chunks.

## Observability

### Events

| Event | Emitter | Fields |
|-------|---------|--------|
| `MessageIngested` | `_ingest_message_impl` | `session_id`, `role`, `domain`, `chunk_count`, `concept_count`, `edge_count`, `latency_ms` |
| `ToolResultIngested` | `_ingest_tool_result_impl` | `tool_name`, `session_id`, `domain`, `concept_count`, `edge_count`, `latency_ms` |
| `ConceptsExtracted` | `_online_index_pipeline` | `concept_count`, `relation_count`, `domain`, `strategy`, `latency_ms`, `chunk_index`, `source_id` |
| `ExtractionPipelineCompleted` | `_online_index_pipeline` | `total_concepts`, `total_relations`, `total_chunks`, `domain`, `strategy`, `latency_ms`, `source_id` |
| `GraphNodesCreated` | `_online_index_pipeline` | `count`, `domain`, `origin` |
| `GraphEdgesCreated` | `_online_index_pipeline` | `count`, `domain`, `origin` |

### Metrics (Prometheus/Grafana)

| Metric | Type | Source Event |
|--------|------|--------------|
| `qortex_concepts_extracted` | Counter | `ConceptsExtracted` |
| `qortex_relations_extracted` | Counter | `ConceptsExtracted` |
| `qortex_extraction_duration_seconds` | Histogram | `ConceptsExtracted` |
| `qortex_extraction_pipeline_duration_seconds` | Histogram | `ExtractionPipelineCompleted` |
| `qortex_extraction_concepts_per_chunk` | Histogram | `ConceptsExtracted` |
| `qortex_extraction_relations_per_chunk` | Histogram | `ConceptsExtracted` |
| `qortex_extractions` | Counter | `ExtractionPipelineCompleted` |
| `qortex_graph_nodes_created_total` | Counter | `GraphNodesCreated` |
| `qortex_graph_edges_created_total` | Counter | `GraphEdgesCreated` |

### Grafana panels

The **KG Growth** section of the `qortex-main` dashboard shows:

- **Total Nodes / Total Edges**: lifetime stat panels.
- **Nodes vs Edges over time**: time series showing growth rate.
- **By Origin**: breakdown of `online_index` vs `manifest` vs `co_occurrence`.

The **Concept Extraction** section shows:

- **Extractions Total / Concepts Extracted / Relations Extracted**: lifetime stat panels.
- **Concepts per Chunk (p50/p95)**: distribution of extraction density.
- **Extraction Latency per chunk (p50/p95/p99)**: per-chunk extraction time.
- **Pipeline Latency (p50/p95)**: total extraction time across all chunks.
- **Concepts by Strategy & Domain**: breakdown by extraction strategy and domain.

### Jaeger traces

When OTel is enabled (`QORTEX_OTEL_ENABLED=true`), each ingest call produces a trace tree:

```
mcp.tool.qortex_ingest_message
  └─ online_index.pipeline
       ├─ online_index.chunk
       ├─ online_index.embed
       ├─ online_index.vec_add
       ├─ online_index.add_chunk_node (x N)
       ├─ online_index.extract_chunk (x N)
       │    └─ extraction.spacy (or extraction.llm)
       │         ├─ extraction.spacy.nlp_process
       │         ├─ extraction.spacy.extract_entities
       │         ├─ extraction.spacy.extract_noun_chunks
       │         ├─ extraction.spacy.deduplicate
       │         └─ extraction.spacy.infer_relations
       ├─ online_index.add_concept_nodes (x N)
       ├─ online_index.add_relation_edges (x N)
       └─ online_index.co_occurrence_edges
```

## Next steps

- [Observability](observability.md) -- full metrics and tracing reference
- [Querying](querying.md) -- how retrieval uses online-indexed content
- [Consumer Integration](consumer-integration.md) -- how external systems consume projected knowledge
