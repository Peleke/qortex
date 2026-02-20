# qortex-online

Online session indexing for [qortex](https://github.com/Peleke/qortex): chunking, concept extraction, and real-time graph wiring.

## Install

```bash
pip install qortex-online                # core (chunking + extraction protocol)
pip install 'qortex-online[nlp]'         # + spaCy NER extraction
pip install 'qortex-online[all]'         # everything
```

## Quick Start

```python
from qortex.online import default_chunker, SpaCyExtractor

# Chunk conversation text
chunks = default_chunker("User said JWT tokens expire after 30 minutes. The auth module validates them.")

# Extract concepts and relations
extractor = SpaCyExtractor()
for chunk in chunks:
    result = extractor(chunk.text, domain="auth")
    for concept in result.concepts:
        print(f"  {concept.name} ({concept.confidence:.1f})")
    for rel in result.relations:
        print(f"  {rel.source_name} --{rel.relation_type}--> {rel.target_name}")
```

## What It Does

**qortex-online** handles the real-time path from conversation text to knowledge graph nodes and edges. While `qortex-ingest` handles batch document ingestion with LLM extraction, `qortex-online` handles the live session path: chunking messages as they arrive, extracting named concepts locally, and wiring them into the graph with typed relationships.

### Phase 1: Chunking

`SentenceBoundaryChunker` splits text on sentence boundaries (regex `[.!?\n]`), using a 1 token = 4 chars approximation. Each chunk gets a deterministic SHA256 ID for deduplication across sessions.

```python
from qortex.online import default_chunker, Chunk

chunks: list[Chunk] = default_chunker(
    text="Long conversation...",
    max_tokens=256,       # ~1024 chars per chunk
    overlap_tokens=32,    # 128-char overlap for context
    source_id="session-1",
)
```

### Phase 2: Concept Extraction

Three pluggable strategies, selected via `QORTEX_EXTRACTION` env var:

| Strategy | Env Value | Speed | Cost | Features |
|----------|-----------|-------|------|----------|
| `SpaCyExtractor` | `spacy` (default) | Fast | Free | NER entities + noun chunks + dep-parse relations |
| `LLMExtractor` | `llm` | Slow | API cost | Full Anthropic/Ollama extraction via qortex-ingest |
| `NullExtractor` | `none` | Instant | Free | No-op, pipeline uses raw text only |

#### SpaCy Extraction Pipeline

The default `SpaCyExtractor` runs four sub-steps, each with its own OpenTelemetry span:

1. **NLP Processing** (`extraction.spacy.nlp_process`) -- Run the spaCy `en_core_web_sm` pipeline
2. **Entity Extraction** (`extraction.spacy.extract_entities`) -- Pull NER entities (PERSON, ORG, PRODUCT, GPE, WORK_OF_ART, EVENT, FAC, LAW, LANGUAGE, NORP)
3. **Noun Chunk Extraction** (`extraction.spacy.extract_noun_chunks`) -- Collect noun phrases, filtering pronouns and determiners
4. **Deduplication** (`extraction.spacy.deduplicate`) -- Merge entities and noun chunks, preferring NER on span overlap
5. **Relation Inference** (`extraction.spacy.infer_relations`) -- Dependency-parse verb patterns and coordination

### Phase 3: Relation Inference

Relations are inferred from dependency parse patterns:

| Verb Pattern | Relation Type |
|-------------|---------------|
| use, utilize, call, invoke | `USES` |
| require, need, depend, import | `REQUIRES` |
| contain, include, have, hold | `CONTAINS` |
| implement, extend, inherit | `IMPLEMENTS` |
| refine, specialize, customize | `REFINES` |
| "X and Y" coordination | `SIMILAR_TO` |

## Pluggable Strategies

Both chunking and extraction follow the protocol pattern. Any callable matching the signature works:

```python
from qortex.online import ChunkingStrategy, ExtractionStrategy, Chunk, ExtractionResult

# Custom chunker (e.g. tiktoken-based)
class TiktokenChunker:
    def __call__(
        self, text: str, max_tokens: int = 256,
        overlap_tokens: int = 32, source_id: str = "",
    ) -> list[Chunk]:
        ...

# Custom extractor (e.g. OpenAI function calling)
class OpenAIExtractor:
    def __call__(self, text: str, domain: str = "") -> ExtractionResult:
        ...
```

## Observability

Every extraction step emits OpenTelemetry spans visible in Jaeger:

```
extraction.spacy                    [total time]
  extraction.spacy.nlp_process      [spaCy pipeline]
  extraction.spacy.extract_entities [NER pass]
  extraction.spacy.extract_noun_chunks [noun chunks]
  extraction.spacy.deduplicate      [span merging]
  extraction.spacy.infer_relations  [dep-parse]
```

When `QORTEX_OTEL_ENABLED=true`, these spans are exported alongside the parent `online_index_pipeline` span from the MCP server.

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `QORTEX_EXTRACTION` | `spacy` | Extraction strategy: `spacy`, `llm`, `none` |
| `QORTEX_OTEL_ENABLED` | `false` | Enable OpenTelemetry span export |

## Data Types

```python
@dataclass(frozen=True)
class Chunk:
    id: str       # SHA256[:16] deterministic hash
    text: str     # Chunk content
    index: int    # Position in sequence

@dataclass(frozen=True)
class ExtractedConcept:
    name: str           # e.g. "JWT Tokens"
    description: str    # One-sentence context
    confidence: float   # 0.9 (NER), 0.7 (noun chunk)

@dataclass(frozen=True)
class ExtractedRelation:
    source_name: str     # Source concept name
    target_name: str     # Target concept name
    relation_type: str   # Maps to RelationType enum
    confidence: float    # 0.5-0.8 depending on signal

@dataclass(frozen=True)
class ExtractionResult:
    concepts: list[ExtractedConcept]
    relations: list[ExtractedRelation]
```

## Requirements

- Python 3.11+
- [spaCy](https://spacy.io/) 3.7+ with `en_core_web_sm` (optional, for SpaCy extraction)
- `qortex-observe` (optional, for OpenTelemetry span tracing)
- `qortex-ingest` (optional, for LLM extraction backend)

## License

MIT
