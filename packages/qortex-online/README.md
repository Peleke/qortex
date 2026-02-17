# qortex-online

Online session indexing for [qortex](https://github.com/Peleke/qortex): chunking, concept extraction, and real-time graph wiring.

<div align="center">

<!-- Architecture: Online Extraction Pipeline -->
<svg viewBox="0 0 620 480" xmlns="http://www.w3.org/2000/svg" aria-label="qortex-online architecture: conversation text flows through chunking, concept extraction, and relation inference into the knowledge graph">
  <style>
    .onl-bg { fill: #0d1117; }
    .onl-box { fill: #161b22; stroke: #30363d; stroke-width: 1; rx: 6; }
    .onl-box-accent { fill: #161b22; stroke: #6366f1; stroke-width: 1.5; rx: 6; filter: url(#onl-glow); }
    .onl-label { font-family: 'JetBrains Mono', monospace; font-size: 8px; fill: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .onl-title { font-family: system-ui, sans-serif; font-size: 13px; fill: #e6edf3; }
    .onl-subtitle { font-family: system-ui, sans-serif; font-size: 10px; fill: #8b949e; }
    .onl-flow { stroke: #6366f1; stroke-width: 1.2; stroke-dasharray: 4 3; fill: none; opacity: 0.5; }
    .onl-flow-anim { animation: onl-dash 2s linear infinite; }
    @keyframes onl-dash { to { stroke-dashoffset: -14; } }
    .onl-arrow { fill: #6366f1; opacity: 0.5; }
  </style>
  <defs>
    <filter id="onl-glow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="620" height="480" class="onl-bg"/>

  <!-- Input -->
  <rect x="180" y="20" width="260" height="50" class="onl-box"/>
  <text x="195" y="38" class="onl-label">input</text>
  <text x="195" y="55" class="onl-title">Conversation text (MCP session)</text>

  <!-- Chunking -->
  <rect x="180" y="110" width="260" height="55" class="onl-box"/>
  <text x="195" y="128" class="onl-label">phase 1 · chunking</text>
  <text x="195" y="148" class="onl-title">SentenceBoundaryChunker</text>

  <!-- Flow: input → chunking -->
  <line x1="310" y1="70" x2="310" y2="110" class="onl-flow onl-flow-anim"/>
  <polygon points="310,108 306,100 314,100" class="onl-arrow"/>

  <!-- Chunking annotations -->
  <text x="460" y="125" class="onl-subtitle">256 tokens / chunk</text>
  <text x="460" y="140" class="onl-subtitle">32-token overlap</text>
  <text x="460" y="155" class="onl-subtitle">SHA256 deterministic IDs</text>

  <!-- Extraction (accented) -->
  <rect x="180" y="205" width="260" height="65" class="onl-box-accent"/>
  <text x="195" y="223" class="onl-label">phase 2 · extraction</text>
  <text x="195" y="243" class="onl-title">ExtractionStrategy (pluggable)</text>
  <text x="195" y="259" class="onl-subtitle">SpaCy / LLM / Null</text>

  <!-- Flow: chunking → extraction -->
  <line x1="310" y1="165" x2="310" y2="205" class="onl-flow onl-flow-anim"/>
  <polygon points="310,203 306,195 314,195" class="onl-arrow"/>

  <!-- Extractor row -->
  <rect x="20" y="205" width="140" height="65" class="onl-box"/>
  <text x="35" y="223" class="onl-label">spacy (default)</text>
  <text x="35" y="243" class="onl-title">NER + noun chunks</text>
  <text x="35" y="259" class="onl-subtitle">dep-parse relations</text>

  <rect x="460" y="205" width="140" height="65" class="onl-box"/>
  <text x="475" y="223" class="onl-label">llm (opt-in)</text>
  <text x="475" y="243" class="onl-title">Anthropic / Ollama</text>
  <text x="475" y="259" class="onl-subtitle">via qortex-ingest</text>

  <!-- Relation Inference -->
  <rect x="180" y="310" width="260" height="55" class="onl-box"/>
  <text x="195" y="328" class="onl-label">phase 3 · relation inference</text>
  <text x="195" y="348" class="onl-title">Verb patterns + coordination</text>

  <!-- Flow: extraction → relations -->
  <line x1="310" y1="270" x2="310" y2="310" class="onl-flow onl-flow-anim"/>
  <polygon points="310,308 306,300 314,300" class="onl-arrow"/>

  <!-- Relation annotations -->
  <text x="20" y="325" class="onl-subtitle">USES, REQUIRES, CONTAINS</text>
  <text x="20" y="340" class="onl-subtitle">IMPLEMENTS, REFINES</text>
  <text x="20" y="355" class="onl-subtitle">SIMILAR_TO (coordination)</text>

  <!-- Output: Graph -->
  <rect x="180" y="405" width="260" height="55" class="onl-box-accent"/>
  <text x="195" y="423" class="onl-label">output</text>
  <text x="195" y="443" class="onl-title">ConceptNode[] + ConceptEdge[]</text>

  <!-- Flow: relations → graph -->
  <line x1="310" y1="365" x2="310" y2="405" class="onl-flow onl-flow-anim"/>
  <polygon points="310,403 306,395 314,395" class="onl-arrow"/>

  <!-- Output annotations -->
  <text x="460" y="420" class="onl-subtitle">CONTAINS (chunk → concept)</text>
  <text x="460" y="435" class="onl-subtitle">Typed edges (dep-parse)</text>
  <text x="460" y="450" class="onl-subtitle">SIMILAR_TO (co-occurrence)</text>
</svg>

</div>

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
