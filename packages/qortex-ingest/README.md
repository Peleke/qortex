# qortex-ingest

Pluggable document ingestion for [qortex](https://github.com/Peleke/qortex): extract concepts, relations, and rules from any source into a knowledge graph.

<div align="center">

<!-- Architecture: Ingestion Pipeline -->
<svg viewBox="0 0 620 420" xmlns="http://www.w3.org/2000/svg" aria-label="qortex-ingest architecture: sources flow through chunking, extraction, and assembly into an ingestion manifest">
  <style>
    .ing-bg { fill: #0d1117; }
    .ing-box { fill: #161b22; stroke: #30363d; stroke-width: 1; rx: 6; }
    .ing-box-accent { fill: #161b22; stroke: #6366f1; stroke-width: 1.5; rx: 6; filter: url(#ing-glow); }
    .ing-label { font-family: 'JetBrains Mono', monospace; font-size: 8px; fill: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .ing-title { font-family: system-ui, sans-serif; font-size: 13px; fill: #e6edf3; }
    .ing-subtitle { font-family: system-ui, sans-serif; font-size: 10px; fill: #8b949e; }
    .ing-flow { stroke: #6366f1; stroke-width: 1.2; stroke-dasharray: 4 3; fill: none; opacity: 0.5; }
    .ing-flow-anim { animation: ing-dash 2s linear infinite; }
    @keyframes ing-dash { to { stroke-dashoffset: -14; } }
    .ing-arrow { fill: #6366f1; opacity: 0.5; }
  </style>
  <defs>
    <filter id="ing-glow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="620" height="420" class="ing-bg"/>

  <!-- Sources row -->
  <rect x="20" y="20" width="120" height="50" class="ing-box"/>
  <text x="35" y="38" class="ing-label">text</text>
  <text x="35" y="55" class="ing-title">TextIngestor</text>

  <rect x="160" y="20" width="130" height="50" class="ing-box"/>
  <text x="175" y="38" class="ing-label">markdown</text>
  <text x="175" y="55" class="ing-title">MarkdownIngestor</text>

  <rect x="310" y="20" width="130" height="50" class="ing-box"/>
  <text x="325" y="38" class="ing-label">online</text>
  <text x="325" y="55" class="ing-title">SentenceChunker</text>

  <rect x="460" y="20" width="140" height="50" class="ing-box"/>
  <text x="475" y="38" class="ing-label">custom</text>
  <text x="475" y="55" class="ing-title">ChunkingStrategy</text>

  <!-- Chunking -->
  <rect x="180" y="120" width="260" height="50" class="ing-box"/>
  <text x="195" y="138" class="ing-label">phase 1 · chunking</text>
  <text x="195" y="157" class="ing-title">Format-specific → Chunk[]</text>

  <!-- Flow: sources → chunking -->
  <line x1="80" y1="70" x2="250" y2="120" class="ing-flow ing-flow-anim"/>
  <line x1="225" y1="70" x2="290" y2="120" class="ing-flow ing-flow-anim"/>
  <line x1="375" y1="70" x2="340" y2="120" class="ing-flow ing-flow-anim"/>
  <line x1="530" y1="70" x2="380" y2="120" class="ing-flow ing-flow-anim"/>

  <!-- Extraction -->
  <rect x="180" y="220" width="260" height="65" class="ing-box-accent"/>
  <text x="195" y="238" class="ing-label">phase 2 · extraction</text>
  <text x="195" y="258" class="ing-title">LLM Backend (pluggable)</text>
  <text x="195" y="274" class="ing-subtitle">Anthropic / Ollama / Stub</text>

  <!-- Flow: chunking → extraction -->
  <line x1="310" y1="170" x2="310" y2="220" class="ing-flow ing-flow-anim"/>
  <polygon points="310,218 306,210 314,210" class="ing-arrow"/>

  <!-- Manifest -->
  <rect x="180" y="335" width="260" height="60" class="ing-box-accent"/>
  <text x="195" y="353" class="ing-label">output</text>
  <text x="195" y="373" class="ing-title">IngestionManifest</text>

  <!-- Flow: extraction → manifest -->
  <line x1="310" y1="285" x2="310" y2="335" class="ing-flow ing-flow-anim"/>
  <polygon points="310,333 306,325 314,325" class="ing-arrow"/>

  <!-- Manifest contents (side annotations) -->
  <text x="460" y="345" class="ing-subtitle">ConceptNode[]</text>
  <text x="460" y="360" class="ing-subtitle">ConceptEdge[]</text>
  <text x="460" y="375" class="ing-subtitle">ExplicitRule[]</text>
  <text x="460" y="390" class="ing-subtitle">CodeExample[]</text>

  <!-- Extraction details (side annotations) -->
  <text x="20" y="235" class="ing-subtitle">Pass 1: concepts</text>
  <text x="20" y="250" class="ing-subtitle">Pass 2: reconcile</text>
  <text x="20" y="265" class="ing-subtitle">Relations (10 types)</text>
  <text x="20" y="280" class="ing-subtitle">Rules + examples</text>
</svg>

</div>

## Install

```bash
pip install qortex-ingest
```

With extraction backends:

```bash
pip install "qortex-ingest[anthropic]"   # Claude API extraction
pip install "qortex-ingest[pdf]"         # PDF support (pymupdf + pdfplumber)
pip install "qortex-ingest[all]"         # everything
```

## Quick Start

```python
from qortex.ingest import IngestionManifest
from qortex.ingest.text import TextIngestor
from qortex.ingest.backends import get_extraction_backend

# Auto-detect best available backend (Anthropic > Ollama > Stub)
backend = get_extraction_backend()

ingestor = TextIngestor(backend=backend)
manifest: IngestionManifest = ingestor.ingest(
    source_path="notes.txt",
    domain="my-project",
)

print(f"Extracted {len(manifest.concepts)} concepts, {len(manifest.edges)} relations")
```

## What It Does

**qortex-ingest** converts documents into structured knowledge graph components:

1. **Chunk** — Split source by format (paragraphs, headings, sentences)
2. **Extract** — Two-pass LLM extraction: generalizable concepts, then illustrative examples reconciled onto parents
3. **Relate** — 10 relation types: `REQUIRES`, `USES`, `REFINES`, `IMPLEMENTS`, `PART_OF`, `SIMILAR_TO`, `ALTERNATIVE_TO`, `SUPPORTS`, `CHALLENGES`, `CONTRADICTS`
4. **Assemble** — Output a single `IngestionManifest` (the universal contract)

## Ingestors

| Ingestor | Format | Chunking Strategy |
|----------|--------|-------------------|
| `TextIngestor` | Plain text | Fixed-size with configurable overlap |
| `MarkdownIngestor` | Markdown | By heading hierarchy, preserves structure |
| `SentenceBoundaryChunker` | Online/streaming | Regex sentence boundaries, SHA256 IDs |

## Pluggable Chunkers

Any callable matching `ChunkingStrategy` can replace the default:

```python
from qortex.online.chunker import Chunk

def my_chunker(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    source_id: str = "",
) -> list[Chunk]:
    # Your custom chunking logic (tiktoken, semantic, etc.)
    ...
```

## Extraction Backends

| Backend | Cost | Features |
|---------|------|----------|
| `AnthropicExtractionBackend` | ~$0.60/57KB | Full extraction: concepts, relations, rules, code examples |
| `OllamaExtractionBackend` | Free (local) | Concepts, relations, rules (no code examples) |
| `StubLLMBackend` | Free | Testing only — returns configured fixtures |

Auto-detection priority: Anthropic (if `ANTHROPIC_API_KEY` set) > Ollama (if reachable) > Stub.

## Output: IngestionManifest

The manifest is the universal contract between ingestion and the knowledge graph:

```python
@dataclass
class IngestionManifest:
    source: SourceMetadata        # origin info + stats
    domain: str                   # knowledge domain name
    concepts: list[ConceptNode]   # extracted concepts with embeddings
    edges: list[ConceptEdge]      # typed relations between concepts
    rules: list[ExplicitRule]     # best practices, warnings, principles
    code_examples: list[CodeExample]  # linked to concepts and rules
```

## Requirements

- Python 3.11+
- `qortex` (for core models — `IngestionManifest`, `ConceptNode`, etc.)
- `anthropic` (optional, for Claude extraction)
- `pymupdf` + `pdfplumber` (optional, for PDF support)

## License

MIT
