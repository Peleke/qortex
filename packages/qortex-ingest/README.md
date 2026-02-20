# qortex-ingest

Pluggable document ingestion for [qortex](https://github.com/Peleke/qortex): extract concepts, relations, and rules from any source into a knowledge graph.

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
