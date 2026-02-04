# BMAD Input: qortex Status Report

> Generated: 2026-02-04
> Branch: feat/e2e-demo-2
> For PRD elicitation and architectural review

---

## Vision

`qortex` is a knowledge extraction and retrieval system that converts external sources (textbooks, documentation, blog posts) into structured, queryable knowledge graphs. It serves as the "long-term memory" layer for agent systems, complementing buildlog's "working memory" of learned patterns.

```
External Knowledge          qortex                    Agent Systems
─────────────────          ───────                    ─────────────
                    ┌─────────────────────┐
 📚 Textbooks       │  Ingestors          │
 📄 Documentation  ──▶ (PDF/MD/Text)      │
 📝 Blog posts      │         │           │         ┌─────────────┐
                    │         ▼           │   MCP   │  buildlog   │
                    │  Domain Graphs      │◀───────▶│  (rules)    │
                    │  (Cortical regions) │         └─────────────┘
                    │         │           │
                    │         ▼           │         ┌─────────────┐
                    │  Hippocampus        │   MCP   │  OpenClaw   │
                    │  (HippoRAG layer)   │◀───────▶│  (agents)   │
                    │         │           │         └─────────────┘
                    │         ▼           │
                    │  Projectors         │         ┌─────────────┐
                    │  (flat rules, etc.) │   MCP   │ Claude Code │
                    └─────────────────────┘◀───────▶│  (IDE)      │
                                                    └─────────────┘
```

---

## What's Built

### 1. Core Package (`src/qortex/`)

| Component | Status | Notes |
|-----------|--------|-------|
| **models.py** | ✅ Complete | ConceptNode, ConceptEdge, Rule, IngestionManifest, RelationType enum |
| **backend.py** | ✅ Protocol defined | GraphBackend protocol with all methods |
| **memgraph.py** | ⚠️ Implemented, untested | Full Memgraph backend with gqlalchemy, needs Docker testing |
| **hippocampus/** | ⚠️ Scaffolded | HippoRAG retrieval stubbed, BFS fallback implemented, PPR needs MAGE |
| **projectors/** | ⚠️ Scaffolded | FlatRuleProjector stubbed |
| **checkpoints/** | ⚠️ Scaffolded | CheckpointManager stubbed |

### 2. Ingest Package (`src/qortex_ingest/`)

| Component | Status | Notes |
|-----------|--------|-------|
| **llm/** | ✅ Complete | Strategy-based backend system |
| ├─ KeywordLLMBackend | ✅ Works | Always available, no deps |
| ├─ AnthropicLLMBackend | ✅ Works | Needs API key |
| ├─ OllamaLLMBackend | ✅ Works | Needs Ollama running |
| └─ Registry | ✅ Works | Auto-detection, priority-based selection |
| **strategies/** | ✅ Complete | Input/Output strategy system |
| ├─ TextInputStrategy | ✅ Works | Size-based chunking |
| ├─ MarkdownInputStrategy | ✅ Works | Heading-based chunking |
| ├─ PDFInputStrategy | ⚠️ Scaffolded | Needs PyMuPDF |
| ├─ ManifestOutputStrategy | ✅ Works | Default |
| ├─ JSONOutputStrategy | ✅ Works | Export |
| ├─ YAMLOutputStrategy | ✅ Works | buildlog-compatible |
| └─ Pipeline | ✅ Works | Composable processor |

### 3. Tests

| Test Suite | Passed | Skipped | Notes |
|------------|--------|---------|-------|
| test_models.py | 4 | 0 | Core data models |
| test_llm_backends.py | 23 | 4 | 4 skipped = no ANTHROPIC_API_KEY |
| test_strategies.py | 21 | 0 | Input/output strategies + pipeline |
| **Total** | **~48** | **4** | |

---

## What's NOT Built Yet

1. **Memgraph E2E test** - Backend implemented but needs Docker + integration test
2. **HippoRAG PPR** - Needs MAGE algorithms in Memgraph
3. **PDF ingestion** - PyMuPDF not installed
4. **MCP server** - Stubbed only
5. **Checkpoints** - Manager stubbed
6. **Cross-domain bridges** - Hippocampus needs implementation
7. **Embedded Memgraph** - Investigation not started

---

## Architectural Decisions Made

| Decision | Implementation |
|----------|----------------|
| Strategy pattern for LLM | `get_llm_backend()` with registry |
| Strategy pattern for Input | `get_input_strategy()` with auto-detection |
| Strategy pattern for Output | `get_output_strategy()` with multiple formats |
| Separable ingest layer | `qortex_ingest` is its own package |
| Memgraph-first | Primary backend, SQLite fallback planned |
| HippoRAG-inspired | Scaffolded, graceful degradation pattern |

---

## Open Architectural Questions

1. **Graph storage location**: `~/.qortex/`? Per-project? Configurable?
2. **LLM provider config**: Separate from buildlog's LLMBackend or shared?
3. **Feedback loop**: How does buildlog reward signal flow to qortex?
4. **Embedded vs Docker**: Feasibility of embedded Memgraph?
5. **Domain model**: Current labels approach vs separate graphs?

---

## Neural Analogy Architecture

| Component | Brain Region | Function |
|-----------|--------------|----------|
| Domain Graphs | Cortical regions | Dense, specialized knowledge stores |
| Hippocampus | Hippocampus | Cross-domain integration, pattern completion |
| Ingestors | Sensory cortex | Process raw input into structured form |
| Projectors | Motor cortex | Translate knowledge into actionable output |
| Checkpoints | Memory consolidation | Snapshot and restore states |

---

## Key Design: Separable Layers

### Ingest Layer (could be separate package)
```
qortex_ingest/
├── llm/           # LLM backends (swappable)
│   ├── anthropic  # Claude
│   ├── ollama     # Local models
│   └── keyword    # No-LLM fallback
└── strategies/    # Input/Output strategies
    ├── input/     # Text, Markdown, PDF, URL...
    └── output/    # Manifest, JSON, YAML, KG...
```

### KG Layer (embeddable)
```
qortex/
├── core/          # Models, backend protocol
├── hippocampus/   # HippoRAG retrieval
├── projectors/    # Rule derivation
├── checkpoints/   # State management
└── mcp/           # Agent interface
```

---

## HippoRAG Integration (Cross-Domain)

```
┌─────────────────────────────────────────────────────────────┐
│                     Global Graph                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Domain: FP_JS   │  │ Domain: SDP     │  │ Domain: ...  │ │
│  │ (dense local    │  │ (Software       │  │              │ │
│  │  graph)         │  │  Design Python) │  │              │ │
│  └────────┬────────┘  └────────┬────────┘  └──────────────┘ │
│           │                    │                             │
│           └────────┬───────────┘                             │
│                    ▼                                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    HIPPOCAMPUS                       │   │
│   │                  (HippoRAG layer)                    │   │
│   │                                                      │   │
│   │  • Sparse index over domain concepts                 │   │
│   │  • Cross-domain edges (bridges)                      │   │
│   │  • Pattern completion for retrieval                  │   │
│   │  • Personalized PageRank for relevance               │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Rule Derivation: Phase C → Phase B

**Phase C (Current)**: Explicit rules linked to concepts
```
Concept: Pure Functions
  └── Rule: "Avoid mutations in map/filter/reduce callbacks"
```

**Phase B (Future)**: Rules derived from relationship types
```
Concept: Pure Functions ──contradicts──▶ Concept: Mutable State
                              │
                              ▼
                    Rule: "When applying pure functions, avoid mutable state"
```

Edge templates enable Phase B:
```python
EDGE_RULE_TEMPLATES = {
    "contradicts": "When applying {source}, avoid {target}",
    "requires": "Before {target}, ensure {source} is satisfied",
    "refines": "{target} is a more specific form of {source}",
}
```

---

## MCP Server Tools (Planned)

```python
@tool
def qortex_query(context: str, domains: list[str] | None = None) -> list[Rule]:
    """Given context, return relevant rules via HippoRAG retrieval."""

@tool
def qortex_ingest(source_path: str, source_type: str, domain: str | None = None) -> dict:
    """Ingest source into domain. If domain=None, LLM suggests name."""

@tool
def qortex_domains() -> list[dict]:
    """List available domains with stats."""

@tool
def qortex_checkpoint(name: str, domains: list[str] | None = None) -> str:
    """Create checkpoint of current state."""

@tool
def qortex_restore(checkpoint: str) -> None:
    """Restore to named checkpoint."""
```

---

## Milestones (Original Plan)

### M1: Foundation ⚠️ In Progress
- [x] Repo setup, pyproject.toml, CI
- [x] GraphBackend protocol + Memgraph implementation
- [x] Domain model (create, query, isolate)
- [x] Basic models (Node, Edge, Concept, Rule)
- [ ] **E2E test with Docker Memgraph**

### M2: Ingestion Pipeline ✅ Complete
- [x] Ingestor protocol
- [x] Text ingestor (simplest, LLM-chunked)
- [x] Markdown ingestor
- [ ] PDF ingestor (needs PyMuPDF)
- [x] LLM extraction (concepts, relations)

### M3: Hippocampus ⚠️ Scaffolded
- [ ] Sparse index over domains
- [ ] Cross-domain bridge creation
- [x] Simple traversal retrieval (fallback)
- [ ] PPR retrieval via MAGE (full HippoRAG)

### M4: Projectors + Checkpoints ⚠️ Scaffolded
- [ ] Flat rule projector (buildlog-compatible)
- [ ] Context projector (query-based)
- [ ] Checkpoint create/restore/diff
- [ ] Auto-rollback on metric degradation

### M5: MCP Server ⚠️ Stubbed
- [ ] MCP server skeleton
- [ ] qortex_query tool
- [ ] qortex_ingest tool
- [ ] qortex_domains / checkpoint tools
- [ ] Integration test with buildlog

### M6: Embedded Investigation ❌ Not Started
- [ ] Research Memgraph embedded mode
- [ ] Prototype single-binary deployment
- [ ] Evaluate tradeoffs vs Docker

---

## Related Issues

- qortex #1: Architecture reference
- qortex #2: E2E demo (this branch)
- buildlog #87: Integrate qortex into buildlog
- buildlog #20: Domain-Specific Rule Schemas (qortex provides extraction)
- buildlog #46: Source fetching (qortex ingestors supersede this)
- buildlog #47: Ontology generation (qortex hippocampus handles this)

---

## References

- HippoRAG paper: https://arxiv.org/abs/2405.14831
- Memgraph MAGE: https://memgraph.com/docs/mage
- buildlog experiment engine: `src/buildlog/core/bandit.py`

---

## What's Next (After PRD Review)

Suggested sequence:
1. **M1 completion**: Get Memgraph E2E demo working with Docker
2. **M2**: PDF support + real LLM extraction quality testing
3. **M3**: HippoRAG with PPR (needs MAGE)
4. **M5**: MCP server for agent integration
5. **buildlog integration**: Close #87
