# qortex

Knowledge graph ingestion engine with HippoRAG-style cross-domain retrieval.

## Vision

qortex converts external knowledge (textbooks, documentation, blog posts) into structured, queryable knowledge graphs. It serves as the "long-term memory" layer for agent systems.

```
📚 Sources    →    🧠 qortex    →    🤖 Agents
(PDF/MD/Text)      (KG + HippoRAG)     (buildlog, OpenClaw, Claude Code)
```

## Architecture

**Neural Analogy**:
- **Domain Graphs** (Cortical regions): Dense, specialized knowledge stores
- **Hippocampus** (HippoRAG): Cross-domain integration, pattern completion
- **Ingestors** (Sensory): Transform raw input into structured form
- **Projectors** (Motor): Translate knowledge into actionable rules

**Key Design**: Ingest is separable from the KG core. The KG only knows about `IngestionManifest` - it doesn't care if input was PDF, Markdown, or custom format.

## Quick Start

```bash
# Start Memgraph
cd docker && docker-compose up -d

# Install
pip install -e ".[dev]"

# Run tests
pytest
```

## Status

🚧 **Pre-alpha** - Architecture in place, implementations stubbed.

See [Architecture Issue](https://github.com/Peleke/qortex/issues/1) for roadmap.

## License

MIT
