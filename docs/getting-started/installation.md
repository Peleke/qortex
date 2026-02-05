# Installation

## Requirements

- Python 3.11 or higher
- pip or uv for package management

## Basic Installation

```bash
pip install qortex
```

## Optional Dependencies

qortex has several optional dependency groups for different use cases:

### Memgraph (Production)

```bash
pip install qortex[memgraph]
```

Adds the `neo4j` driver for connecting to Memgraph. Required for:

- Production deployments with persistent storage
- MAGE graph algorithms (Personalized PageRank)
- Cypher query support

### PDF Ingestion

```bash
pip install qortex[pdf]
```

Adds PDF parsing libraries for ingesting PDF documents.

### LLM Enrichment

```bash
pip install qortex[llm]
```

Adds the Anthropic SDK for LLM-powered rule enrichment.

### All Dependencies

```bash
pip install qortex[all]
```

Installs everything: memgraph, pdf, llm, mcp, and dev tools.

## Development Installation

For contributing to qortex:

```bash
git clone https://github.com/Peleke/qortex.git
cd qortex
pip install -e ".[dev]"
```

This installs pytest, ruff, mypy, and other development tools.

## Verify Installation

```bash
# Check CLI is available
qortex --help

# Check version
python -c "import qortex; print(qortex.__version__)"
```

## Memgraph Setup (Optional)

For production use with Memgraph:

```bash
# Start Memgraph with Docker
qortex infra up

# Verify connection
qortex infra status
```

See [Using Memgraph](../guides/memgraph.md) for detailed setup instructions.

## Next Steps

- [Quick Start](quickstart.md) - Project your first rules
- [Core Concepts](concepts.md) - Understand the data model
