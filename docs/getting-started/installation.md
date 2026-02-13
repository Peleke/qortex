# Installation

## Requirements

- Python 3.11 or higher
- pip or uv for package management

## Basic Installation

```bash
pip install qortex
```

This gives you the knowledge graph, MCP server, and vector-level tools. Consumers provide their own embeddings.

## Optional Dependencies

qortex has several optional dependency groups for different capabilities:

| Capability | Install | What you get |
|-----------|---------|-------------|
| Core + MCP tools | `pip install qortex` | Knowledge graph, MCP server, vector-level tools. Consumers provide embeddings. |
| Text-level search | `pip install qortex[vec]` | qortex embeds text with sentence-transformers. Adds ~2GB for PyTorch + model weights. |
| Persistent vectors | `pip install qortex[vec-sqlite]` | SQLite-backed vector index. Without this, vectors are in-memory only. |
| PDF ingestion | `pip install qortex[pdf]` | PDF parsing via PyMuPDF and pdfplumber. |
| LLM enrichment | `pip install qortex[llm]` | Anthropic SDK for LLM-powered rule enrichment. |
| Production graph | `pip install qortex[memgraph]` | Memgraph backend for production-scale graph operations. |
| Causal analysis | `pip install qortex[causal]` | NetworkX for DAG support and d-separation. |
| Causal inference | `pip install qortex[causal-dowhy]` | DoWhy for causal inference and refutation. |
| Bayesian causal | `pip install qortex[causal-full]` | Pyro + ChirHo for full Bayesian causal modeling. |
| PostgreSQL sources | `pip install qortex[source-postgres]` | asyncpg for ingesting from PostgreSQL databases. |
| Observability | `pip install qortex[observability]` | OpenTelemetry + Prometheus for metrics and tracing. |
| Everything | `pip install qortex[all]` | All of the above plus dev tools. |

### Which groups do I need?

- **Trying it out?** Start with `pip install qortex[vec]` for embedded text search.
- **Persistent storage?** Add `vec-sqlite` so vectors survive restarts: `pip install qortex[vec-sqlite]`.
- **Production?** Use `pip install qortex[all]` and configure Memgraph for graph operations.
- **Framework integration only?** Plain `pip install qortex` is enough if your framework (LangChain, agno) provides embeddings.

## MCP Server

The fastest way to use qortex with an AI assistant:

**Claude Code**
```bash
claude mcp add qortex -- uvx qortex mcp-serve
```

**Cursor / Windsurf**
```bash
uvx qortex mcp-serve  # add as stdio MCP server in settings
```

**Any MCP client**
```bash
pip install qortex[all] && qortex mcp-serve
```

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
