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

qortex has 13 optional dependency groups for different capabilities:

| Group | Install | What it adds |
|-------|---------|--------------|
| (core) | `pip install qortex` | Knowledge graph, MCP server, vector-level tools, `qortex-observe` logging. Consumers provide embeddings. |
| `vec` | `pip install qortex[vec]` | sentence-transformers for text embedding. Adds ~2GB for PyTorch + model weights. |
| `vec-sqlite` | `pip install qortex[vec-sqlite]` | SQLite-backed vector index via sqlite-vec. Without this, vectors are in-memory only. Includes `vec`. |
| `pdf` | `pip install qortex[pdf]` | PDF parsing via PyMuPDF and pdfplumber. |
| `llm` | `pip install qortex[llm]` | Anthropic SDK for LLM-powered extraction and rule enrichment. |
| `memgraph` | `pip install qortex[memgraph]` | neo4j driver for the Memgraph production graph backend. |
| `mcp` | `pip install qortex[mcp]` | fastmcp (also a core dependency, so this is a no-op unless pinning). |
| `causal` | `pip install qortex[causal]` | NetworkX for causal DAG support and d-separation queries. |
| `causal-dowhy` | `pip install qortex[causal-dowhy]` | NetworkX + DoWhy for causal inference and refutation. |
| `causal-full` | `pip install qortex[causal-full]` | NetworkX + Pyro + ChirHo for full Bayesian causal modeling. |
| `source-postgres` | `pip install qortex[source-postgres]` | asyncpg for connecting to and ingesting from PostgreSQL databases. |
| `observability` | `pip install qortex[observability]` | qortex-observe with OpenTelemetry exporters for distributed tracing and Prometheus metrics. |
| `dev` | `pip install qortex[dev]` | pytest, ruff, mypy, hypothesis, and other development/testing tools. |
| `all` | `pip install qortex[all]` | All of the above (except `causal-dowhy` and `causal-full`). |

### Which groups do I need?

- **Trying it out?** Start with `pip install qortex[vec]` for embedded text search.
- **Persistent storage?** Add `vec-sqlite` so vectors survive restarts: `pip install qortex[vec-sqlite]`.
- **Production?** Use `pip install qortex[all]` and configure Memgraph for graph operations.
- **Database sources?** Add `source-postgres` to connect to PostgreSQL and ingest schemas or row data.
- **Observability?** Add `observability` for OpenTelemetry traces and Prometheus metrics via qortex-observe.
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
