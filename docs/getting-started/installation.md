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
| `postgres` | `pip install qortex[postgres]` | asyncpg + pgvector for PostgreSQL-backed vec, learning, and interoception stores. |
| `source-postgres` | `pip install qortex[source-postgres]` | asyncpg for connecting to and ingesting from PostgreSQL databases. |
| `observability` | `pip install qortex[observability]` | qortex-observe with OpenTelemetry exporters for distributed tracing and Prometheus metrics. |
| `serve` | `pip install qortex[serve]` | Starlette + uvicorn for the REST API server (`qortex serve`). |
| `dev` | `pip install qortex[dev]` | pytest, ruff, mypy, hypothesis, and other development/testing tools. |
| `all` | `pip install qortex[all]` | All of the above (except `causal-dowhy` and `causal-full`). |

### Which groups do I need?

- **Trying it out?** Start with `pip install qortex[vec]` for embedded text search.
- **Persistent storage (SQLite)?** Add `vec-sqlite` so vectors survive restarts: `pip install qortex[vec-sqlite]`.
- **Persistent storage (PostgreSQL)?** Use `pip install qortex[postgres]` for pgvector-backed vectors, learning, and interoception. Requires a PostgreSQL instance with the pgvector extension.
- **REST API server?** Add `serve` to run `qortex serve`: `pip install qortex[serve]`.
- **Production?** Use `pip install qortex[all]` and configure Memgraph for graph operations.
- **PostgreSQL backends?** Add `source-postgres` for pgvector search, postgres-backed interoception, and learning stores.
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

## REST API Server (Optional)

To expose qortex as a REST API for remote clients:

```bash
pip install qortex[serve]

# Start the server
qortex serve

# With authentication
QORTEX_API_KEY=my-secret-key qortex serve

# With PostgreSQL backends
QORTEX_STORE=postgres DATABASE_URL=postgresql://user:pass@localhost/qortex qortex serve
```

Connect from Python:

```python
from qortex.http_client import HttpQortexClient

async with HttpQortexClient("http://localhost:8741", api_key="my-secret-key") as client:
    result = await client.query("error handling patterns")
```

See [CLI Reference](../reference/cli.md#serve) for all options.

## PostgreSQL Setup (Optional)

For production deployments with persistent storage across all stores:

```bash
# Install the postgres extras
pip install qortex[postgres]
```

**Requirements:**

- PostgreSQL 15+ with the [pgvector](https://github.com/pgvector/pgvector) extension
- A database with pgvector enabled:

```sql
CREATE DATABASE qortex;
\c qortex
CREATE EXTENSION IF NOT EXISTS vector;
```

**Docker (quickest):**

```bash
docker run -d --name qortex-pg \
  -e POSTGRES_DB=qortex \
  -e POSTGRES_USER=qortex \
  -e POSTGRES_PASSWORD=qortex \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Enable the extension
docker exec -it qortex-pg psql -U qortex -d qortex -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Configure qortex:**

```bash
export QORTEX_STORE=postgres
export DATABASE_URL=postgresql://qortex:qortex@localhost:5432/qortex
```

Tables are created automatically on first startup. To migrate existing SQLite vectors:

```bash
qortex migrate vec --from sqlite
```

See [Docker Infrastructure](../guides/docker.md) for the full compose stack including PostgreSQL.

## Memgraph Setup (Optional)

For production use with Memgraph:

```bash
# Start Memgraph with Docker
qortex infra up

# Verify connection
qortex infra status
```

See [Using Memgraph](../guides/memgraph.md) for detailed setup instructions.

## REST API Server

qortex includes a full HTTP API for programmatic access:

```bash
# Start the REST API server
qortex serve

# With all production backends
QORTEX_STORE=postgres QORTEX_VEC=pgvector QORTEX_GRAPH=memgraph qortex serve
```

See [REST API](../guides/rest-api.md) for endpoint documentation.

## Next Steps

- [Quick Start](quickstart.md) - Build a graph and run your first query
- [Core Concepts](concepts.md) - Understand the data model
- [PostgreSQL Setup](../guides/postgres-setup.md) - Configure postgres backends
- [REST API](../guides/rest-api.md) - HTTP API reference
