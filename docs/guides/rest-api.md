# REST API

qortex exposes a full HTTP API for programmatic access to the knowledge graph, learning system, and vector operations.

## Starting the Server

```bash
# Default: localhost:8400
qortex serve

# Custom host/port with auto-reload
qortex serve --host 0.0.0.0 --port 9000 --reload
```

With Postgres backends:

```bash
QORTEX_STORE=postgres \
QORTEX_VEC=pgvector \
QORTEX_GRAPH=memgraph \
qortex serve
```

## Authentication

The API supports two auth modes. Both are optional — if neither `QORTEX_API_KEYS` nor `QORTEX_HMAC_SECRET` is set, all endpoints are open.

### API Key

Set comma-separated keys (SHA-256 hashed internally):

```bash
export QORTEX_API_KEYS="my-secret-key,another-key"
```

Client usage:

```bash
curl -H "Authorization: Bearer my-secret-key" http://localhost:8400/v1/status
```

```python
from qortex.http_client import HttpQortexClient

client = HttpQortexClient(
    base_url="http://localhost:8400",
    api_key="my-secret-key",
)
```

### HMAC-SHA256

For service-to-service authentication with replay protection:

```bash
export QORTEX_HMAC_SECRET="shared-secret-between-services"
export QORTEX_HMAC_MAX_AGE=300  # seconds (default)
```

The client signs each request with `X-Qortex-Timestamp` and `X-Qortex-Signature` headers.

```python
client = HttpQortexClient(
    base_url="http://localhost:8400",
    hmac_secret="shared-secret-between-services",
)
```

### Public Endpoints

`GET /v1/health` requires no authentication regardless of configuration.

## Middleware Stack

Requests pass through four middleware layers in order:

1. **CORS** — configurable via `QORTEX_CORS_ORIGINS` (default: `*`)
2. **Tracing** — creates OTel spans per request (when enabled)
3. **Logging** — structured logs with method, path, status, latency
4. **Auth** — API key or HMAC verification

---

## Endpoints

### Health & Status

#### `GET /v1/health`

Liveness check. Always returns `200`.

```json
{"status": "ok"}
```

#### `GET /v1/status`

System status including backend types, domain count, and interoception state.

```bash
curl http://localhost:8400/v1/status
```

```json
{
  "status": "ok",
  "backend": "MemgraphBackend",
  "vector_index": "PgVectorIndex",
  "vector_search": true,
  "graph_algorithms": true,
  "domain_count": 3,
  "embedding_model": "all-MiniLM-L6-v2",
  "interoception": {
    "factors": {"count": 12, "mean": 1.05, "min": 0.8, "max": 1.3},
    "buffer": {"buffered_edges": 5, "total_promoted": 42},
    "backend": "postgres"
  }
}
```

#### `GET /v1/domains`

List all knowledge domains and their sizes.

#### `GET /v1/stats`

Comprehensive graph statistics.

---

### Query & Feedback

#### `POST /v1/query`

Retrieve knowledge using hybrid vec + graph search with PPR ranking.

```bash
curl -X POST http://localhost:8400/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "context": "How does Thompson sampling balance exploration?",
    "domains": ["ml-foundations"],
    "top_k": 10,
    "min_confidence": 0.0,
    "mode": "auto"
  }'
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `context` | string | *required* | The query text |
| `domains` | string[] | all | Restrict to specific domains |
| `top_k` | int | 20 | Max results |
| `min_confidence` | float | 0.0 | Score threshold |
| `mode` | string | "auto" | `"auto"`, `"vec"`, or `"graph"` |

**Response:**

```json
{
  "items": [
    {
      "id": "ml-foundations:9d032eccbfa7",
      "content": "Thompson Sampling: Bayesian approach...",
      "score": 0.356,
      "domain": "ml-foundations",
      "metadata": {
        "vec_score": 0.563,
        "ppr_score": 0.03,
        "kg_coverage": 0.0
      }
    }
  ],
  "query_id": "b48763b7-...",
  "rules": []
}
```

#### `POST /v1/feedback`

Report which results were useful. Drives the interoception learning loop.

```bash
curl -X POST http://localhost:8400/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "b48763b7-...",
    "outcomes": {
      "ml-foundations:9d032eccbfa7": "accepted",
      "ml-foundations:172817c72e0d": "rejected"
    }
  }'
```

---

### Ingestion

#### `POST /v1/ingest`

Ingest a file from disk.

```json
{"source_path": "/path/to/doc.txt", "domain": "my-domain"}
```

#### `POST /v1/ingest/text`

Ingest raw text directly.

```bash
curl -X POST http://localhost:8400/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Thompson sampling uses Beta distributions...",
    "domain": "ml-foundations",
    "name": "sampling-notes"
  }'
```

#### `POST /v1/ingest/structured`

Ingest pre-extracted concepts and edges (bypasses LLM extraction).

```bash
curl -X POST http://localhost:8400/v1/ingest/structured \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "ml-foundations",
    "concepts": [
      {"name": "Thompson Sampling", "description": "Bayesian bandit approach"},
      {"name": "Beta Distribution", "description": "Binary reward model"}
    ],
    "edges": [
      {"source": "Thompson Sampling", "target": "Beta Distribution", "relation": "uses"}
    ]
  }'
```

#### `POST /v1/ingest/message`

Lightweight message indexing for conversation turns.

```json
{"text": "User asked about...", "session_id": "abc123", "role": "user", "domain": "conversations"}
```

---

### Graph Exploration

#### `POST /v1/explore`

Traverse the knowledge graph from a concept node.

```json
{"node_id": "ml-foundations:9d032eccbfa7", "depth": 2}
```

#### `POST /v1/rules`

Query projected rules with filtering.

```json
{"domains": ["ml-foundations"], "categories": ["constraint"], "min_confidence": 0.7}
```

---

### Learning (Thompson Sampling)

#### `POST /v1/learning/select`

Select arms via Thompson Sampling.

```bash
curl -X POST http://localhost:8400/v1/learning/select \
  -H "Content-Type: application/json" \
  -d '{
    "learner": "prompt-optimizer",
    "candidates": [
      {"id": "prompt:basic", "token_cost": 100},
      {"id": "prompt:chain-of-thought", "token_cost": 200},
      {"id": "prompt:few-shot", "token_cost": 300}
    ],
    "k": 1,
    "token_budget": 500
  }'
```

#### `POST /v1/learning/observe`

Record an outcome for a selected arm.

```json
{"learner": "prompt-optimizer", "arm_id": "prompt:chain-of-thought", "outcome": "accepted"}
```

#### `GET /v1/learning/{learner}/posteriors`

Current posterior distributions for all arms.

#### `GET /v1/learning/{learner}/metrics`

Learning metrics (convergence, selection rates, reward rates).

#### `POST /v1/learning/reset`

Reset a learner's state.

```json
{"learner": "prompt-optimizer"}
```

---

### Vector Migration

#### `POST /v1/admin/migrate-vec`

Migrate vectors between backends.

```json
{"source_type": "sqlite", "batch_size": 500, "dry_run": true}
```

---

### Data Sources

#### `POST /v1/sources/connect`

Connect a PostgreSQL database as a knowledge source.

#### `POST /v1/sources/{source_id}/sync`

Sync data from a connected source.

#### `GET /v1/sources`

List connected sources.

#### `DELETE /v1/sources/{source_id}`

Disconnect a source.

---

## HttpQortexClient

The Python HTTP client implements the `QortexClient` protocol over REST.

```python
from qortex.http_client import HttpQortexClient

async with HttpQortexClient(
    base_url="http://localhost:8400",
    api_key="my-key",
    timeout=30.0,
) as client:
    result = await client.query(
        context="How does PPR work?",
        domains=["ml-foundations"],
    )
    print(result.items)

    await client.feedback(
        query_id=result.query_id,
        outcomes={"node-id": "accepted"},
    )
```

## Next Steps

- [PostgreSQL Setup](postgres-setup.md) — configure pgvector and postgres stores
- [Docker Infrastructure](docker.md) — run the full observability stack
- [Vec Migration](vec-migration.md) — migrate vectors between backends
