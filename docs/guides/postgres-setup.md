# PostgreSQL Setup

qortex can use PostgreSQL with pgvector for all persistent state: vector embeddings, interoception factors, and learning arm states. One database, one connection pool, three subsystems.

## Architecture

```
qortex process (stateless)
  ├─ PgVectorIndex ──────────┐
  ├─ PostgresInteroceptionStore ─┤── shared asyncpg pool ──→ PostgreSQL + pgvector
  └─ PostgresLearningStore ──────┘
```

All three stores share a single asyncpg connection pool. The pool is created once at startup and closed on shutdown.

## Prerequisites

PostgreSQL 15+ with the pgvector extension:

```bash
# Docker (recommended for local dev)
docker run -d --name qortex-postgres \
  -e POSTGRES_USER=qortex \
  -e POSTGRES_PASSWORD=qortex \
  -e POSTGRES_DB=qortex \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# Or use the docker-compose stack
cd docker && docker compose up -d postgres
```

Install the Python dependencies:

```bash
pip install "qortex[source-postgres]"
# or
pip install "asyncpg>=0.29" "pgvector>=0.3"
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QORTEX_STORE` | `sqlite` | Set to `postgres` to use Postgres for interoception + learning |
| `QORTEX_VEC` | `sqlite` | Set to `pgvector` to use PgVectorIndex |
| `PGVECTOR_DSN` | *(constructed)* | Full connection string. Overrides component vars below. |
| `PGVECTOR_HOST` | `localhost` | PostgreSQL host |
| `PGVECTOR_PORT` | `5432` | PostgreSQL port |
| `PGVECTOR_USER` | `qortex` | PostgreSQL user |
| `PGVECTOR_PASSWORD` | `qortex` | PostgreSQL password |
| `PGVECTOR_DB` | `qortex` | PostgreSQL database name |

### DSN Construction

If `PGVECTOR_DSN` is not set, qortex constructs it from the component variables:

```
postgresql://{PGVECTOR_USER}:{PGVECTOR_PASSWORD}@{PGVECTOR_HOST}:{PGVECTOR_PORT}/{PGVECTOR_DB}
```

### Minimal Setup

```bash
# Everything on localhost with defaults — just set the backend flags
QORTEX_STORE=postgres \
QORTEX_VEC=pgvector \
QORTEX_GRAPH=memgraph \
qortex serve
```

### Full Production Setup

```bash
QORTEX_STORE=postgres \
QORTEX_VEC=pgvector \
QORTEX_GRAPH=memgraph \
PGVECTOR_HOST=db.internal \
PGVECTOR_PORT=5432 \
PGVECTOR_USER=qortex_prod \
PGVECTOR_PASSWORD=secure-password \
PGVECTOR_DB=qortex \
MEMGRAPH_HOST=graph.internal \
QORTEX_OTEL_ENABLED=true \
QORTEX_PROMETHEUS_ENABLED=true \
qortex serve --host 0.0.0.0
```

## Database Schema

qortex auto-creates all tables on first access. No manual migration needed.

### Vector Embeddings (`qortex_vectors`)

```sql
CREATE TABLE qortex_vectors (
    id   TEXT PRIMARY KEY,
    vec  vector(384)          -- pgvector column (dimension matches embedding model)
);
CREATE INDEX ON qortex_vectors USING hnsw (vec vector_cosine_ops);
```

### Interoception Factors (`interoception_factors`)

```sql
CREATE TABLE interoception_factors (
    node_id    TEXT PRIMARY KEY,
    weight     DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    updated_at TIMESTAMPTZ DEFAULT now()
);
```

### Interoception Edge Buffer (`interoception_edge_buffer`)

```sql
CREATE TABLE interoception_edge_buffer (
    src_id    TEXT NOT NULL,
    tgt_id    TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    scores    JSONB NOT NULL DEFAULT '[]',
    last_seen TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (src_id, tgt_id)
);
```

### Learning Arm States (`learning_arm_states`)

```sql
CREATE TABLE learning_arm_states (
    learner_name TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    arm_id       TEXT NOT NULL,
    alpha        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    beta         DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    pulls        INTEGER NOT NULL DEFAULT 0,
    total_reward DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    last_updated TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (learner_name, context_hash, arm_id)
);
CREATE INDEX idx_learning_arm_states_context
    ON learning_arm_states(learner_name, context_hash);
```

## Connection Pool

The shared pool is configured at startup:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_size` | 2 | Minimum connections kept open |
| `max_size` | 10 | Maximum concurrent connections |
| `init` | pgvector codec | Callback run on each new connection (registers vector type) |

The pool is a singleton — all three stores share the same connections. This avoids connection exhaustion and ensures consistent codec registration.

## Verifying the Setup

```bash
# Start the server
QORTEX_STORE=postgres QORTEX_VEC=pgvector qortex serve &

# Check status
curl http://localhost:8400/v1/status | python3 -m json.tool

# Look for:
#   "vector_index": "PgVectorIndex"
#   "interoception": { "backend": "postgres" }
```

Check tables in postgres:

```bash
psql postgresql://qortex:qortex@localhost:5432/qortex -c "\dt"
```

Expected tables: `qortex_vectors`, `interoception_factors`, `interoception_edge_buffer`, `learning_arm_states`.

## Monitoring

With Prometheus enabled (`QORTEX_PROMETHEUS_ENABLED=true`), the following postgres-backed metrics are available:

- `qortex_vec_add_total` — vector inserts
- `qortex_vec_add_duration_seconds` — insert latency
- `qortex_vec_search_duration_seconds` — search latency
- `qortex_learning_selections_total` — bandit selections
- `qortex_learning_observations_total` — bandit observations
- `qortex_factor_updates_total` — interoception factor updates

All store methods are traced via OpenTelemetry (`learning.pg.get`, `learning.pg.put`, `interoception.pg.load_factors`, etc.).

## Next Steps

- [Docker Infrastructure](docker.md) — full docker-compose stack including pgvector
- [Vec Migration](vec-migration.md) — migrate from SQLite to pgvector
- [REST API](rest-api.md) — HTTP API reference
