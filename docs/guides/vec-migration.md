# Vector Migration

Migrate vector embeddings between storage backends without re-computing embeddings. Supports SQLite, pgvector, and in-memory backends.

## Overview

The migration tool streams vectors in batches from a source index to a destination index. It is:

- **Idempotent** — destination uses upsert, safe to re-run
- **Streaming** — processes in configurable batch sizes to control memory
- **Observable** — progress callbacks, logging, and OTel traces

## CLI

```bash
# Migrate SQLite vectors to pgvector
qortex migrate vec --from sqlite

# Dry run (report what would be migrated, don't write)
qortex migrate vec --from sqlite --dry-run

# Custom batch size
qortex migrate vec --from sqlite --batch-size 1000
```

The destination is determined by `QORTEX_VEC`:

| `--from` | Destination (`QORTEX_VEC`) | What happens |
|----------|---------------------------|--------------|
| `sqlite` | `pgvector` | SQLite → PostgreSQL (most common) |
| `sqlite` | `memory` | SQLite → in-memory |
| `memory` | `pgvector` | In-memory → PostgreSQL |

### Environment

The CLI reads `QORTEX_VEC` and `PGVECTOR_*` variables to configure the destination:

```bash
QORTEX_VEC=pgvector \
PGVECTOR_HOST=localhost \
PGVECTOR_PORT=5432 \
PGVECTOR_USER=qortex \
PGVECTOR_PASSWORD=qortex \
qortex migrate vec --from sqlite
```

## REST API

For remote or automated migrations:

```bash
# Start migration
curl -X POST http://localhost:8400/v1/admin/migrate-vec \
  -H "Content-Type: application/json" \
  -d '{"source_type": "sqlite", "batch_size": 500, "dry_run": false}'
```

**Response:**

```json
{
  "source_type": "sqlite",
  "dest_type": "pgvector",
  "batches": 3,
  "vectors_read": 1316,
  "vectors_written": 1316,
  "duration_seconds": 2.45,
  "dry_run": false
}
```

## MCP Tool

Available as `qortex_vec_migrate` in any MCP client:

```
Use qortex_vec_migrate to migrate vectors from sqlite to pgvector.
```

## Python API

```python
from qortex.vec.migrate import migrate_vec

result = await migrate_vec(
    source=sqlite_index,
    destination=pgvector_index,
    batch_size=500,
    dry_run=False,
    on_progress=lambda done, total: print(f"{done}/{total}"),
)

print(f"Migrated {result.vectors_written} vectors in {result.duration_seconds:.1f}s")
```

### MigrateResult

| Field | Type | Description |
|-------|------|-------------|
| `source_type` | str | Source backend name |
| `dest_type` | str | Destination backend name |
| `batches` | int | Number of batches processed |
| `vectors_read` | int | Vectors read from source |
| `vectors_written` | int | Vectors written to destination |
| `duration_seconds` | float | Total migration time |
| `dry_run` | bool | Whether this was a dry run |

## Next Steps

- [PostgreSQL Setup](postgres-setup.md) — configure the pgvector destination
- [CLI Reference](../reference/cli.md) — all CLI commands
