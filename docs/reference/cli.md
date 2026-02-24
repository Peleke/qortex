# CLI Reference

The qortex CLI manages knowledge graphs, infrastructure, projections, and data migrations.

## Global Options

```bash
qortex --help          # Show all commands
qortex <command> --help  # Show command help
```

## Command Groups

| Group | Description |
|-------|-------------|
| `serve` | Start the REST API server (top-level command) |
| `infra` | Infrastructure management (Memgraph) |
| `ingest` | Content ingestion |
| `migrate` | Data migration between backends |
| `project` | Rule projection |
| `inspect` | Graph inspection |
| `viz` | Visualization and queries |
| `interop` | Consumer interop protocol |
| `prune` | Edge pruning and analysis |
| `serve` | Start the REST API server (top-level command) |
| `migrate` | Data migration tools |
| `mcp-serve` | Start the MCP server (top-level command) |

---

## serve

### `qortex serve`

Start the REST API server. Exposes the full HTTP API for queries, ingestion, learning, and administration.

```bash
# Default: localhost:8400
qortex serve

# Custom host and port
qortex serve --host 0.0.0.0 --port 9000

# With auto-reload for development
qortex serve --host 0.0.0.0 --port 8400 --reload
```

Options:
- `--host`: Bind address (default: `127.0.0.1`)
- `--port`: Listen port (default: `8400`)
- `--reload`: Enable auto-reload on code changes (development only)

With postgres backends:

```bash
QORTEX_STORE=postgres \
QORTEX_VEC=pgvector \
QORTEX_GRAPH=memgraph \
qortex serve --host 0.0.0.0
```

See [REST API](../guides/rest-api.md) for endpoint documentation and [PostgreSQL Setup](../guides/postgres-setup.md) for backend configuration.

---

## migrate

Data migration between storage backends.

### `qortex migrate vec`

Migrate vector embeddings from one backend to another without re-computing embeddings.

```bash
# Migrate SQLite vectors to pgvector (destination from QORTEX_VEC)
qortex migrate vec --from sqlite

# Dry run — report counts without writing
qortex migrate vec --from sqlite --dry-run

# Custom batch size
qortex migrate vec --from sqlite --batch-size 1000
```

Options:
- `--from`: Source backend type (`sqlite`, `memory`) — **required**
- `--batch-size`: Vectors per batch (default: `500`)
- `--dry-run`: Report what would be migrated without writing

The destination is determined by the `QORTEX_VEC` environment variable:

| `--from` | `QORTEX_VEC` | Migration |
|----------|-------------|-----------|
| `sqlite` | `pgvector` | SQLite → PostgreSQL (most common) |
| `sqlite` | `memory` | SQLite → in-memory |
| `memory` | `pgvector` | In-memory → PostgreSQL |

See [Vec Migration](../guides/vec-migration.md) for full documentation.

---

## infra

Manage Memgraph infrastructure.

### `qortex infra up`

Start Memgraph and Lab containers.

```bash
qortex infra up
```

Options:
- `--detach / -d`: Run in background (default: true)

### `qortex infra down`

Stop Memgraph containers.

```bash
qortex infra down
```

### `qortex infra status`

Check Memgraph connection status.

```bash
qortex infra status
```

---

## ingest

Ingest content into the knowledge graph using LLM-powered extraction.

### `qortex ingest file <path>`

Extract concepts, relations, rules, and code examples from a file.

```bash
# Basic usage (auto-detects backend)
qortex ingest file chapter.txt --domain software_design

# Specify extraction backend
qortex ingest file chapter.txt --backend anthropic --domain patterns
qortex ingest file chapter.txt --backend ollama --model dolphin-mistral

# Preview without saving
qortex ingest file chapter.txt --domain test --dry-run

# Save manifest for recovery/inspection
qortex ingest file chapter.txt -d patterns -o manifest.json
```

Options:
- `--domain / -d`: Target domain (default: auto-suggested by LLM)
- `--backend / -b`: Extraction backend: `anthropic`, `ollama`, or `auto` (default: auto)
- `--model / -m`: Model override for the extraction backend
- `--dry-run`: Show extracted content without saving to graph
- `--save-manifest / -o`: Save extraction manifest to JSON (useful for recovery)

**Output:**
```
Domain: design_patterns
Concepts extracted: 285
Relations extracted: 119
Rules extracted: 6
Code examples extracted: 12
```

**Backend auto-detection:**

1. `anthropic` if `ANTHROPIC_API_KEY` is set
2. `ollama` if server is reachable at `OLLAMA_HOST` (default: localhost:11434)
3. Falls back to stub backend (empty results, for testing pipeline)

**Manifest auto-save:** If graph connection fails, the manifest is automatically saved to `<source>.manifest.json` so you don't lose extraction results.

**Supported formats:**
- `.txt`, `.text`: Plain text
- `.md`, `.markdown`: Markdown (preserves structure)
- `.pdf`: PDF (requires `pymupdf`)

### `qortex ingest load <manifest>`

Load a previously saved manifest into the graph (skip re-extraction).

```bash
qortex ingest load manifest.json
```

Useful for:
- Retrying after connection failures
- Loading extractions done offline
- Sharing manifests between systems

---

## project

Project rules from the knowledge graph.

### `qortex project buildlog`

Project rules in the universal schema format.

```bash
# To stdout
qortex project buildlog --domain error_handling

# To file
qortex project buildlog --domain error_handling -o rules.yaml

# To interop pending directory
qortex project buildlog --domain error_handling --pending
```

Options:
- `--domain / -d`: Limit to specific domain
- `--output / -o`: Output file path
- `--enrich / --no-enrich`: Enable/disable enrichment (default: enabled)
- `--persona / -p`: Persona name for output (default: "qortex")
- `--pending`: Write to interop pending directory
- `--emit / --no-emit`: Emit signal event with `--pending` (default: emit)

### `qortex project flat`

Project rules as flat YAML list.

```bash
qortex project flat --domain error_handling
```

Options:
- `--domain / -d`: Limit to specific domain
- `--output / -o`: Output file path

### `qortex project json`

Project rules as JSON.

```bash
qortex project json --domain error_handling
```

Options:
- `--domain / -d`: Limit to specific domain
- `--output / -o`: Output file path

---

## inspect

Inspect graph contents.

### `qortex inspect domains`

List all domains.

```bash
qortex inspect domains
```

Output:
```
Domains:
  error_handling: 15 concepts, 23 edges, 8 rules
  testing: 12 concepts, 18 edges, 5 rules
```

### `qortex inspect rules`

List rules in a domain.

```bash
qortex inspect rules --domain error_handling
```

Options:
- `--domain / -d`: Filter by domain
- `--limit / -n`: Max rules to show (default: 20)
- `--derived / --no-derived`: Include derived rules (default: include)

### `qortex inspect stats`

Show graph statistics.

```bash
qortex inspect stats
```

Output:
```
Graph Statistics:
  Domains: 3
  Concepts: 45
  Edges: 67
  Rules: 23
  Sources: 5
```

---

## viz

Visualization and Cypher queries.

### `qortex viz open`

Open Memgraph Lab in browser.

```bash
qortex viz open
```

### `qortex viz query`

Execute a Cypher query.

```bash
qortex viz query "MATCH (n) RETURN count(n)"
qortex viz query "MATCH (c:Concept) WHERE c.domain = 'error_handling' RETURN c.name"
```

Options:
- `--format / -f`: Output format (table, json, csv)

---

## interop

Consumer interop protocol management.

### `qortex interop init`

Initialize interop directories.

```bash
qortex interop init
```

Creates:
- `~/.qortex/seeds/pending/`
- `~/.qortex/seeds/processed/`
- `~/.qortex/seeds/failed/`
- `~/.qortex/signals/projections.jsonl`

### `qortex interop status`

Show interop status summary.

```bash
qortex interop status
```

Output:
```
Interop Status:
  Pending: 2 seeds
  Processed: 15 seeds
  Failed: 0 seeds
  Signals: 17 events
```

### `qortex interop pending`

List pending seed files.

```bash
qortex interop pending
```

Output:
```
Pending Seeds:
  error_rules_2026-02-05T12-00-00.yaml (5 rules)
  testing_rules_2026-02-05T12-30-00.yaml (3 rules)
```

### `qortex interop signals`

List recent signal events.

```bash
qortex interop signals
qortex interop signals --limit 50
```

Options:
- `--limit / -n`: Max events to show (default: 20)
- `--since`: Show events after timestamp (ISO format)

### `qortex interop schema`

Export JSON Schema files.

```bash
qortex interop schema --output ./schemas/
```

Options:
- `--output / -o`: Output directory (required)

Creates:
- `seed.v1.schema.json`
- `event.v1.schema.json`

### `qortex interop validate`

Validate a seed file against schema.

```bash
qortex interop validate rules.yaml
```

Exit codes:
- 0: Valid
- 1: Invalid (errors printed to stderr)

### `qortex interop config`

Show current interop configuration.

```bash
qortex interop config
```

Output:
```yaml
seeds:
  pending: /Users/you/.qortex/seeds/pending
  processed: /Users/you/.qortex/seeds/processed
  failed: /Users/you/.qortex/seeds/failed
signals:
  projections: /Users/you/.qortex/signals/projections.jsonl
```

---

## prune

Prune and analyze graph edges from saved manifests.

### `qortex prune manifest <path>`

Apply the 6-step pruning pipeline to edges in a saved manifest:

1. Minimum evidence length
2. Confidence floor
3. Jaccard deduplication
4. Competing relation resolution
5. Isolated weak edge removal
6. Structural/causal layer tagging

```bash
# Preview what would be pruned
qortex prune manifest ch05.manifest.json --dry-run

# Prune with custom thresholds and save
qortex prune manifest ch05.manifest.json -c 0.6 -o pruned.json

# Show details of dropped edges
qortex prune manifest ch05.manifest.json --show-dropped
```

Options:
- `--dry-run / -n`: Show what would be pruned without modifying
- `--min-confidence / -c`: Confidence floor (default: 0.55)
- `--min-evidence / -e`: Minimum evidence tokens (default: 8)
- `--output / -o`: Output path for pruned manifest
- `--show-dropped`: Show details of dropped edges

### `qortex prune stats <path>`

Show edge statistics without pruning. Displays confidence distribution, relation type breakdown, layer breakdown, and evidence quality metrics.

```bash
qortex prune stats ch05.manifest.json
```

Output:
```
Edge Statistics for: ch05.manifest.json
  Total edges: 119
  Total concepts: 285
  Edge density: 0.42 edges/concept

Confidence distribution:
      <0.55:    5 (  4.2%)
  0.55-0.70:   23 ( 19.3%)
  0.70-0.85:   51 ( 42.9%)
     >=0.85:   40 ( 33.6%)
```

---

## serve

### `qortex serve`

Start the qortex REST API server.

```bash
# Default: localhost:8741
qortex serve

# Custom host and port
qortex serve --host 0.0.0.0 --port 9000

# With API key authentication
QORTEX_API_KEY=my-secret-key qortex serve

# With CORS for web clients
QORTEX_CORS_ORIGINS="http://localhost:3000,https://myapp.com" qortex serve

# With PostgreSQL backends
QORTEX_STORE=postgres DATABASE_URL=postgresql://user:pass@localhost/qortex qortex serve
```

Options:
- `--host`: Bind address (default: `127.0.0.1`)
- `--port`: Port number (default: `8741`)
- `--reload`: Auto-reload on code changes (development only)

The REST API exposes all QortexClient operations as HTTP endpoints. See [API Reference](api.md#rest-api) for the full endpoint list.

**Authentication modes:**

| Mode | Header | Description |
|------|--------|-------------|
| API Key | `Authorization: Bearer <key>` | Simple token auth. Set `QORTEX_API_KEY` env var. |
| HMAC-SHA256 | `X-Qortex-Signature` + `X-Qortex-Timestamp` | Request signing with replay protection (60s window). Set `QORTEX_HMAC_SECRET` env var. |
| None | (no header) | Open access. Default when no auth env vars are set. |

---

## migrate

Data migration tools for moving between storage backends.

### `qortex migrate vec`

Migrate vector data between index backends.

```bash
# Migrate from SQLite to pgvector
qortex migrate vec --from sqlite --to pgvector

# Migrate with custom database URL
DATABASE_URL=postgresql://user:pass@localhost/qortex qortex migrate vec --from sqlite

# Dry run (count vectors without migrating)
qortex migrate vec --from sqlite --dry-run
```

Options:
- `--from`: Source backend (`sqlite`)
- `--to`: Target backend (default: `pgvector`)
- `--batch-size`: Vectors per batch (default: `1000`)
- `--dry-run`: Count vectors without migrating

The migration streams vectors via `iter_all()` to handle large indexes without loading everything into memory. Progress is reported as a percentage during the migration.

This command is also available as a REST endpoint (`POST /api/v1/migrate/vec`) and as an MCP tool (`qortex_migrate_vec`).

---

## mcp-serve

### `qortex mcp-serve`

Start the qortex MCP server.

```bash
# Default: stdio transport
qortex mcp-serve

# SSE transport (for web clients)
qortex mcp-serve --transport sse
```

Options:
- `--transport`: Transport protocol: `"stdio"` or `"sse"` (default: `"stdio"`)

---

## Environment Variables

### Core

| Variable | Description | Default |
|----------|-------------|---------|
| `QORTEX_STORE` | Storage backend for all persistent stores: `sqlite` or `postgres` | `sqlite` |
| `QORTEX_VEC` | Vector layer backend: `memory`, `sqlite`, or `pgvector` | `sqlite` |
| `QORTEX_GRAPH` | Graph layer backend: `memory` or `memgraph` | `memory` |
| `QORTEX_STATE_DIR` | Override for learning state persistence directory | (none) |
| `QORTEX_CONFIG` | Config file path | `~/.claude/qortex-consumers.yaml` |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM extraction and enrichment | (none) |
| `OLLAMA_HOST` | Ollama server URL for local LLM extraction | `http://localhost:11434` |

### REST API Server

| Variable | Description | Default |
|----------|-------------|---------|
| `QORTEX_API_KEY` | API key for Bearer token authentication | (none, no auth) |
| `QORTEX_HMAC_SECRET` | Secret for HMAC-SHA256 request signing | (none) |
| `QORTEX_CORS_ORIGINS` | Comma-separated allowed CORS origins | (none, CORS disabled) |

### PostgreSQL

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string for shared asyncpg pool | (none) |

### Memgraph

| Variable | Description | Default |
|----------|-------------|---------|
| `QORTEX_GRAPH` | Graph backend: `memory` or `memgraph` | `memory` |
| `QORTEX_VEC` | Vector backend: `memory`, `sqlite`, or `pgvector` | `sqlite` |
| `QORTEX_STORE` | Persistence backend: `sqlite` or `postgres` | `sqlite` |
| `QORTEX_STATE_DIR` | Override for state directory | `~/.qortex` |
| `PGVECTOR_DSN` | Full PostgreSQL connection string | *(constructed)* |
| `PGVECTOR_HOST` | PostgreSQL host | `localhost` |
| `PGVECTOR_PORT` | PostgreSQL port | `5432` |
| `PGVECTOR_USER` | PostgreSQL user | `qortex` |
| `PGVECTOR_PASSWORD` | PostgreSQL password | `qortex` |
| `PGVECTOR_DB` | PostgreSQL database | `qortex` |
| `MEMGRAPH_HOST` | Memgraph hostname | `localhost` |
| `MEMGRAPH_PORT` | Memgraph port | `7687` |
| `MEMGRAPH_USER` | Memgraph username | (none) |
| `MEMGRAPH_PASSWORD` | Memgraph password | (none) |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM extraction and enrichment | (none) |
| `OLLAMA_HOST` | Ollama server URL for local LLM extraction | `http://localhost:11434` |
| `QORTEX_CONFIG` | Config file path | `~/.claude/qortex-consumers.yaml` |

See [Environment Variables](environment-variables.md) for the full reference including observability, logging, and auth settings.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments (with `--help`) |

---

## Examples

### Full Workflow

```bash
# 1. Start infrastructure
qortex infra up

# 2. Ingest content
qortex ingest file book.txt --domain software_design -o manifest.json

# 3. Inspect what was ingested
qortex inspect domains
qortex inspect rules --domain software_design

# 4. Visualize in Memgraph Lab
qortex viz open
# Query: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50

# 5. Project rules
qortex project buildlog --domain software_design --pending

# 6. Check interop
qortex interop status
```

### Query Examples

```bash
# Count concepts per domain
qortex viz query "MATCH (c:Concept) RETURN c.domain, count(c)"

# Find concepts with high confidence
qortex viz query "MATCH (c:Concept) WHERE c.confidence > 0.9 RETURN c.name"

# Find REQUIRES relationships
qortex viz query "MATCH (a)-[:REQUIRES]->(b) RETURN a.name, b.name"
```
