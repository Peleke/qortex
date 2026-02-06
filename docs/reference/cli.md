# CLI Reference

qortex provides a comprehensive CLI for managing knowledge graphs.

## Global Options

```bash
qortex --help          # Show all commands
qortex <command> --help  # Show command help
```

## Command Groups

| Group | Description |
|-------|-------------|
| `infra` | Infrastructure management (Memgraph) |
| `ingest` | Content ingestion |
| `project` | Rule projection |
| `inspect` | Graph inspection |
| `viz` | Visualization and queries |
| `interop` | Consumer interop protocol |

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

### `qortex ingest <path>`

Extract concepts, relations, and rules from a file.

```bash
# Basic usage (auto-detects backend)
qortex ingest chapter.txt --domain software_design

# Specify extraction backend
qortex ingest chapter.txt --backend anthropic --domain patterns
qortex ingest chapter.txt --backend ollama --model dolphin-mistral

# Preview without saving
qortex ingest chapter.txt --domain test --dry-run
```

Options:
- `--domain / -d`: Target domain (default: auto-suggested by LLM)
- `--backend / -b`: Extraction backend: `anthropic`, `ollama`, or `auto` (default: auto)
- `--model / -m`: Model override for the extraction backend
- `--dry-run`: Show extracted concepts/relations/rules without saving to graph

**Backend auto-detection:**

1. `anthropic` if `ANTHROPIC_API_KEY` is set
2. `ollama` if server is reachable at `OLLAMA_HOST` (default: localhost:11434)
3. Falls back to stub backend (empty results, for testing pipeline)

**Supported formats:**
- `.txt`, `.text` — Plain text
- `.md`, `.markdown` — Markdown (preserves structure)
- `.pdf` — PDF (not yet implemented)

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

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMGRAPH_HOST` | Memgraph hostname | localhost |
| `MEMGRAPH_PORT` | Memgraph port | 7687 |
| `MEMGRAPH_USER` | Memgraph username | (none) |
| `MEMGRAPH_PASSWORD` | Memgraph password | (none) |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM enrichment | (none) |
| `QORTEX_CONFIG` | Config file path | ~/.claude/qortex-consumers.yaml |

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
qortex ingest book.pdf --domain software_design

# 3. Inspect what was ingested
qortex inspect domains
qortex inspect rules --domain software_design

# 4. Project rules
qortex project buildlog --domain software_design --pending

# 5. Check interop
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
