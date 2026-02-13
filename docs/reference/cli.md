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
| `prune` | Edge pruning and analysis |
| `mcp-serve` | Start the MCP server (top-level command) |

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

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMGRAPH_HOST` | Memgraph hostname | `localhost` |
| `MEMGRAPH_PORT` | Memgraph port | `7687` |
| `MEMGRAPH_USER` | Memgraph username | (none) |
| `MEMGRAPH_PASSWORD` | Memgraph password | (none) |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM extraction and enrichment | (none) |
| `QORTEX_CONFIG` | Config file path | `~/.claude/qortex-consumers.yaml` |
| `QORTEX_VEC` | Vector layer backend: `memory` or `sqlite` | `sqlite` |
| `QORTEX_GRAPH` | Graph layer backend: `memory` or `memgraph` | `memory` |
| `QORTEX_STATE_DIR` | Override for learning state persistence directory | (none) |
| `OLLAMA_HOST` | Ollama server URL for local LLM extraction | `http://localhost:11434` |

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
