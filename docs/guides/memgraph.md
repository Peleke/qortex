# Using Memgraph

Memgraph is qortex's production backend, providing persistent storage and advanced graph algorithms.

## Why Memgraph?

| Feature | InMemoryBackend | MemgraphBackend |
|---------|-----------------|-----------------|
| Persistence | No | Yes |
| Scalability | Small datasets | Production scale |
| Cypher queries | No | Yes |
| MAGE algorithms | No | Yes (PageRank, etc.) |
| Use case | Testing, dev | Production |

## Quick Start

### 1. Start Memgraph

```bash
qortex infra up
```

This starts:
- Memgraph database on port 7687
- Memgraph Lab (web UI) on port 3000

### 2. Verify Connection

```bash
qortex infra status
```

### 3. Use in Code

```python
from qortex.core.backend import MemgraphBackend

backend = MemgraphBackend(
    host="localhost",
    port=7687,
)
backend.connect()

# Use like InMemoryBackend
backend.create_domain("error_handling", "Error patterns")
backend.add_node(node)
backend.add_edge(edge)
```

## Docker Setup

The default Docker Compose configuration:

```yaml
# docker/docker-compose.yml
services:
  memgraph:
    image: memgraph/memgraph-mage:latest
    ports:
      - "7687:7687"
      - "7444:7444"
    volumes:
      - memgraph-data:/var/lib/memgraph
    environment:
      - MEMGRAPH_USER=${MEMGRAPH_USER:-}
      - MEMGRAPH_PASSWORD=${MEMGRAPH_PASSWORD:-}

  lab:
    image: memgraph/lab:latest
    ports:
      - "3000:3000"
    depends_on:
      - memgraph
```

### Custom Configuration

Set credentials via environment:

```bash
export MEMGRAPH_USER=admin
export MEMGRAPH_PASSWORD=secret
qortex infra up
```

## Cypher Queries

Execute Cypher directly:

```python
# Find all concepts in a domain
results = backend.query_cypher("""
    MATCH (c:Concept {domain: $domain})
    RETURN c.id, c.name, c.description
""", {"domain": "error_handling"})

# Find concepts connected by REQUIRES
results = backend.query_cypher("""
    MATCH (a:Concept)-[:REQUIRES]->(b:Concept)
    WHERE a.domain = $domain
    RETURN a.name, b.name
""", {"domain": "error_handling"})
```

Via CLI:

```bash
qortex viz query "MATCH (n) RETURN n LIMIT 10"
```

## MAGE Algorithms

Memgraph's MAGE library provides graph algorithms.

### Personalized PageRank

Find the most relevant concepts to a query:

```python
scores = backend.personalized_pagerank(
    source_ids=["circuit_breaker"],
    domain="error_handling",
    damping_factor=0.85,
)

# Returns: {"timeout": 0.35, "retry": 0.28, ...}
```

PPR enables future HippoRAG-style retrieval: start from query-matched concepts and spread activation through the graph.

## Schema

qortex uses this Cypher schema:

```cypher
# Domain node
(:Domain {name: string, description: string, created_at: datetime, updated_at: datetime})

# Concept node (labeled by domain)
(:Concept {id: string, name: string, description: string, domain: string, source_id: string})

# Typed relationships
(s:Concept)-[:REQUIRES {confidence: float}]->(t:Concept)
(s:Concept)-[:CONTRADICTS {confidence: float}]->(t:Concept)
# ... etc for all RelationTypes

# Rule node
(:Rule {id: string, text: string, domain: string, category: string})
(:Rule)-[:REFERENCES]->(:Concept)
```

## Checkpoints

Save and restore graph state:

```python
# Save current state
checkpoint_id = backend.checkpoint()

# ... make changes ...

# Restore to checkpoint
backend.restore(checkpoint_id)
```

## CLI Commands

```bash
# Start infrastructure
qortex infra up

# Check status
qortex infra status

# Stop infrastructure
qortex infra down

# Inspect graph
qortex inspect domains
qortex inspect rules --domain error_handling
qortex inspect stats

# Run Cypher
qortex viz query "MATCH (n) RETURN count(n)"

# Open Memgraph Lab
qortex viz open
```

## Configuration

Configure connection via environment:

```bash
export MEMGRAPH_HOST=localhost
export MEMGRAPH_PORT=7687
export MEMGRAPH_USER=admin
export MEMGRAPH_PASSWORD=secret
```

Or in code:

```python
backend = MemgraphBackend(
    host="memgraph.example.com",
    port=7687,
    username="admin",
    password="secret",
)
```

## Migrations

When upgrading qortex with schema changes:

```python
# Check current schema version
version = backend.get_schema_version()

# Run migrations
backend.migrate()
```

## Testing with Memgraph

Integration tests require a running Memgraph instance:

```bash
# Start Memgraph
qortex infra up

# Run integration tests
pytest tests/test_memgraph_integration.py -v

# Tests are skipped if Memgraph unavailable
```

## Performance Tips

1. **Batch operations**: Use `ingest_manifest()` for bulk inserts
2. **Index domains**: Memgraph auto-indexes `:Domain` nodes
3. **Limit PPR iterations**: Default damping factor (0.85) converges quickly
4. **Use projections**: Don't fetch full nodes if you only need IDs

```python
# Good: batch insert
backend.ingest_manifest(manifest)

# Avoid: individual inserts in loop
for node in nodes:
    backend.add_node(node)  # Slower
```

## Troubleshooting

### Connection refused

```
Error: Connection refused on localhost:7687
```

Solution: Ensure Memgraph is running:

```bash
qortex infra up
qortex infra status
```

### Authentication failed

```
Error: Authentication failed
```

Solution: Check credentials:

```bash
export MEMGRAPH_USER=admin
export MEMGRAPH_PASSWORD=your_password
```

### Out of memory

For large graphs, increase Memgraph memory:

```yaml
# docker-compose.override.yml
services:
  memgraph:
    command: ["--memory-limit=4096"]
```

## Next Steps

- [Projecting Rules](projecting-rules.md) - Project from Memgraph
- [Consumer Integration](consumer-integration.md) - Distribute to consumers
- [Architecture Overview](../architecture/overview.md) - System design
