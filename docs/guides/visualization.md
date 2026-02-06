# Visualizing Your Knowledge Graph

You've ingested content. You've got concepts and edges in a database. Now you want to *see* it.

Memgraph Lab is a browser-based graph visualization tool that ships with the qortex Docker setup. One command gets you there.

## Quick Start

```bash
# Start Memgraph + Lab
qortex infra up

# Open the visualization UI
qortex viz open
```

Your browser opens to Memgraph Lab at `http://localhost:3000`. You're looking at an empty canvas. Time to query.

## See Everything You Ingested

Click **Query Execution** in the left sidebar (or press `Ctrl+Enter` in the query box).

### View all concepts

```cypher
MATCH (c:Concept)
RETURN c
LIMIT 50
```

Nodes appear as circles. Each one is a concept from your ingested content. Hover to see properties (name, description, domain).

### View all relationships

```cypher
MATCH (a:Concept)-[r]->(b:Concept)
RETURN a, r, b
LIMIT 100
```

Now you see the edges. The graph structure is visible. Concepts cluster around shared relationships.

### View a specific domain

```cypher
MATCH (c:Concept {domain: "error_handling"})
RETURN c
```

Or with relationships:

```cypher
MATCH (a:Concept {domain: "error_handling"})-[r]->(b:Concept)
RETURN a, r, b
```

## Useful Queries

### What domains exist?

```cypher
MATCH (d:Domain)
RETURN d.name, d.description
```

### How many concepts per domain?

```cypher
MATCH (c:Concept)
RETURN c.domain, count(c) as concept_count
ORDER BY concept_count DESC
```

### What relationships connect two concepts?

```cypher
MATCH (a:Concept {name: "Circuit Breaker"})-[r]->(b:Concept)
RETURN a.name, type(r), b.name
```

### Find paths between concepts

```cypher
MATCH path = shortestPath(
  (a:Concept {name: "Retry"})-[*]-(b:Concept {name: "Timeout"})
)
RETURN path
```

This visualizes the shortest connection between two concepts. Powerful for understanding how ideas relate.

### View rules and their source concepts

```cypher
MATCH (r:Rule)-[:REFERENCES]->(c:Concept)
RETURN r.text, collect(c.name) as concepts
LIMIT 20
```

## From the CLI

Don't want to open the browser? Run queries directly:

```bash
# Count all nodes
qortex viz query "MATCH (n) RETURN count(n)"

# List domains
qortex viz query "MATCH (d:Domain) RETURN d.name"

# Find concepts by pattern
qortex viz query "MATCH (c:Concept) WHERE c.name CONTAINS 'error' RETURN c.name"
```

Output prints to terminal. Good for scripts and quick checks.

## Tips for Exploration

**Zoom and pan**: Scroll to zoom, drag to pan. Large graphs need navigation.

**Click nodes**: Selecting a node shows its properties in the sidebar.

**Expand neighbors**: Right-click a node → "Expand" to fetch connected nodes without rerunning the query.

**Layout options**: Memgraph Lab has force-directed, hierarchical, and circular layouts. Experiment to find what makes your graph readable.

**Export**: You can export query results as CSV or JSON for further analysis.

## The "Aha" Moment

Run this after ingesting a book chapter:

```cypher
MATCH (a:Concept)-[r]->(b:Concept)
WHERE a.domain = "your_domain"
RETURN a, r, b
```

Watch the graph render. Those aren't just database rows. That's the *structure* of the knowledge you ingested. Concepts you authored are now nodes. Relationships you defined (or that qortex derived) are visible edges.

This is what qortex builds. The visualization is just the proof.

## Next Steps

- [Using Memgraph](memgraph.md): Full Memgraph backend reference
- [Projecting Rules](projecting-rules.md): Turn this graph into actionable rules
- [Quick Start](../getting-started/quickstart.md): Ingest your first content
