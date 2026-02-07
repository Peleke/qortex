# Part I: Forward Flow

**Book chapters to agent personas.**

This walkthrough covers ingesting a design patterns chapter, exploring the resulting knowledge graph, projecting rules to buildlog format, and loading them into the agent system.

[Back to Overview](full-loop-overview.md)

---

## Connecting to Memgraph

Start the Docker stack and open Memgraph Lab at `http://localhost:3000`. Enter the credentials (memgraph/memgraph) to connect.

![Memgraph Connected](../demo-screenshots/02-memgraph-connected.png)

**Starting state**: 988 nodes, 549 edges across 5 domains from previously ingested book chapters (ch05, ch08-ch11).

## Exploring the Graph

Run a basic query to see the concept network:

```cypher
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50
```

![Graph Visualization](../demo-screenshots/03-graph-visualization.png)

The force-directed layout reveals concept clusters. Each node is a concept extracted from a design patterns book. The edges represent semantic relationships.

![Graph Results](../demo-screenshots/04-graph-results.png)

## Edge Types Matter

Edge types are Cypher labels, not properties. This enables graph algorithms like PageRank that operate on relationship types.

```cypher
MATCH ()-[r]->()
RETURN DISTINCT type(r) AS edge_type, count(*) AS count
ORDER BY count DESC
```

![Edge Types Distribution](../demo-screenshots/05-edge-types-distribution.png)

**10 semantic edge types**:

| Edge Type | Count | Meaning |
|-----------|-------|---------|
| SUPPORTS | 141 | Concept A reinforces concept B |
| USES | 117 | Concept A employs concept B |
| IMPLEMENTS | 112 | Concept A realizes concept B |
| PART_OF | 53 | Concept A is a component of B |
| REQUIRES | 40 | Concept A depends on concept B |
| CHALLENGES | 33 | Concept A conflicts with concept B |
| REFINES | 25 | Concept A specializes concept B |
| SIMILAR_TO | 15 | Concepts are analogous |
| ALTERNATIVE_TO | 9 | Concepts are interchangeable |
| CONTRADICTS | 4 | Concepts are mutually exclusive |

## Ingesting a New Chapter

Add Chapter 12 (Observer Pattern) to the graph:

```bash
set -a && source .env.local && set +a
echo "y" | uv run qortex ingest file \
  "data/books/ch12_12_The_Observer_Design_Pattern.txt" \
  --domain observer_pattern \
  --backend anthropic \
  --save-manifest data/manifests/ch12_manifest.json
```

After extraction:

- **265 concepts** extracted
- **139 edges** (relationships between concepts)
- **6 explicit rules** from the text
- **2 code examples**

![Graph After Ch12](../demo-screenshots/06-graph-after-ch12.png)

**Updated state**: 1,132 nodes (+144), 827 edges (+278)

## Domain Distribution

Query concept counts by domain:

```cypher
MATCH (n:Concept)
RETURN n.domain AS domain, count(*) AS concepts
ORDER BY concepts DESC
```

![Domains Concept Counts](../demo-screenshots/07-domains-concept-counts.png)

| Domain | Concepts |
|--------|----------|
| iterator_visitor_patterns | 249 |
| implementation_hiding | 237 |
| template_strategy_patterns | 170 |
| factory_patterns | 158 |
| observer_pattern | 137 |
| adapter_facade_patterns | 137 |

## Graph Analytics: PageRank

Which concepts are most central to the knowledge graph?

```cypher
CALL pagerank.get()
YIELD node, rank
RETURN node.name AS concept, node.domain AS domain, rank
ORDER BY rank DESC LIMIT 15
```

![PageRank Top Concepts](../demo-screenshots/08-pagerank-top-concepts.png)

**Top concepts by PageRank**:

1. Interface Compatibility (adapter_facade_patterns) — 0.0071
2. Visitor Design Pattern (iterator_visitor_patterns) — 0.0068
3. Tree Node Processing (iterator_visitor_patterns) — 0.0067
4. Architecture Problem (iterator_visitor_patterns) — 0.0063
5. Algorithm Encapsulation (iterator_visitor_patterns) — 0.0059

These are the hub concepts that connect many others.

## Projecting Rules to Buildlog

Project the observer_pattern rules to buildlog format:

```bash
uv run qortex project buildlog \
  --domain observer_pattern \
  -p qortex_observer \
  -o /tmp/qortex_observer.yaml
```

**Output**: 135 rules in universal schema format.

The seed file structure:

```yaml
persona: qortex_observer
version: 1
rules:
  - rule: "Publishers should not know how subscribers process data..."
    category: observer_pattern
    provenance:
      id: "observer_pattern:rule:0"
      domain: observer_pattern
      derivation: explicit
      confidence: 0.9
metadata:
  source: qortex
  source_version: "0.1.0"
  projected_at: "2026-02-07T00:24:00+00:00"
  rule_count: 135
```

## Ingesting into Buildlog

Copy the seed to the interop directory and ingest:

```bash
cp /tmp/qortex_observer.yaml ~/.qortex/seeds/pending/
cd ../buildlog-template
uv run buildlog ingest-seeds
```

**Output**:

```
[qortex] 1 ingested
  qortex_observer.yaml (qortex_observer, 135 rules)
```

The rules now live in `.buildlog/seeds/qortex_observer.yaml` and influence agent behavior during gauntlet reviews.

---

**Next**: [Part II: Backward Flow](full-loop-backward.md) — Agent mistakes become graph insights.
