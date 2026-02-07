# Full Loop Demo: Books to Rules to Mistakes and Back

This tutorial walks through the complete bidirectional knowledge flow between qortex and buildlog, demonstrating how design pattern knowledge from books can influence AI agent behavior, and how agent mistakes can feed back into the knowledge graph.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BIDIRECTIONAL KNOWLEDGE FLOW                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   FORWARD FLOW (Knowledge → Agents)                                 │
│   ════════════════════════════════                                  │
│   📚 Book Chapter                                                   │
│      ↓ qortex ingest file --backend anthropic                       │
│   🧠 Memgraph Knowledge Graph (concepts, edges, rules)              │
│      ↓ qortex project buildlog --domain X                           │
│   📋 YAML Seed File (universal schema)                              │
│      ↓ buildlog ingest-seeds                                        │
│   🤖 Agent Personas (test_terrorist, security_karen, etc.)          │
│                                                                     │
│   REVERSE FLOW (Agents → Knowledge)                                 │
│   ═════════════════════════════════                                 │
│   🤖 Agent makes mistakes during work                               │
│      ↓ buildlog emits mistake_manifest                              │
│   📁 ~/.buildlog/emissions/pending/                                 │
│      ↓ transform to qortex manifest                                 │
│   🧠 Memgraph (experiential domain)                                 │
│      ↓ cross-domain edges                                           │
│   🔗 Mistakes linked to relevant design patterns                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker running with Memgraph + Lab
- qortex installed (`uv sync` in the repo)
- buildlog-template configured
- Anthropic API key in `.env.local`

## Part 1: Starting State

First, let's see Memgraph Lab before we connect.

![Memgraph Lab Initial](../demo-screenshots/01-memgraph-lab-initial.png)

The Lab shows "Memgraph not connected" - we need to authenticate. After the auth fix (adding `MEMGRAPH_USER=memgraph` and `MEMGRAPH_PASSWORD=memgraph` to docker-compose), we can connect.

## Part 2: Connecting to the Graph

After entering credentials (memgraph/memgraph), we connect and see the existing graph state.

![Memgraph Connected](../demo-screenshots/02-memgraph-connected.png)

**Initial state**: 988 nodes, 549 edges across 5 domains from previously ingested book chapters (ch05, ch08-ch11).

## Part 3: Exploring the Graph

Running a basic query `MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50` shows the concept network:

![Graph Visualization](../demo-screenshots/03-graph-visualization.png)

The force-directed layout reveals concept clusters:

![Graph Results](../demo-screenshots/04-graph-results.png)

Each node is a concept extracted from a design patterns book. The edges represent semantic relationships.

## Part 4: Edge Types Matter

A critical discovery: edge types must be Cypher labels, not properties. Running:

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

These typed edges enable MAGE analytics (PageRank, centrality) that wouldn't work with generic `:REL` edges.

## Part 5: Ingesting a New Chapter

Let's add Chapter 12 (Observer Pattern) to the graph:

```bash
set -a && source .env.local && set +a
echo "y" | uv run qortex ingest file \
  "data/books/ch12_12_The_Observer_Design_Pattern.txt" \
  --domain observer_pattern \
  --backend anthropic \
  --save-manifest data/manifests/ch12_manifest.json
```

After extraction completes:
- **265 concepts** extracted
- **139 edges** (relationships between concepts)
- **6 explicit rules** from the text
- **2 code examples**

![Graph After Ch12](../demo-screenshots/06-graph-after-ch12.png)

**Updated state**: 1,132 nodes (+144), 827 edges (+278)

## Part 6: Domain Distribution

Querying concept counts by domain:

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

## Part 7: Graph Analytics (PageRank)

Which concepts are most central to the knowledge graph?

```cypher
CALL pagerank.get()
YIELD node, rank
RETURN node.name AS concept, node.domain AS domain, rank
ORDER BY rank DESC LIMIT 15
```

![PageRank Top Concepts](../demo-screenshots/08-pagerank-top-concepts.png)

**Top concepts by PageRank**:
1. Interface Compatibility (adapter_facade_patterns) - 0.0071
2. Visitor Design Pattern (iterator_visitor_patterns) - 0.0068
3. Tree Node Processing (iterator_visitor_patterns) - 0.0067
4. Architecture Problem (iterator_visitor_patterns) - 0.0063
5. Algorithm Encapsulation (iterator_visitor_patterns) - 0.0059

These are the "hub" concepts that connect many others.

## Part 8: Projecting Rules to Buildlog

Now we project the observer_pattern rules to buildlog format:

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

## Part 9: Ingesting into Buildlog

Copy the seed to the interop directory and ingest:

```bash
cp /tmp/qortex_observer.yaml ~/.qortex/seeds/pending/
cd ../buildlog-template
uv run buildlog ingest-seeds
```

**Output**:
```
[qortex] 1 ingested
  ✓ qortex_observer.yaml (qortex_observer, 135 rules)
```

The rules now live in `.buildlog/seeds/qortex_observer.yaml` (111KB) and will influence agent behavior during gauntlet reviews.

## Part 10: The Reverse Flow - Ingesting Mistakes

Buildlog emits mistake data to `~/.buildlog/emissions/pending/`. Let's bring this back into qortex.

**Emission counts**:
- 338 learned_rules
- 192 mistake_manifests
- 28 reward_signals

Transform and load:

```python
# Extract mistake concepts from emissions
mistakes = []
for f in emissions_dir.glob("mistake_manifest_*.json"):
    data = json.load(f)
    for concept in data["concepts"]:
        mistakes.append({
            "id": f"experiential:{concept['name']}",
            "name": concept["properties"]["description"],
            "domain": "experiential",
            "properties": {
                "error_class": concept["properties"]["error_class"],
                "was_repeat": concept["properties"]["was_repeat"]
            }
        })
```

After loading:

![Buildlog Mistakes Ingested](../demo-screenshots/09-buildlog-mistakes-ingested.png)

**Graph updated**: 1,233 nodes (+101 mistake concepts)

**Error class distribution**:
| Error Class | Total | Repeats | Rate |
|-------------|-------|---------|------|
| test | 99 | 61 | **61.6%** |
| missing_test | 53 | 26 | **49.1%** |
| security | 13 | 0 | 0.0% |
| validation | 13 | 0 | 0.0% |
| typo | 13 | 0 | 0.0% |

**Key insight**: Test-related errors have dramatically higher repeat rates (61.6%), indicating persistent blind spots.

## Part 11: Cross-Domain Linking

The real power comes from linking mistakes to relevant design patterns:

```cypher
MATCH (e:Concept {domain: 'experiential'})-[r]->(p:Concept)
WHERE p.domain <> 'experiential'
RETURN e.name, type(r), p.name, p.domain
```

![Cross-Domain Mistake Links](../demo-screenshots/10-cross-domain-mistake-links.png)

**Cross-domain edges created**:
| Mistake Type | Relation | Pattern Concept | Domain |
|--------------|----------|-----------------|--------|
| Test Errors | CHALLENGES | Algorithm Encapsulation | iterator_visitor_patterns |
| Test Errors | CHALLENGES | Object Creation | factory_patterns |
| Missing Test Errors | CHALLENGES | Algorithm Steps | template_strategy_patterns |
| Security Errors | SUPPORTS | Private Methods | implementation_hiding |
| Validation Errors | CHALLENGES | External Code Integration | adapter_facade_patterns |

**Final graph**: 1,238 nodes, 833 edges, 7 domains

## Implications

### For test_terrorist persona
The 61.6% repeat rate on test errors suggests augmenting the persona with rules from:
- iterator_visitor_patterns (Algorithm Encapsulation)
- factory_patterns (Object Creation)

### For security_karen persona
Security errors link to implementation_hiding patterns - proactive rules about Private Methods could prevent these errors.

### For the feedback loop
High-repeat errors should trigger confidence boosts on related rules in buildlog's bandit system, creating a self-improving cycle.

## Summary

We demonstrated:

1. **Forward flow**: Book → KG → Rules → Agent personas
2. **Reverse flow**: Agent mistakes → KG → Cross-domain links
3. **Analytics**: PageRank identifies hub concepts
4. **Insights**: 61.6% test error repeat rate reveals blind spots

The bidirectional flow creates a learning system where:
- Design pattern knowledge improves agent behavior
- Agent mistakes reveal which patterns need emphasis
- Cross-domain links suggest targeted improvements

## Commands Reference

```bash
# Start Memgraph
docker compose -f docker/docker-compose.yml up -d

# Ingest a book chapter
uv run qortex ingest file chapter.txt --domain my_domain --backend anthropic

# Project rules to buildlog
uv run qortex project buildlog --domain my_domain -p persona_name -o output.yaml

# Ingest seeds in buildlog
cd ../buildlog-template && uv run buildlog ingest-seeds

# Check interop status
uv run qortex interop status

# Load a manifest
uv run qortex ingest load manifest.json
```

## Next Steps

- [ ] Automate the reverse flow (scheduled emission ingestion)
- [ ] Build confidence feedback from reward_signals
- [ ] Implement HippoRAG for cross-domain retrieval
- [ ] Create retro analytics dashboard
