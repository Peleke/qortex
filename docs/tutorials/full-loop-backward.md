# Part II: Backward Flow

**Agent mistakes become graph insights.**

Buildlog emits structured mistake data. We transform it, load it into the knowledge graph, and link mistakes to the design patterns that might prevent them.

[Back to Overview](full-loop-overview.md) | [Part I: Forward Flow](full-loop-forward.md)

---

## The Emission Data

Buildlog emits data to `~/.buildlog/emissions/pending/`. After several weeks of agent work:

- 338 learned_rules
- 192 mistake_manifests
- 28 reward_signals

The mistake manifests capture what went wrong: error class, whether it was a repeat, and descriptive text.

## Transforming for Ingestion

Each mistake manifest becomes a concept node. We aggregate by error class to create summary nodes.

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

Load the transformed manifest:

```bash
qortex ingest load /tmp/buildlog_mistakes_manifest.json
```

```
Loaded manifest: experiential
  Concepts: 100
  Edges: 0
  Rules: 0
```

Why zero edges and rules? The mistake manifests contain only concept data. Edges come next, when we link mistakes to design patterns. Rules aren't part of experiential data — mistakes are observations, not prescriptions.

![Buildlog Mistakes Ingested](../demo-screenshots/09-buildlog-mistakes-ingested.png)

## Error Class Distribution

| Error Class | Total | Repeats | Rate |
|-------------|-------|---------|------|
| test | 99 | 61 | **61.6%** |
| missing_test | 53 | 26 | **49.1%** |
| security | 13 | 0 | 0.0% |
| validation | 13 | 0 | 0.0% |
| typo | 13 | 0 | 0.0% |

Test-related errors have dramatically higher repeat rates. These are persistent blind spots.

## Cross-Domain Linking

The real insight comes from linking mistakes to relevant design patterns. We create edges connecting experiential aggregates to pattern concepts:

```cypher
MATCH (e:Concept {domain: 'experiential'})-[r]->(p:Concept)
WHERE p.domain <> 'experiential'
RETURN e.name, type(r), p.name, p.domain
```

![Cross-Domain Mistake Links](../demo-screenshots/10-cross-domain-mistake-links.png)

| Mistake Type | Relation | Pattern Concept | Domain |
|--------------|----------|-----------------|--------|
| Test Errors (Aggregate) | CHALLENGES | Algorithm Encapsulation | iterator_visitor_patterns |
| Test Errors (Aggregate) | CHALLENGES | Object Creation | factory_patterns |
| Missing Test Errors (Aggregate) | CHALLENGES | Algorithm Steps | template_strategy_patterns |
| Security Errors (Aggregate) | SUPPORTS | Private Methods | implementation_hiding |
| Validation Errors (Aggregate) | CHALLENGES | External Code Integration | adapter_facade_patterns |

The "Test Errors (Aggregate)" label is a summary node we created during transformation. It represents all 99 individual test-related mistakes. The aggregate node links to pattern concepts; individual mistakes stay as leaf nodes for detailed analysis.

The 61.6% repeat rate on test errors isn't a bug — it's a diagnostic. It points directly at iterator_visitor_patterns and factory_patterns as the knowledge domains that need reinforcement. The graph tells us where the gaps are.

## What the Agent Knows Now

After both flows complete, the agent has access to:

- 6 design pattern domains (1,088 concepts from book chapters)
- 1 experiential domain (100 mistake concepts + 5 aggregates)
- 833 edges connecting concepts within and across domains
- 135+ rules per domain projected to agent personas

The cross-domain links mean a query about test errors can surface relevant Iterator and Factory pattern rules. The agent's knowledge isn't siloed — it's a connected graph.

## Implications

### For test_terrorist

The 61.6% repeat rate suggests augmenting the persona with rules from:
- iterator_visitor_patterns (Algorithm Encapsulation)
- factory_patterns (Object Creation)

### For security_karen

Security errors link to implementation_hiding patterns. Proactive rules about Private Methods could prevent these errors before they happen.

### For the feedback loop

High-repeat errors should trigger confidence boosts on related rules in buildlog's bandit system. When test errors keep happening, the system should weight test-related rules higher. This creates a self-correcting cycle — but the mechanism isn't implemented yet.

---

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

# Load a manifest
uv run qortex ingest load manifest.json

# Check interop status
uv run qortex interop status
```

## What's Next

- Automate the reverse flow (scheduled emission ingestion)
- Build confidence feedback from reward_signals
- Implement HippoRAG for cross-domain retrieval
- Create retro analytics dashboard

[Back to Overview](full-loop-overview.md)
