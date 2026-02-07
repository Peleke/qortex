# Experiential Mistakes EDA

Analysis of mistake data ingested from buildlog emissions into the qortex knowledge graph.

## Data Source

- **Source**: `~/.buildlog/emissions/pending/mistake_manifest_*.json`
- **Total emissions**: 558 (338 learned_rules, 192 mistake_manifests, 28 reward_signals)
- **Concepts loaded**: 100 individual mistakes + 5 aggregate error class nodes

## Research Questions

1. Which error classes have highest repeat rates? (persistent blind spots)
2. Do mistakes cluster in time? (fatigue/context-switching)
3. How many mistakes per session?
4. Can we link mistakes to design pattern concepts?
5. Do certain error types co-occur?

## Findings

### Q1: Repeat Rates by Error Class

| Error Class | Total | Repeats | Rate |
|-------------|-------|---------|------|
| test | 99 | 61 | **61.6%** |
| missing_test | 53 | 26 | **49.1%** |
| security | 13 | 0 | 0.0% |
| validation | 13 | 0 | 0.0% |
| typo | 13 | 0 | 0.0% |
| type-errors | 1 | 0 | 0.0% |

**Insight**: Test-related errors have dramatically higher repeat rates, indicating persistent blind spots that design patterns could address.

### Q2: Temporal Clustering

- **Peak hours**: 00:00 (61 mistakes), 20:00 (55), 22:00 (46), 23:00 (30)
- **Pattern**: Late night/early morning coding sessions produce the most mistakes

**Insight**: Mistakes cluster in evening hours, suggesting fatigue-related patterns.

### Q3: Session Patterns

- **Total sessions**: 192
- **Avg mistakes/session**: 1.0
- **Structure**: Each emission represents one mistake event

### Q4: Cross-Domain Linkages

Created edges connecting experiential mistake aggregates to relevant design pattern concepts:

| Mistake Type | Relation | Pattern Concept | Domain |
|--------------|----------|-----------------|--------|
| Test Errors | CHALLENGES | Algorithm Encapsulation | iterator_visitor_patterns |
| Test Errors | CHALLENGES | Object Creation | factory_patterns |
| Missing Test Errors | CHALLENGES | Algorithm Steps | template_strategy_patterns |
| Security Errors | SUPPORTS | Private Methods | implementation_hiding |
| Security Errors | CHALLENGES | Interface Compatibility | adapter_facade_patterns |
| Validation Errors | CHALLENGES | External Code Integration | adapter_facade_patterns |

**Insight**: High-repeat error classes (test, missing_test) connect to algorithmic patterns, suggesting that better understanding of Iterator/Visitor and Factory patterns could reduce these errors.

### Q5: Error Class Co-occurrence

No significant co-occurrence detected due to 1:1 session:mistake ratio in current emission structure.

## Graph Statistics

After loading experiential data:

- **Nodes**: 1,238 (up from 1,132)
- **Edges**: 833 (up from 827)
- **Domains**: 7 (added `experiential`)

## Visualizations

See `docs/demo-screenshots/`:
- `09-buildlog-mistakes-ingested.png` - Error class distribution
- `10-cross-domain-mistake-links.png` - Cross-domain linkages

## Implications

1. **Test Terrorist Enhancement**: The 61.6% repeat rate on test errors suggests the test_terrorist persona should be augmented with rules from iterator_visitor_patterns and factory_patterns.

2. **Security Karen Enhancement**: Security errors (0% repeat) are being caught and fixed, but link to implementation_hiding patterns suggests proactive rules could prevent them.

3. **Feedback Loop**: High-repeat errors should trigger confidence boosts on related rules in buildlog's bandit system.

## Reproduction

```bash
# Load mistake emissions into qortex
python3 scripts/load_buildlog_emissions.py

# Or manually:
qortex ingest load /tmp/buildlog_mistakes_manifest.json
qortex ingest load /tmp/cross_domain_links_manifest.json

# Query in Memgraph Lab
MATCH (e:Concept {domain: 'experiential'})-[r]->(p:Concept)
WHERE p.domain <> 'experiential'
RETURN e.name, type(r), p.name, p.domain
```
