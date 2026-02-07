# Qortex + Buildlog Integration

[Buildlog](https://peleke.github.io/buildlog/) is qortex's first intentional consumer. It uses projected rules to power AI code review personas, and emits structured feedback that flows back into the knowledge graph.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INTEGRATION FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   QORTEX (Knowledge Graph)              BUILDLOG (Agent System)     │
│   ════════════════════════              ═══════════════════════     │
│                                                                     │
│   Book chapters, docs, code             AI code review personas     │
│          │                                       ▲                  │
│          ▼                                       │                  │
│   Concept nodes + edges                 Gauntlet reviewers          │
│          │                                       ▲                  │
│          ▼                                       │                  │
│   Projected rules ──────────────────────> Seed files                │
│          ▲                                       │                  │
│          │                                       ▼                  │
│   Experiential domain <─────────────── Mistake emissions            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
~/.qortex/
  seeds/
    pending/                    # Outbox: qortex writes here
      qortex_impl_hiding.yaml
      qortex_observer.yaml

~/.buildlog/
  seeds/                        # Inbox: buildlog copies seeds here
    qortex_impl_hiding.yaml
  emissions/
    pending/                    # Outbox: buildlog writes here
      mistake_manifest_*.json
      learned_rules_*.json
      reward_signal_*.json
    signal.jsonl                # Append-only event log
  buildlog.db                   # SQLite database
```

---

## Forward Flow: Rules to Agents

### Step 1: Project Rules

```bash
uv run qortex project buildlog \
  --domain implementation_hiding \
  -p qortex_impl_hiding \
  -o ~/.qortex/seeds/pending/qortex_impl_hiding.yaml
```

Options:
- `--domain`: Which knowledge domain to project
- `-p, --persona`: Name for the buildlog persona
- `-o, --output`: Output path (use `~/.qortex/seeds/pending/` for interop)

### Step 2: Ingest into Buildlog

```bash
cd path/to/buildlog-template
uv run buildlog ingest-seeds
```

Output:
```
[qortex] 1 ingested
  ✓ qortex_impl_hiding.yaml (qortex_impl_hiding, 135 rules)
```

### Step 3: Verify

```bash
# List ingested seeds
ls .buildlog/seeds/

# Check active rules
uv run buildlog_gauntlet_rules()
```

---

## Seed File Format

```yaml
persona: qortex_impl_hiding
version: 1
rules:
  - rule: "Make every instance variable private unless external access is required."
    category: encapsulation
    context: "When designing class interfaces"
    antipattern: "Exposing internal state via public attributes"
    rationale: "Encapsulation prevents coupling to implementation details"
    tags: [implementation_hiding, encapsulation, data_hiding]
    provenance:
      id: "impl_hiding:rule:0"
      domain: implementation_hiding
      derivation: explicit        # or "derived" for edge-projected rules
      confidence: 0.9
      source_concepts: ["ch5:encapsulation", "ch5:data_hiding"]
      template_id: null           # non-null for derived rules
metadata:
  source: qortex
  source_version: "0.1.0"
  projected_at: "2026-02-07T00:24:00+00:00"
  rule_count: 135
```

### Key Fields

| Field | Purpose |
|-------|---------|
| `persona` | Flat string, becomes the persona name in buildlog |
| `version` | Integer, for schema versioning |
| `rule` | The actual rule text (not `text`) |
| `category` | Used for grouping/filtering in gauntlet |
| `provenance` | Opaque to buildlog, used for attribution analysis |
| `provenance.confidence` | Feeds into buildlog's bandit priors |

---

## Backward Flow: Mistakes to Graph

### Emission Types

Buildlog emits structured data to `~/.buildlog/emissions/pending/`:

**mistake_manifest_*.json**
```json
{
  "source_id": "buildlog:abc123",
  "domain": "experiential",
  "concepts": [{
    "name": "mistake:test-error-20260207",
    "properties": {
      "error_class": "test",
      "description": "Missing edge case test",
      "was_repeat": true,
      "semantic_hash": "abc123"
    }
  }],
  "metadata": {
    "project_id": "abc123",
    "mistake_id": "mistake-test-20260207"
  }
}
```

**reward_signal_*.json**
```json
{
  "outcome": "accepted",
  "reward_value": 1.0,
  "rules_active": ["qortex_impl_hiding:rule:0", "qortex_impl_hiding:rule:5"],
  "session_id": "session-20260207"
}
```

### Loading into Qortex

```bash
# Transform emissions to qortex manifest format
python scripts/load_buildlog_emissions.py

# Load directly
uv run qortex ingest load /tmp/buildlog_mistakes_manifest.json
```

### Cross-Domain Linking

Link mistakes to relevant design patterns:

```cypher
-- Find test errors that challenge algorithm encapsulation
MATCH (m:Concept {domain: 'experiential'})
WHERE m.name CONTAINS 'Test Errors'
MATCH (p:Concept {domain: 'iterator_visitor_patterns', name: 'Algorithm Encapsulation'})
CREATE (m)-[:CHALLENGES]->(p)
```

---

## Buildlog's Reward System

Buildlog uses a Thompson Sampling bandit to select which rules to surface:

### Implicit Feedback (mistakes)
- When a mistake is logged, active rules get `reward=0`
- Beta(α, β) → Beta(α, β+1) — rule becomes less likely to be selected

### Explicit Feedback (rewards)
- `accepted` → reward=1.0, all active rules get credit
- `rejected` → reward=0.0, all active rules get penalized
- `revision` → reward = 1.0 - revision_distance, partial credit

### Attribution (coming soon)
- `corrected_by_rule` on mistakes → direct positive attribution
- Per-rule, per-issue matching instead of session-level smearing

---

## SQLite Schema

Buildlog stores structured data in `~/.buildlog/buildlog.db`:

### review_learnings
```sql
CREATE TABLE review_learnings (
    id TEXT PRIMARY KEY,
    rule TEXT NOT NULL,
    category TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_reinforced TEXT NOT NULL,
    reinforcement_count INTEGER DEFAULT 1,
    contradiction_count INTEGER DEFAULT 0
);
```

### mistakes
```sql
CREATE TABLE mistakes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    error_class TEXT NOT NULL,
    was_repeat INTEGER DEFAULT 0,
    corrected_by_rule TEXT  -- direct attribution when populated
);
```

### reward_events
```sql
CREATE TABLE reward_events (
    id TEXT PRIMARY KEY,
    outcome TEXT NOT NULL,
    reward_value REAL NOT NULL,
    rules_active TEXT  -- JSON array of rule IDs
);
```

---

## Analysis

See `notebooks/mistake_time_series.ipynb` for:

1. **Rule strength trends**: reinforcement_count vs contradiction_count by category
2. **Repeat rate analysis**: Which error classes have persistent blind spots
3. **Intervention effects**: Before/after comparison around seed ingestion dates
4. **Attribution**: When `corrected_by_rule` is populated, measure per-rule effectiveness

### Quick Analysis

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect(Path.home() / ".buildlog" / "buildlog.db")

# Rule strength by category
learnings = pd.read_sql_query("""
    SELECT category,
           COUNT(*) as rules,
           SUM(reinforcement_count) as reinforcements,
           SUM(contradiction_count) as contradictions
    FROM review_learnings
    GROUP BY category
    ORDER BY reinforcements DESC
""", conn)
```

---

## Case Study

For a complete walkthrough of both flows with screenshots, see:
- [Case Study Overview](../tutorials/full-loop-overview.md)
- [Part I: Forward Flow](../tutorials/full-loop-forward.md)
- [Part II: Backward Flow](../tutorials/full-loop-backward.md)
