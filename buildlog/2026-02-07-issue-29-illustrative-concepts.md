# Build Journal: Fix Illustrative vs Generalizable Concepts (Issue #29)

**Date:** 2026-02-07
**Duration:** ~30 min
**Status:** Complete

---

## The Goal

When qortex ingests a source that uses concrete examples to illustrate patterns (e.g., a textbook chapter on the Observer pattern), the ingestor extracts `BaseballReporter`, `HitEvent`, etc. as peer-level `ConceptNode`s. These are teaching scaffolding, not generalizable principles. Downstream projectors derive rules from them and buildlog ingests noise.

Fix: teach the extraction prompt to classify concepts, then route illustrative ones into `parent.properties["examples"]` instead of standalone nodes.

---

## What We Built

### Architecture

```
LLM Extraction (Anthropic/Ollama)
  │ concept_role: "generalizable" | "illustrative"
  │ illustrates: parent concept name (if illustrative)
  ▼
Ingestor.ingest() — Two-Pass Processing
  │
  ├─ Pass 1: Collect generalizable → ConceptNode list
  │          Queue illustrative → (raw_dict, location) list
  │
  └─ Pass 2: Reconcile illustrative → parent.properties["examples"]
             Orphans → fallback ConceptNode with role tags
```

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| Anthropic prompt | Working | Added concept_role + illustrates fields |
| Ollama prompt | Working | Same update |
| Two-pass ingest | Working | Case-insensitive parent matching |
| StubLLMBackend | Working | Now injectable for testing |
| Test suite | Working | 21 tests, 5 classes |

---

## The Journey

### Phase 1: Prompt + Backend Updates

**What we tried:**
Updated both extraction prompts to ask the LLM to classify each concept with `concept_role` and `illustrates`. Added pass-through in the return dict comprehensions.

**What happened:**
Clean implementation. Both backends now return the two new fields with sensible defaults (`"generalizable"` and `None`).

**Lesson:**
Default values in `c.get("concept_role", "generalizable")` provide backward compatibility for free.

---

### Phase 2: Two-Pass Reconciliation

**What we tried:**
Replaced single-pass concept creation with two-pass: collect generalizable first, then match illustrative to parents via case-insensitive name lookup.

**What happened:**
Initial implementation worked. Discovered during testing that rule `concept_ids` weren't being filtered against surviving concepts — rules could reference absorbed illustrative concepts. Added `valid_concept_ids` filtering.

**The fix:**
```python
valid_concept_ids = {c.id for c in concepts}
concept_ids=[cid for cid in r.get("concept_ids", []) if cid in valid_concept_ids],
```

**Lesson:**
When filtering entities out of a graph, always check all edges/references for dangling pointers.

---

## Test Results

### test_ingest_concept_roles.py

**Command:**
```bash
uv run pytest tests/test_ingest_concept_roles.py -v
```

**Result:** 21/21 passed

### Full Suite

**Command:**
```bash
uv run pytest tests/ -v
```

**Result:** 530 passed, 41 skipped, 0 failures

### Static Analysis

```bash
uv run mypy src/qortex_ingest/     # Success: no issues found
uv run ruff check + format --check  # All checks passed
```

---

## Code Samples

### Two-Pass Concept Processing (base.py)

```python
# Pass 2: Reconcile illustrative concepts → parent properties["examples"]
concept_by_name: dict[str, ConceptNode] = {
    c.name.lower(): c for c in generalizable_concepts
}

for raw, location in illustrative_raw:
    parent_name = raw.get("illustrates")
    parent = concept_by_name.get(parent_name.lower()) if parent_name else None

    if parent is not None:
        if "examples" not in parent.properties:
            parent.properties["examples"] = []
        parent.properties["examples"].append({
            "name": raw["name"],
            "description": raw.get("description", ""),
            "source_location": location,
            "confidence": raw.get("confidence", 1.0),
        })
```

Case-insensitive matching handles LLM variability in casing.

---

## What's Left

- [ ] Phase 2: Graph-based heuristic detection (future, tracked in #29)
- [ ] Retroactive reconciliation for examples appearing before parents across chunks

---

## Improvements

### Architectural

- Two-pass processing is a clean pattern for any "collect then reconcile" problem in extraction pipelines
- Case-insensitive matching should be the default for any LLM-generated name lookups

### Workflow

- Writing the test first for the rule filtering edge case would have caught the dangling concept_id issue before implementation

### Domain Knowledge

- LLM extraction prompts benefit from explicit classification taxonomy — "generalizable vs illustrative" is a useful frame beyond this specific feature
- `ConceptNode.properties["examples"]` was already in the schema but never populated — check existing fields before adding new ones

---

## Files Changed

```
src/qortex_ingest/
├── backends/
│   ├── anthropic.py    # Updated extraction prompt + return fields
│   └── ollama.py       # Same prompt update
└── base.py             # Two-pass ingest, StubLLMBackend injectable

tests/
└── test_ingest_concept_roles.py  # 21 tests across 5 classes (NEW)
```

---

*Next entry: Phase 2 graph-based heuristic detection, or buildlog gauntlet review*
