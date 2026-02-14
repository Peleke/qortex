# Bug Fix Plan: Relations Extraction Returns Empty

**Status:** ✅ FIXED
**Priority:** IMMEDIATE (blocks tutorial demos)
**Found via:** Dogfooding - ingested ch05, got 409 concepts, 7 rules, 0 relations

---

## The Problem

```
Concepts extracted: 409 ✓
Relations extracted: 0 ✗
Rules extracted: 7 ✓
```

Relations are the whole point of a knowledge GRAPH. Without edges, we have a knowledge LIST.

---

## Investigation Plan

### Step 1: Trace the extraction flow
- [ ] Read `src/qortex_ingest/backends/anthropic.py` - find `extract_relations()`
- [ ] Check the prompt - is it asking for relations?
- [ ] Check the response parsing - is it handling the JSON correctly?
- [ ] Add debug logging to see what the LLM returns

### Step 2: Identify the bug
Likely culprits:
1. **Prompt issue** - not asking for relations, or asking wrong format
2. **Parsing issue** - LLM returns relations but we drop them
3. **Enum mismatch** - relation_type strings don't match RelationType enum
4. **Empty concepts list** - passing empty list to extract_relations()

### Step 3: Fix for demo (immediate)
- [ ] Fix the extraction backend
- [ ] Re-run ingestion on ch05 with fixed code
- [ ] Verify edges appear in Memgraph

### Step 4: Retroactive fix capability
Build a `qortex ingest relations` subcommand that:
- Takes a manifest JSON (concepts already extracted)
- Takes the source text
- Calls only `extract_relations()` with existing concepts
- Merges new edges into the manifest
- Can reload to graph

This lets us fix past extractions without re-running full LLM extraction.

---

## Files to Investigate

1. `src/qortex_ingest/backends/anthropic.py` - the extraction backend
2. `src/qortex_ingest/base.py` - the Ingestor.ingest() flow
3. `src/qortex/core/models.py` - RelationType enum

---

## Debug Commands

```python
# Test extract_relations directly
from qortex.ingest.backends.anthropic import AnthropicExtractionBackend
from qortex.core.models import ConceptNode

backend = AnthropicExtractionBackend()
concepts = [
    ConceptNode(id="test:enc", name="Encapsulation", description="...", domain="test", source_id="test"),
    ConceptNode(id="test:plk", name="Principle of Least Knowledge", description="...", domain="test", source_id="test"),
]
text = "Encapsulation requires the Principle of Least Knowledge to be effective."

relations = backend.extract_relations(concepts, text)
print(relations)  # Should NOT be empty
```

---

## Success Criteria

1. Re-running ch05 ingestion produces edges
2. `MATCH ()-[r]->() RETURN type(r), count(r)` returns non-zero
3. Tutorial screenshots show connected graph
4. `qortex ingest relations` command exists for retroactive fixes
