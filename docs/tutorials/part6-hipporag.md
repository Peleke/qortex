# Part 6: HippoRAG First Principles

You've seen the pieces. Now let's put them together.

HippoRAG is a retrieval system that mimics how your hippocampus indexes and retrieves memories. Instead of treating documents as isolated vectors, it builds a graph of associations and retrieves by spreading activation.

## The two phases

**Offline (Indexing)**: Build the knowledge graph from your documents.

**Online (Retrieval)**: Given a query, spread activation through the graph to find relevant passages.

## Phase 1: Indexing

```
Documents
    ↓
[Extract triples via LLM]
    ↓
("Metformin", "TREATS", "Diabetes")
("Metformin", "RISK_WITH", "Renal Impairment")
    ↓
[Build knowledge graph]
    ↓
Nodes: Metformin, Diabetes, Renal Impairment, ...
Edges: TREATS, RISK_WITH, ...
    ↓
[Link nodes to source passages]
    ↓
Metformin node → Passage 1, Passage 7
    ↓
[Embed nodes for matching]
    ↓
Ready for retrieval
```

The key insight: **extraction creates discrete representations**. "Metformin" is its own node, not a point in embedding space that might blur with similar drugs. This is pattern separation: distinct concepts get distinct representations.

## Phase 2: Retrieval

```
Query: "metformin and kidney problems"
    ↓
[Extract query entities]
    ↓
["metformin", "kidney"]
    ↓
[Match to graph nodes via embedding similarity]
    ↓
Metformin node (0.95 match)
Renal Impairment node (0.87 match for "kidney")
    ↓
[Run Personalized PageRank from matched nodes]
    ↓
High scores: Metformin, Renal Impairment, Lactic Acidosis, ...
    ↓
[Rank passages by sum of their nodes' scores]
    ↓
Passage 3 (contains Metformin + Lactic Acidosis): score 1.7
Passage 1 (contains Metformin only): score 0.9
    ↓
[Return top passages to LLM]
```

The retrieval found Lactic Acidosis (a crucial concept for this query) even though the query never mentioned it. Pattern completion discovered the connection.

## The brain analogy

| Brain Component | HippoRAG | Function |
|-----------------|----------|----------|
| Neocortex | Documents + LLM | Stores actual content |
| Hippocampal index | Knowledge graph | Network of associations |
| Pattern separation | Triple extraction | Distinct representations |
| Pattern completion | Personalized PageRank | Spread from partial cues |

The hippocampus doesn't store memories. It indexes them. The neocortex stores the content; the hippocampus stores the associations between fragments.

HippoRAG does the same. The LLM and documents are the "neocortex." The knowledge graph is the "hippocampal index."

## Why it beats standard RAG

| Aspect | Standard RAG | HippoRAG |
|--------|--------------|----------|
| Representation | Dense vectors | Graph + embeddings |
| Retrieval | Nearest neighbor | PPR on graph |
| Multi-hop | Fails (needs iteration) | Single pass |
| Cost for multi-hop | 10-20x more LLM calls | Same as single-hop |
| Speed for multi-hop | 6-13x slower | Same as single-hop |

Standard RAG asks: "What documents look like this query?"

HippoRAG asks: "What concepts connect to this query's concepts?"

The second question is the right question for multi-hop reasoning.

## What qortex provides

qortex is the **indexing layer**. It builds the knowledge graph that HippoRAG retrieves from:

1. **Ingest**: Documents become structured manifests
2. **Store**: Concepts and edges go into GraphBackend
3. **Project**: Rules for downstream consumers (buildlog tests them)
4. **Retrieve** (Phase 2): PPR-based pattern completion

The scaffolding for retrieval exists in `src/qortex/hippocampus/`. Full implementation is Phase 2 of the roadmap.

## What you learned

- HippoRAG has two phases: offline indexing, online retrieval
- Indexing extracts triples and builds a knowledge graph
- Retrieval matches query to nodes, then spreads activation via PPR
- Pattern completion finds relevant concepts the query never mentioned
- qortex provides the indexing layer; retrieval is Phase 2

## Next steps

Ready to use this? Head to the [Quick Start](../getting-started/quickstart.md) to ingest your first content, build a graph, and project rules.
