# Part 5: Pattern Completion

A doctor asks: "What should I watch for if I prescribe metformin to a patient with kidney disease?"

Your system needs to:

1. Match "metformin" to concepts in the graph
2. Match "kidney disease" to concepts
3. Discover that metformin connects to "lactic acidosis" which connects to "renal impairment"
4. Return rules about all these connected concepts

This isn't keyword search. It's **pattern completion** — starting from partial cues and filling in the rest.

## How your brain does it

Smell your grandmother's perfume. What happens?

You don't just think "perfume." You think of her kitchen. Her voice. The cookies she made. The chair you sat in.

That's pattern completion. A partial cue (the smell) activates a network of associated memories. The whole memory emerges from any part.

Your hippocampus does this constantly. It maintains an index of associations between memory fragments. When one fragment activates, activation spreads through the index until related fragments light up.

## The algorithmic equivalent

Personalized PageRank does the same thing, but on a graph.

Imagine a random walker starting at your query concepts. It takes random steps along edges. Sometimes it teleports back to the starting concepts. After many steps, where does it end up most often?

Nodes that are well-connected to your starting concepts accumulate more visits. They "light up."

```
Query: "metformin kidney disease"

Step 1: Match to graph nodes
   - metformin → Metformin node
   - kidney disease → Renal Impairment node

Step 2: Random walks from these seeds
   - Metformin → connects to → Lactic Acidosis
   - Lactic Acidosis → INCREASES_RISK_WITH → Renal Impairment
   - Renal Impairment was already a seed — reinforced!

Step 3: After many walks, nodes have scores
   - Lactic Acidosis: high score (connects both seeds)
   - Metformin: high score (seed)
   - Renal Impairment: high score (seed)
   - Insulin: medium score (connected to Metformin)
   - Headache: low score (barely connected)

Step 4: Retrieve rules from high-scoring nodes
```

The algorithm found "lactic acidosis" even though the query never mentioned it. It found it by spreading activation through the graph.

## The intuition

Think of it like this:

You're looking for someone at a party. You know two things about them: they're a doctor and they play guitar.

You could ask everyone "are you a doctor who plays guitar?" That's vector similarity — one shot, hope for a direct match.

Or you could ask the doctors "who here plays guitar?" and ask the guitar players "who here is a doctor?" Then find the overlap. That's pattern completion — spread from multiple starting points, see what connects.

The second approach finds the answer even if nobody self-describes as "doctor who plays guitar."

## The code (simplified)

```python
def personalized_pagerank(graph, seeds, damping=0.85, iterations=100):
    # Initialize: probability concentrated on seeds
    scores = {n: 1/len(seeds) if n in seeds else 0 for n in graph.nodes}

    for _ in range(iterations):
        new_scores = {}
        for node in graph.nodes:
            # Score from random walk (follow edges)
            walk_score = sum(
                damping * scores[neighbor] / graph.out_degree(neighbor)
                for neighbor in graph.predecessors(node)
            )
            # Score from teleport (jump back to seeds)
            teleport = (1 - damping) / len(seeds) if node in seeds else 0
            new_scores[node] = walk_score + teleport
        scores = new_scores

    return scores  # Higher score = more relevant to query
```

That's it. The magic is in letting uncertainty propagate through structure.

## What you learned

- Pattern completion starts from partial cues and fills in connections
- Your hippocampus does this for memories; PPR does it for graphs
- Random walks from query nodes find well-connected concepts
- Nodes that bridge multiple query concepts score highest
- This finds relevant concepts even when the query doesn't mention them

## Next

[Part 6: HippoRAG First Principles](part6-hipporag.md) — The full algorithm, from indexing to retrieval.
