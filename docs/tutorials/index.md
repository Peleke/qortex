# Theory: From RAG to HippoRAG

Why does your retrieval system fail when the answer spans multiple documents?

The short version: vector similarity finds documents that *look like* your query. But sometimes you need documents that are *connected to* your query through concepts the embedding never learned to represent.

This is the multi-hop problem. And the fix is surprisingly intuitive once you see it.

## The idea in 30 seconds

Standard RAG embeds documents as vectors and retrieves by similarity. Works great for single-hop questions ("What is X?"). Falls apart for multi-hop questions ("What happens when X interacts with Y in context Z?").

The fix: build a graph of concepts and relationships alongside your embeddings. When a query arrives, match it to concepts in the graph, then *spread activation* through connected nodes. Retrieve documents linked to the high-scoring nodes.

That's HippoRAG, named after your hippocampus, which does exactly this for memory retrieval.

qortex builds the graph. These tutorials explain why it works.

## The series

| Tutorial | What it covers |
|----------|----------------|
| [The Multi-Hop Problem](part1-multi-hop-problem.md) | Why similarity isn't association |
| [Knowledge Graphs 101](part2-knowledge-graphs.md) | Concepts, edges, semantic types |
| [The Projection Pipeline](part3-projection-pipeline.md) | Graph → Rules via Source → Enricher → Target |
| [The Consumer Loop](part4-consumer-loop.md) | Rules as hypotheses; measuring what works |
| [Pattern Completion](part5-pattern-completion.md) | Personalized PageRank and spreading activation |
| [HippoRAG First Principles](part6-hipporag.md) | The full algorithm: index with graphs, retrieve with PPR |

## What these tutorials are (and aren't)

These are intentionally light. You'll get working intuition, enough to use qortex and understand what it's doing. You won't get rigorous math or deep theory.

For the full treatment (probability from first principles, the linear algebra behind PageRank, information geometry for embeddings), there's [Aegir](https://github.com/Peleke/aegir). It's an in-progress curriculum I'm building alongside my own learning journey. Think of it as a super-notebook that'll become a proper book over the next year or two.

## Prerequisites

- Python basics (functions, classes, dicts)
- Comfort with `pip install` and running scripts
- No ML/AI background required

## Ready?

Start with [The Multi-Hop Problem](part1-multi-hop-problem.md), a 2am hospital story about a two-million-dollar system that couldn't answer a simple question.
