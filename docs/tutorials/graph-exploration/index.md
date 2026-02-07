---
title: "Part 3: Exploring Your Knowledge Graph"
---

# Part 3: Exploring Your Knowledge Graph

**Estimated time**: ~90 minutes

**Prerequisites**: You've ingested at least one chapter using qortex (Part 2 covers this). You have a running Memgraph instance with data in it.

---

## The Arc

You built a knowledge graph. The CLI said "47 concepts, 83 edges, 7 rules." Cool.

Now what?

This series takes you from "I have a graph" to "I understand my graph." You'll learn to query it, discover its structure, find the hubs, trace paths, and hit the wall that sets up Part 4.

![a-3-1-what-do-i-have-br-naive](../../images/diagrams/index-1-a-3-1-what-do-i-have-br-naive.svg)

| # | Chapter | Depth | What You'll Learn |
|---|---------|-------|-------------------|
| 3.1 | [What Do I Actually Have?](3.1-what-do-i-have.md) | Deep | MATCH/RETURN, property access, CONTAINS, toLower(), LIMIT |
| 3.2 | [Counting and Grouping](3.2-counting-and-grouping.md) | Light | COUNT, GROUP BY, ORDER BY, aggregation |
| 3.3 | [Following the Connections](3.3-following-the-connections.md) | Deep | Relationship patterns, direction, variable-length paths |
| 3.4 | [Filtering by Pattern](3.4-filtering-by-pattern.md) | Light | WHERE NOT EXISTS, existential subqueries |
| 3.5 | [Finding the Hubs](3.5-finding-the-hubs.md) | Deep | Degree centrality, in/out-degree, why hubs matter |
| 3.6 | [Paths and Distances](3.6-paths-and-distances.md) | Deep | shortestPath, allShortestPaths, path inspection |

## The Running Example

Every chapter works with the same graph: the concepts and relationships extracted from Chapter 5 of *Software Design for Python Programmers* (the Encapsulation chapter). A few supplemental edges have been added to complete causal chains for teaching purposes. These are noted explicitly where they appear.

## Where This Leads

By the end of 3.6, you'll know your graph inside out. You'll also be stuck: multiple paths of the same length, no way to rank them. That's the setup for Part 4, where we bring in graph algorithms (Personalized PageRank) to do what queries alone cannot.
