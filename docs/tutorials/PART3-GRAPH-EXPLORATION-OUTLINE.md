# Part 3: Exploring Your Knowledge Graph

> **Style**: One cohesive 90-minute narrative, structured with natural breakpoints for 6 chapters.
> **Depth**: Deep on 3.1, 3.3, 3.5, 3.6. Light on 3.2, 3.4 (noted explicitly).
> **Data**: Real Ch5 extraction + plausible supplements (noted where added).
> **Voice**: Aegir style (story-first, code-before-formula, jargon-earning, visual hooks).

---

## The Arc

```
OPEN (3.1): "wtf why only 3 nodes" - naive query fails
    ↓
JOURNEY (3.2-3.4): learn to query properly, discover structure
    ↓
CLIMAX (3.5): finally answer "what's the hub?" - but it's not what we expected
    ↓
CLOSE (3.6): "how do I traverse FROM a concept?" - sets up PPR
```

---

## Chapter 3.1: "What Do I Actually Have?" [DEEP]

### Opening Hook

> You just ingested Chapter 5 of "Software Design for Python Programmers." The CLI said "47 concepts, 83 edges, 7 rules." Cool. Now you want to find everything about setters.
>
> ```cypher
> MATCH (c:Concept) WHERE c.name CONTAINS 'setter' RETURN c
> ```
>
> 3 nodes. That's it.
>
> But you KNOW that chapter had 15 pages on dangerous setters, property setters, the whole setter antipattern. Where did everything go?

### What They Learn

1. **Basic MATCH/RETURN** - The query skeleton
2. **Property access** - `c.name`, `c.description`, `c.domain`
3. **CONTAINS vs exact match** - Why their query was too narrow
4. **Case sensitivity** - `toLower()` for robust matching
5. **LIMIT and ordering** - Not drowning in output

### The Aha Moments

1. "Oh, I was only searching `name`, not `description`"
2. "The concepts are there, they just have different names (DangerousSetters, PropertySetters, SetterMethod)"
3. "There are 47 concepts... but wait, how many actually have edges?"

### Key Queries

```cypher
-- Their first attempt (fails)
MATCH (c:Concept) WHERE c.name CONTAINS 'setter' RETURN c

-- Fixed: search description too
MATCH (c:Concept)
WHERE toLower(c.name) CONTAINS 'setter'
   OR toLower(c.description) CONTAINS 'setter'
RETURN c.name, c.description

-- Discovery: how many concepts total?
MATCH (c:Concept) RETURN count(c)

-- Discovery: how many have edges?
MATCH (c:Concept)-[r]-() RETURN count(DISTINCT c)
```

### Closing Bridge

> 47 concepts. But only 32 have edges. That means 15 are orphans. And you still don't know what KINDS of relationships exist. Let's count.

### Visual Hook

Bar chart: concepts with edges vs orphans.

---

## Chapter 3.2: "Counting and Grouping" [LIGHT]

> **Note**: This chapter is intentionally lighter. We cover aggregation basics to support later chapters, not to master Cypher aggregations.

### Opening Hook

> You know you have 83 edges. But what kinds? Is it mostly REQUIRES? CONTRADICTS? SIMILAR_TO?

### What They Learn

1. **COUNT and GROUP BY** - Implicit grouping in Cypher
2. **ORDER BY** - Sorting results
3. **Aggregation on properties** - Counting by relation_type

### Key Queries

```cypher
-- Count edges by type
MATCH ()-[r]->()
RETURN type(r) as relation_type, count(r) as count
ORDER BY count DESC

-- Count concepts by domain
MATCH (c:Concept)
RETURN c.domain, count(c)
```

### The Aha

> REQUIRES is 35 edges. IMPLEMENTS is 18. CONTRADICTS is only 4. So your graph is mostly about dependencies, not conflicts.

### Closing Bridge

> Numbers are nice, but you want to SEE the connections. Let's follow an edge.

---

## Chapter 3.3: "Following the Connections" [DEEP]

### Opening Hook

> You found the DangerousSetters concept. Now: what does it connect to?

### What They Learn

1. **Relationship patterns** - `(a)-[r]->(b)`, `(a)<-[r]-(b)`, `(a)-[r]-(b)`
2. **Direction matters** - Outgoing vs incoming vs both
3. **Relationship properties** - `r.confidence`, `r.source_text`
4. **Variable-length paths** - `*1..3` for multi-hop
5. **Path objects** - Storing and inspecting paths

### Key Queries

```cypher
-- Direct connections from DangerousSetters
MATCH (ds:Concept {name: 'Dangerous Setters'})-[r]->(target)
RETURN ds.name, type(r), target.name

-- What points TO DangerousSetters?
MATCH (source)-[r]->(ds:Concept {name: 'Dangerous Setters'})
RETURN source.name, type(r)

-- Two-hop paths
MATCH path = (ds:Concept {name: 'Dangerous Setters'})-[*1..2]->(target)
RETURN path

-- Find path to a specific target
MATCH path = shortestPath(
  (ds:Concept {name: 'Dangerous Setters'})-[*]-(inv:Concept {name: 'Class Invariant'})
)
RETURN path
```

### The Aha Moments

1. "DangerousSetters CHALLENGES Encapsulation but ENABLES InvalidState"
2. "The path from Setter to ClassInvariant is only 2 hops!"
3. "Wait, there's a 4-hop path to ProductionBug through InvalidState..."

### Visual Hook

Graph visualization of the DangerousSetters neighborhood (2 hops out).

### Closing Bridge

> You can follow paths. But what if you want to find concepts that DON'T have certain connections? That's filtering by pattern.

---

## Chapter 3.4: "Filtering by Pattern" [LIGHT]

> **Note**: Intentionally lighter. Existential subqueries are powerful but niche for exploration.

### Opening Hook

> Which concepts are islands? No incoming, no outgoing edges. These are coverage gaps in your extraction.

### What They Learn

1. **WHERE NOT EXISTS** - Negating patterns
2. **Existential subqueries** - `EXISTS { MATCH ... }`
3. **Counting subqueries** - `COUNT { MATCH ... }`

### Key Queries

```cypher
-- Find orphan concepts (no edges at all)
MATCH (c:Concept)
WHERE NOT EXISTS { MATCH (c)-[]-() }
RETURN c.name, c.description

-- Find concepts with no REQUIRES edges
MATCH (c:Concept)
WHERE NOT EXISTS { MATCH (c)-[:REQUIRES]->() }
RETURN c.name

-- Count how many edges each concept has
MATCH (c:Concept)
RETURN c.name, COUNT { MATCH (c)-[]-() } as degree
ORDER BY degree DESC
```

### The Aha

> 8 orphan concepts. That's 17% of your graph with no connections. Either they're truly standalone, or your extraction missed relationships.

### Closing Bridge

> You've found orphans and counted degrees. Speaking of degree... which concept has the MOST connections?

---

## Chapter 3.5: "Finding the Hubs" [DEEP]

### Opening Hook

> Pop quiz: what's the most connected concept in your Chapter 5 graph? You'd guess Encapsulation, right? It's literally the chapter title.
>
> Let's find out.

### What They Learn

1. **Degree centrality** - Count of connections
2. **In-degree vs out-degree** - Direction matters for directed graphs
3. **Why hubs matter** - Information flow, structural importance
4. **The surprise** - Expectations vs reality

### Key Queries

```cypher
-- Total degree (in + out)
MATCH (c:Concept)
OPTIONAL MATCH (c)-[out]->()
OPTIONAL MATCH (c)<-[in]-()
RETURN c.name,
       count(DISTINCT out) as out_degree,
       count(DISTINCT in) as in_degree,
       count(DISTINCT out) + count(DISTINCT in) as total_degree
ORDER BY total_degree DESC
LIMIT 10

-- Just the top hub
MATCH (c:Concept)-[r]-()
RETURN c.name, count(r) as degree
ORDER BY degree DESC
LIMIT 1
```

### The Aha Moment

> **Properties**: 23 connections.
> **Encapsulation**: 8 connections.
>
> Wait, what?!
>
> Properties is the hub, not Encapsulation. Why? Because Properties is the IMPLEMENTATION mechanism. Every other concept (getters, setters, validation, access control) connects THROUGH Properties to achieve Encapsulation.
>
> The hub isn't the abstract goal. It's the concrete mechanism.

### The Deeper Insight

> This is why centrality matters for retrieval. If someone asks about "data hiding," you might match to Encapsulation. But Properties is more structurally important: it's the crossroads where concepts meet.

### Visual Hook

Degree distribution bar chart. Properties towers over everything else.

### Closing Bridge

> You found the hub. But here's a harder question: what are ALL the paths from DangerousSetters to ProductionBug? And which path matters most?

---

## Chapter 3.6: "Paths and Distances" [DEEP]

### Opening Hook

> The setup: you want to understand how DangerousSetters leads to ProductionBug. You know there's a path (you found it in 3.3). But is there only ONE path? Are there multiple routes? Which one is the "main" path?

### What They Learn

1. **All paths vs shortest path** - `allShortestPaths` vs `shortestPath`
2. **Path length** - `length(path)`
3. **Path inspection** - `nodes(path)`, `relationships(path)`
4. **Multiple paths** - There might be more than one route
5. **The ranking problem** - Which path matters most? (Unanswered here)

### Key Queries

```cypher
-- Shortest path
MATCH path = shortestPath(
  (ds:Concept {name: 'Dangerous Setters'})-[*]-(pb:Concept {name: 'Production Bug'})
)
RETURN path, length(path)

-- ALL shortest paths (there might be ties)
MATCH path = allShortestPaths(
  (ds:Concept {name: 'Dangerous Setters'})-[*]-(pb:Concept {name: 'Production Bug'})
)
RETURN path, length(path)

-- All paths up to length 5
MATCH path = (ds:Concept {name: 'Dangerous Setters'})-[*1..5]-(pb:Concept {name: 'Production Bug'})
RETURN path, length(path)
ORDER BY length(path)

-- Inspect what's in a path
MATCH path = shortestPath(
  (ds:Concept {name: 'Dangerous Setters'})-[*]-(pb:Concept {name: 'Production Bug'})
)
RETURN [n IN nodes(path) | n.name] as concept_names,
       [r IN relationships(path) | type(r)] as relation_types
```

### The Aha Moments

1. "There are 3 different shortest paths, all length 4"
2. "One goes through InvalidState, one through Encapsulation, one through Properties"
3. "But which one is the 'real' causal chain? They're all the same length..."

### The Cliffhanger

> You have paths. You have lengths. But you can't RANK them.
>
> If I start at DangerousSetters and ask "what's most relevant?", I need more than path length. I need some way to weight the connections, to spread outward from my starting point and see what "lights up."
>
> That's not a query. That's an algorithm. It's called Personalized PageRank.
>
> And it's exactly what the hippocampus does when you smell your grandmother's perfume and suddenly remember her kitchen.

### Visual Hook

Multiple paths visualized, all same length, question mark over "which matters?"

---

## Supplements Needed

Based on real Ch5 extraction, we may need to add:

| Supplement | Why | Plausibility |
|------------|-----|--------------|
| `ProductionBug` concept | Endpoint for path examples | High: natural consequence of InvalidState |
| Edge: `InvalidState -[:CAUSES]-> ProductionBug` | Complete the causal chain | High: direct implication |
| Edge: `DangerousSetters -[:ENABLES]-> InvalidState` | Key teaching example | High: explicit in book |

Note these explicitly in the notebook: "We've added a few edges to complete the causal chain for teaching purposes."

---

## Falsifiable Claims

| Chapter | Claim | Test |
|---------|-------|------|
| 3.1 | Searching name-only misses >50% of relevant concepts | Query both, compare counts |
| 3.2 | REQUIRES is the dominant relation type (>30%) | Aggregation query |
| 3.3 | Path from DangerousSetters to ClassInvariant exists in ≤3 hops | shortestPath query |
| 3.5 | The top hub is NOT the chapter title concept | Degree query |
| 3.6 | Multiple paths of same length exist between key concepts | allShortestPaths query |

---

## Jargon-Earning Order

| Term | Introduced | Earned By |
|------|------------|-----------|
| MATCH/RETURN | 3.1 | "here's how you ask questions" |
| edge/relationship | 3.1 | "lines connecting nodes" first |
| aggregation | 3.2 | "counting and grouping" |
| path | 3.3 | "following the arrows" |
| degree | 3.5 | "number of connections" |
| hub | 3.5 | "the concept everything connects to" |
| centrality | 3.5 | after hub is intuitive |
| PageRank | 3.6 cliffhanger | NOT taught, just named as teaser |

---

## Next Steps

1. **Verify Ch5 extraction** - Check actual concepts/edges we have
2. **Identify gaps** - What supplements are actually needed
3. **Draft 3.1** - Full notebook following aegir style
4. **Iterate** - Get feedback, refine voice
5. **Complete remaining chapters**

---

## Cross-References

- **Part 1-6 Redesign**: `REDESIGN-PLAN.md` (Dangerous Setter minitheory)
- **Aegir Voice Guide**: `/Users/peleke/Documents/Projects/aegir/.claude/skills/lesson-generator/SKILL.md`
- **Existing graph-analytics**: `docs/tutorials/graph-analytics/` (Norse theming, different audience)
