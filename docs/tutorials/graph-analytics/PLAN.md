# The Feast of Graphs - Tutorial Series Plan
NOTE: Well done overall, this is grand. Let's tighten narrative and such and a few tightening suggestions below to make it a bit more in line with actual myth but like. Fucking great.

**Status:** Planning
**Author:** Aegir (the sea god who throws the best parties)
**Target:** Developers who have data in Memgraph and want to actually *use* it

---

## Series Overview

### The Narrative

You've ingested a chapter on software design into a knowledge graph. Now Aegir invites you to his hall to learn how to *interrogate* it - from "what the fuck is this?" to "holy shit, I can derive new rules from concept relationships."

### The Tone

Irreverent but educational. We're not dumbing anything down - we're making it accessible by being honest about what's actually happening. No corporate speak. No "leverage synergies." Just graphs, queries, and the occasional mead joke.

### The Data

All examples use real data from our `implementation_hiding` domain:
- ~238 concepts extracted from Chapter 5 of "Software Design for Python Programmers"
- 7 explicit rules about encapsulation, properties, and information hiding
- Edges showing REQUIRES, SUPPORTS, IMPLEMENTS, etc. relationships

### Prerequisites

- `qortex infra up` (Memgraph running)
- Memgraph Lab at http://localhost:3000
- Data ingested via `qortex ingest`

---

## Part Structure

### Part 1: "Welcome to the Hall"

FEEDBACK: If we're going to go this route. Actually? Frame it better: Look up the Har, Jafnhar, and Trhdji setup from Gylfaginning. Let's not be too heavy-handed, now, but a nod in that direction that's a bit tong-ein-cheek is fine. BUT. It is ... Smart, given Odin, knowledge, ya know.
**Aegir's Hook:**
> "Pull up a chair, mortal. You've got data in my graph, but you're staring at it like a cat watching a fish tank. Let me show you how to reach in and grab what you need."

**Learning Objectives:**
1. Understand what nodes and edges actually are (not tables in disguise)
2. Navigate Memgraph Lab without panicking
3. Run your first query and see results

**Key Queries:**
```cypher
-- What's in the hall?
MATCH (n) RETURN labels(n), count(n);

-- Meet the concepts
MATCH (c:Concept {domain: "implementation_hiding"})
RETURN c.name, c.description
LIMIT 10;

-- See the shape of knowledge
MATCH (c:Concept {domain: "implementation_hiding"})
RETURN c;
```

**Screenshot Moments:**
1. Memgraph Lab connection screen (Quick Connect)
2. Query editor with first query
3. Full graph visualization (force-directed layout)
4. Single node inspection panel

---

### Part 2: "The Tongue of Queries"

**Aegir's Hook:**
FEEDBACK: Do something with runes, bindrunes, sigils...
> "Cypher looks like ASCII art had a baby with SQL. That's because it is. The patterns you draw ARE the query. Let me teach you to speak graph."

**Learning Objectives:**
1. Write MATCH patterns that find what you need
2. Filter with WHERE, project with RETURN
3. Understand that `(a)-[r]->(b)` is both a pattern AND a query

**Key Queries:**
```cypher
-- Find concepts by name (SQL mindset)
MATCH (c:Concept)
WHERE c.name CONTAINS "Encapsulation"
RETURN c.name, c.description;

-- Find what Encapsulation connects to (Graph mindset)
MATCH (enc:Concept {name: "Encapsulation"})-[r]-(other:Concept)
RETURN enc.name, type(r), other.name;

-- Count connections per concept
MATCH (c:Concept {domain: "implementation_hiding"})
OPTIONAL MATCH (c)-[r]-()
RETURN c.name, count(r) AS connections
ORDER BY connections DESC;
```

**Screenshot Moments:**
1. Query with WHERE clause and results
2. Pattern match showing relationships
3. Aggregation results as table

---

### Part 3: "Following the Thread"

**Aegir's Hook:**
FEEDBACK: some sort of note or allusion to Ringerike knotwork
> "A single hop is for tourists. Real knowledge seekers follow the thread - two hops, three hops, until you find what you didn't know you were looking for. This is where graphs earn their keep."

**Learning Objectives:**
1. Write variable-length path queries (`[*1..3]`)
2. Find indirect relationships between concepts
3. Trace reasoning chains through the graph

**Key Queries:**
```cypher
-- Direct neighbors
MATCH (c:Concept {name: "Encapsulation"})-[r]-(neighbor)
RETURN neighbor.name, type(r);

-- Two hops out - what's connected to what's connected?
MATCH path = (c:Concept {name: "Encapsulation"})-[*1..2]-(distant)
WHERE c <> distant
RETURN DISTINCT distant.name, length(path) AS hops;

-- Find ALL paths between two concepts
MATCH path = (a:Concept)-[*1..4]-(b:Concept)
WHERE a.name = "Encapsulation" AND b.name CONTAINS "Principle"
RETURN [n IN nodes(path) | n.name] AS concept_chain,
       [r IN relationships(path) | type(r)] AS relationship_chain;

-- The shortest path (if it exists)
MATCH path = shortestPath(
  (a:Concept {name: "Encapsulation"})-[*]-(b:Concept {name: "Law of Demeter"})
)
RETURN [n IN nodes(path) | n.name] AS path;
```

**Screenshot Moments:**
1. Multi-hop query results showing paths
2. Graph visualization with path highlighted
3. Shortest path visualization

**This is the "aha" moment:** Multi-hop queries are why graphs exist. You can't do this efficiently in SQL without recursive CTEs that make your DBA cry.

---

### Part 4: "Measuring Worth"

**Aegir's Hook:**
FEEDBACK: Another explicit place to invoke Odinn and his riddles, given the Gylfaginning is itself just a deception
> "In my hall, not all guests are equal. Some concepts are foundational - everything depends on them. Others are bridges - remove them and the hall splits in two. Let's find out who's who."

**Learning Objectives:**
1. Calculate degree centrality (the simple metric)
2. Run PageRank (the famous one)
3. Interpret what these numbers mean for your knowledge

**Key Queries:**
```cypher
-- Degree centrality: who has the most connections?
MATCH (c:Concept {domain: "implementation_hiding"})
OPTIONAL MATCH (c)-[r]-()
RETURN c.name, count(r) AS degree
ORDER BY degree DESC
LIMIT 10;

-- PageRank: who's actually influential?
CALL pagerank.get()
YIELD node, rank
WHERE node:Concept AND node.domain = "implementation_hiding"
RETURN node.name, round(rank * 1000) / 1000 AS influence
ORDER BY influence DESC;

-- Compare them side by side
MATCH (c:Concept {domain: "implementation_hiding"})
OPTIONAL MATCH (c)-[r]-()
WITH c, count(r) AS degree
CALL pagerank.get()
YIELD node, rank
WHERE node = c
RETURN c.name, degree, round(rank * 1000) / 1000 AS pagerank
ORDER BY pagerank DESC;
```

**Screenshot Moments:**
1. Degree centrality results
2. PageRank results
3. Graph visualization with nodes sized by PageRank

**Insight:** Encapsulation should dominate - it's the foundational concept that everything else builds on.

---

### Part 5: "The Tribes Within"

**Aegir's Hook:**
> "Even in my hall, guests cluster into groups. The warriors drink together, the poets recite together, the skalds... well, they're usually passed out. Your concepts do the same. Let's find the tribes."
FEEDBACK: Haha. We usually are.

**Learning Objectives:**
1. Understand community detection (clustering by connectivity)
2. Run Louvain algorithm
3. Interpret what communities mean for your domain

**Key Queries:**
```cypher
-- Detect communities
CALL community_detection.louvain()
YIELD node, community_id
WHERE node:Concept AND node.domain = "implementation_hiding"
RETURN community_id, collect(node.name) AS tribe_members
ORDER BY size(tribe_members) DESC;

-- Visualize with community labels
CALL community_detection.louvain()
YIELD node, community_id
WHERE node:Concept AND node.domain = "implementation_hiding"
RETURN node.name, community_id;
```

**Screenshot Moments:**
1. Community detection results as table
2. Graph colored by community ID
3. Clustered layout showing separation

**Expected Tribes:**
- Core hiding concepts (Encapsulation, Properties, Private methods)
- Design principles (Law of Demeter, Open-Closed)
- State management (Immutability, Dangerous Setters)

---

### Part 6: "Forging New Rules"

**Aegir's Hook:**
FEEDBACK: fjolkynngi!
> "Now for the real magic. You've learned to see the graph, query it, measure it, and cluster it. But the greatest power is this: you can forge NEW knowledge from the relationships. Watch closely."

**Learning Objectives:**
1. Understand how edge relationships imply rules
2. Write queries that derive rules from structure
3. See how qortex's projection pipeline does this automatically

**Key Queries:**
```cypher
-- Find REQUIRES relationships and derive dependency rules
MATCH (a:Concept)-[r:REL {type: "requires"}]->(b:Concept)
WHERE a.domain = "implementation_hiding"
RETURN
  a.name AS foundation,
  b.name AS dependent,
  "Before implementing " + b.name + ", ensure you understand " + a.name AS derived_rule;

-- Find CONTRADICTS relationships and derive warning rules
MATCH (a:Concept)-[r:REL {type: "contradicts"}]->(b:Concept)
WHERE a.domain = "implementation_hiding"
RETURN
  a.name, b.name,
  "Warning: " + a.name + " conflicts with " + b.name + ". Choose one approach." AS derived_rule;

-- Find highly-connected concepts and derive importance rules
CALL pagerank.get()
YIELD node, rank
WHERE node:Concept AND node.domain = "implementation_hiding" AND rank > 0.1
RETURN
  node.name AS key_concept,
  "Prioritize understanding " + node.name + " - it's foundational to this domain." AS derived_rule
ORDER BY rank DESC;

-- See the explicit rules we extracted
MATCH (r:Rule {domain: "implementation_hiding"})
RETURN r.text AS explicit_rule, r.category
LIMIT 5;
```

**Screenshot Moments:**
1. Derived rules from REQUIRES edges
2. Comparison: explicit rules vs derived rules
3. Final visualization: concepts + rules + derived insights

**The Payoff:**
> "You came to my hall knowing nothing of graphs. Now you can see the shape of knowledge, trace paths through it, measure what matters, find the hidden tribes, and forge new rules from pure structure. Not bad for a mortal."

---

## Screenshot Checklist

| Part | Screenshot | Description |
|------|------------|-------------|
| 1.1 | `lab-connect.png` | Quick Connect screen |
| 1.2 | `first-query.png` | Query editor with MATCH (n) |
| 1.3 | `full-graph.png` | All concepts visualized |
| 1.4 | `node-inspect.png` | Single node properties panel |
| 2.1 | `where-filter.png` | WHERE clause query + results |
| 2.2 | `pattern-match.png` | Relationship pattern visualization |
| 2.3 | `aggregation.png` | COUNT results table |
| 3.1 | `multi-hop.png` | Variable-length path results |
| 3.2 | `path-highlight.png` | Path highlighted in graph |
| 3.3 | `shortest-path.png` | Shortest path visualization |
| 4.1 | `degree.png` | Degree centrality results |
| 4.2 | `pagerank.png` | PageRank results |
| 4.3 | `sized-nodes.png` | Nodes sized by importance |
| 5.1 | `communities-table.png` | Louvain results as table |
| 5.2 | `communities-colored.png` | Graph colored by community |
| 6.1 | `derived-rules.png` | Query deriving rules from edges |
| 6.2 | `explicit-vs-derived.png` | Comparison view |
| 6.3 | `finale.png` | Full visualization with insights |

---

## Implementation Notes

### Playwright Screenshots

Use the Playwright MCP tools to automate:
1. Navigate to Lab
2. Connect to Memgraph
3. Run each query
4. Capture screenshots at key moments

### File Structure

```
docs/tutorials/graph-analytics/
├── index.md                    # Series intro
├── part1-welcome.md            # Foundations
├── part2-cypher.md             # Query language
├── part3-multi-hop.md          # Path queries
├── part4-centrality.md         # Importance metrics
├── part5-communities.md        # Clustering
├── part6-forging-rules.md      # Rule derivation
└── images/
    ├── lab-connect.png
    ├── first-query.png
    └── ... (all screenshots)
```

### Style Guide

- First person from Aegir's perspective
- Code blocks for all queries
- Screenshots after each major query
- "Try this" callouts for reader experimentation
- No corporate speak, no "leverage", no "utilize"
- Occasional mead/feast metaphors, but don't overdo it

---

## Future: Buildlog EDA

This same series structure applies to buildlog data:
- Nodes: Entries, Skills, Personas
- Edges: EXTRACTS, PROMOTES, REFERENCES
- Communities: Skill clusters by domain
- Rule derivation: Patterns from reward events

Document this as a follow-up series after the core tutorial is complete.
