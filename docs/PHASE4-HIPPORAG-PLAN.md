# Phase 4: HippoRAG Integration Plan for qortex

**Status**: Research complete, ready for implementation
**Dependencies**: Phase 0 (vector layer) must land first
**Estimated scope**: ~1500 LOC across 7 files
**Date**: 2026-02-07

---

## Table of Contents

- [A. Algorithm Explanation (for Humans)](#a-algorithm-explanation-for-humans)
  - [A.1 What Is HippoRAG?](#a1-what-is-hipporag)
  - [A.2 The Biological Analogy](#a2-the-biological-analogy)
  - [A.3 Step-by-Step Retrieval Flow](#a3-step-by-step-retrieval-flow-with-concrete-example)
  - [A.4 Personalized PageRank Explained](#a4-what-is-personalized-pagerank-ppr)
  - [A.5 Seed Nodes](#a5-seed-nodes)
  - [A.6 The Teleportation Vector](#a6-the-teleportation-vector)
  - [A.7 Knowledge Graph Construction](#a7-how-the-knowledge-graph-gets-built)
  - [A.8 HippoRAG v2 Improvements](#a8-hipporag-v2-improvements-over-v1)
- [B. Decisions and Algorithm Structure for qortex](#b-decisions-and-algorithm-structure-for-qortex-integration)
  - [B.1 Seed Node Finding](#b1-seed-node-finding)
  - [B.2 PPR Implementation Details](#b2-ppr-implementation-details)
  - [B.3 Rule Collection](#b3-rule-collection-from-activated-concepts)
  - [B.4 Feedback Loop](#b4-feedback--teleportation-factor-loop)
  - [B.5 KG Construction Differences](#b5-knowledge-graph-construction-differences)
- [C. Concrete File-Level Implementation Plan](#c-concrete-file-level-implementation-plan)
- [D. Open Questions and Design Decisions](#d-open-questions-and-design-decisions)

---

## A. Algorithm Explanation (for Humans)

### A.1 What Is HippoRAG?

HippoRAG is a retrieval framework that gives LLMs something closer to long-term memory. Standard RAG (Retrieval-Augmented Generation) works like a search engine: you embed your query as a vector, find the nearest document vectors, and hand those documents to an LLM. This works well when the answer lives in a single document, but fails when the answer requires connecting facts scattered across multiple documents -- so-called "multi-hop" reasoning.

HippoRAG solves this by building a knowledge graph from your documents and using a graph algorithm called Personalized PageRank to spread activation from query concepts through the graph, discovering connections the query never explicitly mentioned. Where standard RAG asks "which documents look like this query?", HippoRAG asks "which concepts are connected to this query's concepts, and which documents contain those connected concepts?"

The framework was developed by researchers at Ohio State University and published at NeurIPS 2024 (v1, arXiv:2405.14831) with a substantially improved v2 published in February 2025 (arXiv:2502.14802, "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"). HippoRAG v1 achieves up to 20% improvement over standard RAG on multi-hop QA, while being 10-30x cheaper and 6-13x faster than iterative retrieval approaches. HippoRAG v2 further improves associative memory tasks by ~7 F1 points over state-of-the-art embedding retrievers.

### A.2 The Biological Analogy

HippoRAG is inspired by the hippocampal indexing theory of human memory. In the brain, three systems cooperate for long-term memory:

| Brain Component | HippoRAG Component | What It Does |
|---|---|---|
| **Neocortex** | LLM + stored documents | Stores actual content. The neocortex processes raw sensory input into high-level representations. In HippoRAG, the LLM reads documents and extracts structured knowledge (triples). |
| **Hippocampal index** | Knowledge graph + PPR | Stores associations between memory fragments, not the memories themselves. When you smell your grandmother's perfume and suddenly recall her kitchen, her voice, the cookies -- that is the hippocampal index activating a web of associations from a partial cue. In HippoRAG, the knowledge graph stores connections between concepts, and PPR spreads activation through those connections. |
| **Parahippocampal regions** | Retrieval encoder (embedding model) | Channels processed information between the neocortex and hippocampus. Detects when two concepts are actually the same thing (synonymy). In HippoRAG, the embedding model links query entities to graph nodes and identifies synonym pairs. |

The brain performs two key operations that HippoRAG replicates:

1. **Pattern separation** (during encoding/indexing): Ensures distinct experiences get distinct representations. When you extract the triple ("Metformin", "treats", "Diabetes"), "Metformin" becomes its own discrete node -- not a point in embedding space that might blur with "Metoprolol" or "Methanol". This is the knowledge graph doing pattern separation.

2. **Pattern completion** (during retrieval): Starting from a partial cue, activation spreads through the index until a complete memory emerges. When you query "metformin and kidney problems", the system finds the Metformin node and the Renal Impairment node, then PPR discovers that Lactic Acidosis strongly connects them -- even though the query never mentioned lactic acidosis.

### A.3 Step-by-Step Retrieval Flow with Concrete Example

Suppose a coding agent asks: *"What patterns should I follow when implementing a repository pattern with dependency injection?"*

**Phase 1: Offline Indexing (already done)**

The knowledge graph was built during document ingestion:

```
Document: "Chapter 5 - Design Patterns"
    |
    v
LLM extracts triples (OpenIE):
  ("Repository Pattern", "requires", "Interface Segregation")
  ("Repository Pattern", "uses", "Dependency Injection")
  ("Dependency Injection", "requires", "Constructor Parameters")
  ("Constructor Parameters", "contradicts", "Service Locator")
  ("Interface Segregation", "supports", "Testability")
    |
    v
Knowledge graph built:
  Nodes: Repository Pattern, Interface Segregation,
         Dependency Injection, Constructor Parameters,
         Service Locator, Testability
  Edges: requires, uses, contradicts, supports
    |
    v
Nodes embedded for similarity matching
Synonym edges added (e.g., "DI" <-> "Dependency Injection")
Each node linked back to source passages
```

**Phase 2: Online Retrieval**

```
Step 1: Extract query concepts
  Query: "repository pattern with dependency injection"
  Extracted: ["repository pattern", "dependency injection"]

Step 2: Find seed nodes via embedding similarity
  "repository pattern" -> Repository Pattern node (0.97 similarity)
  "dependency injection" -> Dependency Injection node (0.95 similarity)
  Seed nodes: {Repository Pattern, Dependency Injection}

Step 3: Personalized PageRank from seeds
  Random walker starts at seed nodes, takes steps along edges,
  occasionally teleports back to seeds.

  After convergence:
    Repository Pattern:      0.28  (seed - high)
    Dependency Injection:    0.25  (seed - high)
    Interface Segregation:   0.18  (connects to both seeds!)
    Constructor Parameters:  0.12  (one hop from DI)
    Testability:             0.09  (two hops, but connected via ISP)
    Service Locator:         0.05  (connected but as contradiction)

Step 4: Collect rules from activated concepts
  From high-scoring concepts, retrieve linked ExplicitRules:
    - "Always define repository interfaces before implementations"
      (from Interface Segregation, score: 0.18)
    - "Inject dependencies via constructor, never service locator"
      (from Constructor Parameters + Service Locator, score: 0.12)
    - "Repository methods should return domain objects, not ORM entities"
      (from Repository Pattern, score: 0.28)

Step 5: Return ranked rules to the agent
```

The system found "Interface Segregation" and "Constructor Parameters" -- concepts the query never mentioned -- because they are structurally connected to the query concepts in the knowledge graph. This is the power of pattern completion.

### A.4 What Is Personalized PageRank (PPR)?

Regular PageRank answers: "If a random person surfs the web by clicking links forever, which pages would they visit most often?" The answer gives you a global importance ranking.

**Personalized** PageRank changes the question to: "If a random person starts on *specific pages I care about*, clicks links randomly, but periodically teleports back to those starting pages, where do they end up most often?"

Think of it this way. You are at a party looking for someone who knows about both cooking and jazz. You start with the people you know who cook, and the people you know who play jazz. You ask each of them who they know. Then you ask *those* people who they know. Periodically you come back to your starting contacts to make sure you do not drift too far. After enough conversations, the people you have talked to most are the ones most connected to both cooking and jazz -- even if they never described themselves that way.

Mathematically, PPR computes a stationary distribution over the graph. At each step, with probability `d` (the damping factor, typically 0.5-0.85), the walker follows an edge to a neighbor. With probability `1-d`, it "teleports" back to one of the seed nodes. After enough iterations, the probability of being at each node converges to a stable distribution. Nodes that are well-connected to the seeds accumulate high probability.

The algorithm is computed via power iteration:

```
Initialize: scores[node] = 1/|seeds| if node in seeds, else 0

Repeat until convergence:
  For each node n:
    walk_score = d * SUM(scores[neighbor] / out_degree(neighbor)
                        for neighbor in predecessors(n))
    teleport_score = (1 - d) * teleportation_vector[n]
    new_scores[n] = walk_score + teleport_score

  scores = new_scores
```

The key insight is that PPR finds nodes that are *structurally central* relative to your starting points, not just similar in embedding space.

### A.5 Seed Nodes

Seed nodes are the starting points for PPR -- the nodes in the knowledge graph that correspond to concepts found in the query. They are the "partial cue" that triggers pattern completion.

In HippoRAG v1, seed nodes were found by extracting named entities from the query (NER) and matching them to graph nodes by string similarity or embedding similarity.

In HippoRAG v2, seed node selection was significantly improved:

1. **Query-to-triple linking**: Instead of just extracting entities from the query, the system embeds the entire query and compares it against stored triples (subject-relation-object). This captures context: "metformin risks" matches the triple ("Metformin", "risk_with", "Renal Impairment") better than just matching the entity "Metformin".

2. **Recognition memory filtering**: An LLM filters the top retrieved triples to remove irrelevant matches, ensuring only genuinely relevant seed nodes enter PPR.

3. **Two types of seed nodes**: HippoRAG v2 uses both *phrase nodes* (from filtered triples) and *passage nodes* (the original document chunks), with a weight factor balancing their influence in the teleportation vector.

The quality of seed node selection directly determines retrieval quality. Bad seeds poison the entire PPR walk. HippoRAG v2's query-to-triple approach improved Recall@5 by 12.5% over v1's NER-to-node approach.

### A.6 The Teleportation Vector

The teleportation vector is what makes PageRank "personalized." In standard PageRank, when the random walker teleports, it goes to a uniformly random node. In PPR, when it teleports, it goes to a node selected according to the teleportation vector -- a probability distribution over nodes that concentrates mass on the seed nodes.

Formally, the teleportation vector `v` is:

```
v[i] = score(i) / sum(score(j) for j in seeds)   if i is a seed node
v[i] = 0                                           otherwise
```

Where `score(i)` is typically the embedding similarity between the query and the seed node (or the ranking score from triple retrieval).

This means not all seeds are equal. If "Repository Pattern" matched the query with 0.97 similarity and "Dependency Injection" matched with 0.60, the walker teleports to Repository Pattern more often, biasing the walk toward that region of the graph.

In HippoRAG v2, the teleportation vector incorporates two types of scores:
- **Phrase node scores**: Based on average ranking across filtered triples
- **Passage node scores**: Based on embedding similarity to the query, multiplied by a weight factor (hyperparameter) to balance phrase vs. passage influence

**For qortex, we add a novel extension**: teleportation *factors* that are learned from feedback. When a consumer accepts a rule that came from a particular concept, that concept's teleportation factor increases. When a rule is rejected, the factor decreases. These factors multiplicatively modulate the base teleportation vector, creating a feedback loop where the system learns which regions of the graph are more useful. This is inspired by Rossi & Gleich (2012) who showed that evolving teleportation captures changes in external interest over time -- but to our knowledge, nobody has used reward signals to modulate teleportation.

### A.7 How the Knowledge Graph Gets Built

HippoRAG builds its KG through an extraction pipeline:

**Step 1: Triple Extraction (OpenIE)**
An LLM reads each passage and extracts (subject, relation, object) triples:
- Input: "Metformin is a first-line medication for type 2 diabetes that carries a risk of lactic acidosis in patients with kidney disease."
- Output: ("Metformin", "is first-line medication for", "Type 2 Diabetes"), ("Metformin", "carries risk of", "Lactic Acidosis"), ("Lactic Acidosis", "elevated risk in", "Kidney Disease")

**Step 2: Graph Construction**
Each unique subject/object becomes a **phrase node**. Each relation becomes a **relation edge**. The graph is schema-less: there are no predefined node types or relation types.

**Step 3: Synonym Detection**
The retrieval encoder (embedding model) computes embeddings for all phrase nodes. When two nodes have cosine similarity above a threshold (tau), a **synonym edge** is added between them. For example, "Kidney Disease" and "Renal Impairment" would get a synonym edge.

**Step 4: Passage-to-Node Linking** (v2)
Each original passage becomes a **passage node**. A **contains edge** connects each passage node to all phrase nodes whose triples were extracted from that passage.

**Step 5: Embedding Index**
All phrase nodes (and in v2, passage nodes) are embedded and indexed for fast similarity search during online retrieval.

HippoRAG v2 uses Llama-3.3-70B-Instruct for extraction and triple filtering, and NV-Embed-v2 as the retrieval encoder.

### A.8 HippoRAG v2 Improvements over v1

| Aspect | v1 (NeurIPS 2024) | v2 (Feb 2025) |
|---|---|---|
| **Query processing** | NER extracts entities from query, matches to nodes | Query-to-triple: embed full query, match against stored triples (+12.5% Recall@5) |
| **Seed filtering** | No filtering | Recognition memory: LLM filters top triples to remove noise |
| **Graph nodes** | Phrase nodes only | Phrase nodes + passage nodes (dense-sparse integration) |
| **PPR teleportation** | Only phrase seed nodes | Both phrase and passage seeds, with weight factor balancing |
| **Passage ranking** | Via node-to-passage linking matrix | Direct passage node scores from PPR |
| **Extraction model** | GPT-3.5/4 (proprietary) | Llama-3.3-70B-Instruct (open-source) |
| **Multi-hop QA (MuSiQue)** | ~44.8 F1 | ~51.9 F1 |
| **Recall@5 (2Wiki)** | ~76.5% | ~90.4% |

The core insight of v2 is that combining sparse (phrase nodes, representing extracted concepts) and dense (passage nodes, representing full document contexts) representations in a single graph, then running PPR over both, captures the best of both worlds: precise concept matching *and* contextual document retrieval.

---

## B. Decisions and Algorithm Structure for qortex Integration

### B.1 Seed Node Finding

**Current state** (`retrieval.py:_extract_query_concepts` and `_find_seed_nodes`):
- `_extract_query_concepts`: Naive keyword split -- splits on whitespace, filters words < 4 chars
- `_find_seed_nodes`: String matching via `backend.find_nodes(name_pattern=concept)`

**Target state**: Embedding-based seed finding with optional LLM enhancement.

#### What to Embed

Embed **node names concatenated with descriptions**. Rationale:
- Node names alone are often terse ("DI", "ISP", "Repository") -- embedding them yields poor discrimination
- Descriptions provide context: "Dependency Injection - a design pattern where objects receive their dependencies through constructors or setters rather than creating them internally"
- The concatenated string `f"{node.name}: {node.description}"` gives the embedding model both the label and the semantics
- This aligns with HippoRAG's approach where phrase nodes carry the semantics of their extracted triples

For the query side, embed the **full query context** (not extracted keywords). This follows HippoRAG v2's query-to-triple insight: the full query carries more context than extracted fragments.

#### Algorithm

```python
def _find_seed_nodes(
    self,
    query: str,
    domains: list[str] | None,
    top_k: int = 10,
    threshold: float = 0.5,
) -> list[tuple[str, float]]:  # Returns (node_id, similarity_score)
    """
    1. Embed the query using EmbeddingModel
    2. Search VectorIndex for top-k similar node embeddings
    3. Filter by domain if specified
    4. Filter by similarity threshold
    5. Return (node_id, score) pairs for teleportation vector
    """
    query_embedding = self.embedding_model.embed(query)

    # VectorIndex.search returns (node_id, score) pairs
    candidates = self.vector_index.search(
        query_embedding, top_k=top_k * 3  # Over-retrieve, then filter
    )

    seeds = []
    for node_id, score in candidates:
        if score < threshold:
            continue
        if domains:
            node = self.backend.get_node(node_id)
            if node and node.domain not in domains:
                continue
        seeds.append((node_id, score))

    return seeds[:top_k]
```

#### Fallback Strategies

1. **No vector index available**: Fall back to current string matching via `find_nodes`
2. **No seed nodes above threshold**: Lower threshold to 0.3 and try again
3. **Still no seeds**: Fall back to keyword extraction (current behavior) + string matching
4. **Zero seeds**: Return empty result (already handled)

#### Decision: Skip recognition memory for MVP

HippoRAG v2's recognition memory uses an LLM call during retrieval to filter seed nodes. This adds latency and cost. For qortex MVP:
- Rely on the similarity threshold for filtering
- Add recognition memory as a Phase 4.1 enhancement if precision is insufficient

### B.2 PPR Implementation Details

#### Power Iteration Algorithm (Pseudocode)

```python
def personalized_pagerank(
    adjacency: dict[str, list[str]],    # node -> neighbors
    seed_scores: dict[str, float],       # node_id -> teleportation weight
    teleportation_factors: dict[str, float] | None,  # node_id -> learned factor
    damping: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """
    Compute PPR via power iteration.

    Args:
        adjacency: The graph as adjacency lists (treat as undirected)
        seed_scores: Base teleportation weights from embedding similarity
        teleportation_factors: Learned per-node multipliers from feedback
        damping: Probability of following an edge (vs teleporting)
        max_iterations: Upper bound on iterations
        tolerance: Convergence threshold (L1 norm of score change)

    Returns:
        node_id -> PPR score mapping
    """
    # Build the teleportation vector
    teleport = {}
    for node_id, base_score in seed_scores.items():
        factor = 1.0
        if teleportation_factors and node_id in teleportation_factors:
            factor = teleportation_factors[node_id]
        teleport[node_id] = base_score * factor

    # Normalize teleportation vector to sum to 1
    total = sum(teleport.values())
    if total > 0:
        teleport = {k: v / total for k, v in teleport.items()}

    # Collect all nodes in the graph
    all_nodes = set(adjacency.keys())
    for neighbors in adjacency.values():
        all_nodes.update(neighbors)

    # Initialize scores: concentrated on seeds
    scores = {n: teleport.get(n, 0.0) for n in all_nodes}

    # Pre-compute out-degrees
    out_degree = {n: len(adjacency.get(n, [])) for n in all_nodes}

    for iteration in range(max_iterations):
        new_scores = {}

        for node in all_nodes:
            # Walk component: sum of score flowing in from predecessors
            walk_score = 0.0
            for neighbor in adjacency.get(node, []):
                deg = out_degree.get(neighbor, 1)
                walk_score += scores.get(neighbor, 0.0) / deg

            # Combined score
            new_scores[node] = (
                damping * walk_score +
                (1 - damping) * teleport.get(node, 0.0)
            )

        # Check convergence (L1 norm)
        diff = sum(abs(new_scores[n] - scores.get(n, 0.0)) for n in all_nodes)

        scores = new_scores

        if diff < tolerance:
            break

    return scores
```

#### Damping Factor Choice

HippoRAG v1 uses a damping factor of **0.5**, not the traditional 0.85 from web PageRank. The reason:

- **0.85** (web PageRank): Walker mostly follows links, rarely teleports. Good for finding globally important pages. Poor for personalization because the seed influence dissipates quickly.
- **0.5** (HippoRAG): Walker follows links half the time, teleports to seeds half the time. Keeps the walk strongly anchored to the query concepts. Better for retrieval because you want results *relevant to the query*, not globally important.

**Recommendation for qortex**: Start with **0.5** (matching HippoRAG). The current codebase uses 0.85 -- this should change. With a small KG (hundreds to low thousands of nodes), a lower damping factor keeps scores concentrated near the query concepts, which is what we want.

#### Convergence Criteria

- **Tolerance**: `1e-6` (L1 norm of score change between iterations)
- **Max iterations**: 100 (power iteration on graphs of qortex's size should converge in 15-30 iterations)
- In practice, for graphs under 10,000 nodes, convergence is nearly instant

#### Teleportation Vector Construction from Seed Nodes

The teleportation vector is built from seed node scores:

```python
# From _find_seed_nodes, we get: [(node_id, similarity_score), ...]
seed_scores = {node_id: score for node_id, score in seeds}

# Apply learned teleportation factors (from feedback)
for node_id in seed_scores:
    if node_id in teleportation_factors:
        seed_scores[node_id] *= teleportation_factors[node_id]

# Normalize to probability distribution
total = sum(seed_scores.values())
teleportation_vector = {k: v / total for k, v in seed_scores.items()}
```

This means:
1. Nodes that matched the query better get more teleportation probability
2. Nodes that historically led to accepted rules get an additional boost
3. Normalization ensures it remains a valid probability distribution

#### Teleportation Factors from Feedback

Each node can have a learned **teleportation factor** -- a multiplier that modulates its base teleportation probability. These factors are updated by the feedback loop (see B.4).

The modulated teleportation vector is:

```
v'[i] = (base_score[i] * factor[i]) / sum(base_score[j] * factor[j] for j in seeds)
```

Where `factor[i]` defaults to 1.0 and is adjusted by feedback:
- Rule accepted -> concepts in that rule get factor += reward_increment
- Rule rejected -> concepts get factor *= decay_multiplier
- Factor is clamped to [0.1, 10.0] to prevent extreme values

#### InMemoryBackend: Real Power Iteration

The current `InMemoryBackend.personalized_pagerank` is a 2-hop BFS with exponential decay. This must be replaced with real power iteration:

```python
def personalized_pagerank(
    self,
    source_nodes: list[str],
    damping_factor: float = 0.5,
    max_iterations: int = 100,
    domain: str | None = None,
    teleportation_factors: dict[str, float] | None = None,
) -> dict[str, float]:
    # Build adjacency from self._edges
    adjacency: dict[str, list[str]] = {}
    for edge in self._edges:
        src, tgt = edge.source_id, edge.target_id
        # Filter by domain if specified
        if domain:
            src_node = self._nodes.get(src)
            tgt_node = self._nodes.get(tgt)
            if not src_node or src_node.domain != domain:
                continue
            if not tgt_node or tgt_node.domain != domain:
                continue
        adjacency.setdefault(src, []).append(tgt)
        adjacency.setdefault(tgt, []).append(src)  # Undirected

    # Build seed scores (uniform for now; caller should pass weighted seeds)
    seed_scores = {}
    for nid in source_nodes:
        if nid in self._nodes:
            if domain and self._nodes[nid].domain != domain:
                continue
            seed_scores[nid] = 1.0 / len(source_nodes)

    return _power_iteration(
        adjacency, seed_scores, teleportation_factors,
        damping_factor, max_iterations
    )
```

#### MemgraphBackend: Passing Teleportation Factors to MAGE PPR

Memgraph MAGE's `pagerank.get()` accepts `personalization_nodes` but does NOT natively support per-node teleportation weights. Options:

1. **Option A (recommended for MVP)**: Use MAGE PPR with uniform seed weights, then post-multiply scores by teleportation factors. This is an approximation but avoids fighting the MAGE API.

2. **Option B (accurate but complex)**: Implement power iteration in Cypher using MAGE utility functions. This gives full control but is slower and harder to maintain.

3. **Option C (future)**: Contribute a weighted teleportation PR to MAGE, or use the upcoming `pagerank.personalized()` if it lands.

**Recommendation**: Option A for MVP. The post-multiplication approximation is:

```python
raw_scores = self._mage_ppr(source_nodes, damping_factor, max_iterations, domain)
if teleportation_factors:
    for node_id in raw_scores:
        if node_id in teleportation_factors:
            raw_scores[node_id] *= teleportation_factors[node_id]
return raw_scores
```

This is not mathematically equivalent to true weighted teleportation, but for the use case (slight preference adjustments based on feedback), the approximation is adequate. If precision matters later, implement Option B.

### B.3 Rule Collection from Activated Concepts

**Current state** (`retrieval.py:_collect_rules`): Returns empty list (stub).

**Target state**: Retrieve ExplicitRules linked to high-scoring concepts, scored by concept activation.

#### How PPR Scores Translate to Rule Relevance

Each `ExplicitRule` has a `concept_ids` field linking it to the concepts it operationalizes. The relevance of a rule is the maximum PPR score among its linked concepts:

```python
def _collect_rules(
    self,
    scores: dict[str, float],
    top_k: int,
) -> list[Rule]:
    # Get all rules from the backend
    all_rules = self.backend.get_rules(domain=None)

    scored_rules = []
    for explicit_rule in all_rules:
        # Rule relevance = max PPR score among its linked concepts
        max_concept_score = 0.0
        activated_concepts = []

        for concept_id in explicit_rule.concept_ids:
            if concept_id in scores:
                concept_score = scores[concept_id]
                max_concept_score = max(max_concept_score, concept_score)
                activated_concepts.append(concept_id)

        if max_concept_score > 0:
            rule = Rule(
                id=explicit_rule.id,
                text=explicit_rule.text,
                domain=explicit_rule.domain,
                derivation="explicit",
                source_concepts=activated_concepts,
                confidence=explicit_rule.confidence,
                relevance=max_concept_score,
                category=explicit_rule.category,
            )
            scored_rules.append(rule)

    # Sort by relevance, return top-k
    scored_rules.sort(key=lambda r: r.relevance, reverse=True)
    return scored_rules[:top_k]
```

#### Filtering/Scoring Strategy

1. **Threshold**: Only include rules where max concept score > 0.01 (already filtered by `_group_by_domain` threshold)
2. **Scoring**: `relevance = max(ppr_score[c] for c in rule.concept_ids)` -- using max rather than sum because a rule linked to one highly activated concept is more relevant than a rule linked to many barely activated concepts
3. **Tie-breaking**: When relevance is equal, prefer rules with higher extraction confidence
4. **Domain filtering**: If domains are specified in the query, rules from other domains are excluded

#### How ExplicitRules Link to ConceptNodes via concept_ids

The linkage already exists in the data model. Each `ExplicitRule` has:
```python
concept_ids: list[str] = field(default_factory=list)
```

These are populated during ingestion (by the LLM extracting triples and rules from source material). The rule "Always define interfaces before implementations" would have `concept_ids` pointing to the "Interface Segregation" and "Implementation" concept nodes.

For rule collection to work, we need the ingestion pipeline to reliably populate `concept_ids`. This is already happening in the current ingestors -- the key is that the LLM is instructed to extract rules AND link them to the concepts they reference.

### B.4 Feedback -> Teleportation Factor Loop

This is the novel component -- reward-modulated teleportation. The theoretical basis is Rossi & Gleich (2012), who showed that time-varying teleportation in PageRank captures changes in external interest. We extend this by using consumer feedback (accepted/rejected/partial) as the signal that drives teleportation changes.

#### How Feedback Updates Per-Node Factors

```python
@dataclass
class TeleportationFactors:
    """Learned per-node teleportation multipliers from feedback."""

    factors: dict[str, float]  # concept_id -> factor (default 1.0)
    _path: Path

    # Hyperparameters
    reward_increment: float = 0.1     # How much to boost on acceptance
    penalty_decay: float = 0.95       # Multiplicative decay on rejection
    min_factor: float = 0.1           # Floor (never fully suppress a node)
    max_factor: float = 10.0          # Ceiling (prevent runaway amplification)

    def update(
        self,
        concept_ids: list[str],
        outcome: Literal["accepted", "rejected", "partial"],
    ) -> None:
        """Update factors based on feedback outcome."""
        for cid in concept_ids:
            current = self.factors.get(cid, 1.0)

            if outcome == "accepted":
                new = current + self.reward_increment
            elif outcome == "rejected":
                new = current * self.penalty_decay
            elif outcome == "partial":
                new = current + (self.reward_increment * 0.3)  # Mild boost
            else:
                continue

            self.factors[cid] = max(self.min_factor, min(self.max_factor, new))

        self.save()

    def get(self, concept_id: str) -> float:
        """Get factor for a concept (default 1.0)."""
        return self.factors.get(concept_id, 1.0)

    def save(self) -> None:
        """Persist factors to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.factors, indent=2))

    @classmethod
    def load(cls, path: Path) -> TeleportationFactors:
        """Load factors from disk."""
        factors = {}
        if path.exists():
            factors = json.loads(path.read_text())
        return cls(factors=factors, _path=path)
```

#### Persistence

Factors are stored as a JSON file at `~/.qortex/teleportation_factors.json`:

```json
{
  "concept-uuid-1": 1.3,
  "concept-uuid-2": 0.85,
  "concept-uuid-3": 1.0
}
```

JSON is chosen over JSONL because:
- The file is read entirely on each query (need all factors)
- It is written atomically after each feedback event
- The file stays small (one float per concept, typically < 100KB even for large graphs)

#### Integration into PPR Teleportation Vector

```python
# In Hippocampus.__init__:
self.factors = TeleportationFactors.load(
    Path("~/.qortex/teleportation_factors.json").expanduser()
)

# In _ppr_completion, construct weighted teleportation:
seed_scores = {nid: score for nid, score in seeds}  # From _find_seed_nodes
teleportation_factors = {
    nid: self.factors.get(nid) for nid, _ in seeds
}

scores = self.backend.personalized_pagerank(
    source_nodes=[nid for nid, _ in seeds],
    damping_factor=0.5,
    max_iterations=100,
    domain=None,
    teleportation_factors=teleportation_factors,
)
```

#### Why This Is Novel

Rossi & Gleich (2012) used time-varying teleportation to model changing external interest (e.g., Wikipedia page views). Their contribution was showing that PageRank can smoothly adapt to changing teleportation without full recomputation.

We use *reward signals* as the source of teleportation change. When a consumer accepts a rule, the concepts behind that rule become more "interesting" (higher teleportation probability). When rules are rejected, those concepts dampen. This creates an online learning loop where the retrieval system improves based on downstream outcomes.

To our knowledge, no prior work has combined PPR teleportation modulation with agent feedback signals. This is a testable contribution: H1 of the research agenda can measure whether reward-modulated teleportation improves rule acceptance rates over static teleportation.

### B.5 Knowledge Graph Construction Differences

#### HippoRAG KG vs. qortex KG

| Aspect | HippoRAG | qortex |
|---|---|---|
| **Node types** | Phrase nodes (concepts) + Passage nodes (v2) | ConceptNodes only (no passage nodes) |
| **Edge types** | Relation edges (from OpenIE, untyped) + Synonym edges + Contains edges (v2) | 10 typed RelationTypes (REQUIRES, CONTRADICTS, etc.) |
| **Schema** | Schema-less (any string for relation) | Typed (RelationType enum) |
| **Extraction** | OpenIE triples: (subj, rel, obj) | LLM extraction: ConceptNodes + typed ConceptEdges + ExplicitRules |
| **Rules** | Not explicit -- passages are the retrieval target | ExplicitRules are first-class, linked to concepts via concept_ids |
| **Domains** | Single corpus, no isolation | Multi-domain with isolation (like Postgres schemas) |
| **Synonym detection** | Embedding similarity above threshold -> synonym edge | Not implemented yet |
| **Passage linking** | Contains edges (v2) | source_id on ConceptNode (implicit) |

#### Adaptations Needed

1. **No passage nodes needed for MVP**: qortex retrieves *rules*, not passages. HippoRAG v2's passage nodes make sense when the goal is to return document passages to an LLM. In qortex, the goal is to return rules, which are already linked to concepts via `concept_ids`. The passage node concept is unnecessary for this use case.

2. **Add synonym edges**: qortex's `RelationType` enum needs a `SYNONYM` type. During ingestion (or as a post-processing step), the embedding model should detect concept pairs with similarity above a threshold and add SYNONYM edges. This is critical for PPR to bridge naming variations.

   ```python
   class RelationType(StrEnum):
       # ... existing types ...
       SYNONYM = "synonym"  # New: detected via embedding similarity
   ```

3. **Typed edges are a feature, not a limitation**: HippoRAG's schema-less edges mean all edges are weighted equally in PPR. qortex's typed edges allow potential future improvements: weighting REQUIRES edges differently from SIMILAR_TO edges in the adjacency matrix. For MVP, treat all edges equally (matching HippoRAG behavior).

4. **Cross-domain edges**: qortex's domain model allows concepts in different domains. PPR should be able to walk across domain boundaries (for the "cross-domain integration" use case). The `domains` parameter in `query()` should filter *results*, not the graph traversal -- PPR should walk the full graph, then filter output by domain.

5. **No additional edge types strictly required**: The existing 10 RelationTypes plus SYNONYM are sufficient. HippoRAG's "contains" edge maps to qortex's `source_id` field on ConceptNode (we can reconstruct which source a concept came from). If we later want passage-level retrieval, we would add passage nodes and contains edges then.

---

## C. Concrete File-Level Implementation Plan

### C.1 `src/qortex/hippocampus/retrieval.py` -- Main Retrieval Logic

**File**: `/Users/peleke/Documents/Projects/qortex-track-c/src/qortex/hippocampus/retrieval.py`

Changes to each method:

| Method | Current | Target |
|---|---|---|
| `__init__` | Takes `backend` only | Add `embedding_model`, `vector_index`, `teleportation_factors` params |
| `query()` | Calls naive methods | Calls updated methods; constructs weighted teleportation vector |
| `_extract_query_concepts()` | Splits on whitespace | **Remove or repurpose**. HippoRAG v2 embeds the full query, not keywords. This method becomes a fallback only. |
| `_find_seed_nodes()` | String matching via `find_nodes` | **Rewrite**: Embed query -> search vector index -> filter by domain/threshold -> return (node_id, score) pairs |
| `_ppr_completion()` | Calls `backend.personalized_pagerank` with uniform seeds | Pass seed scores and teleportation factors to backend PPR |
| `_bfs_completion()` | BFS with decay | Keep as fallback but pass teleportation factors for factor-weighted scoring |
| `_collect_rules()` | Returns `[]` | **Implement**: Retrieve rules linked to activated concepts, score by PPR activation, return top-k |
| `_group_by_domain()` | Groups by domain from node_id prefix | No change needed |
| `_node_in_domains()` | Checks node_id prefix | No change needed |

**New constructor signature**:

```python
class Hippocampus:
    def __init__(
        self,
        backend: GraphBackend,
        embedding_model: EmbeddingModel | None = None,
        vector_index: VectorIndex | None = None,
        factors_path: Path | None = None,
    ):
        self.backend = backend
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self._use_ppr = backend.supports_mage() or (vector_index is not None)
        self.factors = TeleportationFactors.load(
            factors_path or Path("~/.qortex/teleportation_factors.json").expanduser()
        )
```

**New method for feedback**:

```python
def record_feedback(
    self,
    rule_ids: list[str],
    outcome: Literal["accepted", "rejected", "partial"],
) -> None:
    """Record consumer feedback, updating teleportation factors."""
    # Look up which concepts these rules are linked to
    concept_ids = set()
    for rule_id in rule_ids:
        rules = [r for r in self.backend.get_rules() if r.id == rule_id]
        for rule in rules:
            concept_ids.update(rule.concept_ids)

    self.factors.update(list(concept_ids), outcome)
```

### C.2 `src/qortex/hippocampus/factors.py` -- NEW FILE

**File**: `/Users/peleke/Documents/Projects/qortex-track-c/src/qortex/hippocampus/factors.py`

This is a new file containing:

```python
"""Teleportation factors: reward-modulated PPR personalization.

Novel extension: consumer feedback (accepted/rejected/partial) updates
per-node teleportation factors, creating a learning loop in retrieval.

Theoretical basis: Rossi & Gleich (2012) showed evolving teleportation
captures changes in external interest. We use reward signals as the
source of teleportation change.
"""

@dataclass
class TeleportationFactors:
    factors: dict[str, float]
    _path: Path
    reward_increment: float = 0.1
    penalty_decay: float = 0.95
    min_factor: float = 0.1
    max_factor: float = 10.0

    def update(self, concept_ids, outcome) -> None: ...
    def get(self, concept_id) -> float: ...
    def get_batch(self, concept_ids) -> dict[str, float]: ...
    def save(self) -> None: ...
    @classmethod
    def load(cls, path) -> TeleportationFactors: ...
    def reset(self) -> None: ...  # Reset all factors to 1.0
    def summary(self) -> dict: ...  # Stats: min, max, mean, count
```

Estimated: ~80 LOC.

### C.3 `src/qortex/hippocampus/adapter.py` -- NEW FILE

**File**: `/Users/peleke/Documents/Projects/qortex-track-c/src/qortex/hippocampus/adapter.py`

This implements the `RetrievalAdapter` protocol for HippoRAG-style retrieval:

```python
"""HippoRAGAdapter: RetrievalAdapter backed by PPR-based retrieval.

Wraps Hippocampus to implement the RetrievalAdapter protocol,
allowing qortex_query MCP tool to switch between VecOnlyAdapter
(Phase 0) and HippoRAGAdapter (Phase 4) via mode parameter.
"""

@dataclass
class HippoRAGAdapter:
    """RetrievalAdapter using knowledge graph + PPR for retrieval."""

    hippocampus: Hippocampus

    def retrieve(
        self,
        query: str,
        domains: list[str] | None = None,
        top_k: int = 10,
    ) -> list[Rule]:
        result = self.hippocampus.query(query, domains, top_k)
        return result.rules

    def record_feedback(
        self,
        rule_ids: list[str],
        outcome: str,
    ) -> None:
        self.hippocampus.record_feedback(rule_ids, outcome)
```

Estimated: ~40 LOC.

### C.4 `src/qortex/core/memory.py` -- Real PPR in InMemoryBackend

**File**: `/Users/peleke/Documents/Projects/qortex-track-c/src/qortex/core/memory.py`

Replace the BFS-based `personalized_pagerank` method (lines 210-252) with real power iteration:

**Change the method signature** to accept `teleportation_factors`:

```python
def personalized_pagerank(
    self,
    source_nodes: list[str],
    damping_factor: float = 0.5,       # Changed default from 0.85
    max_iterations: int = 100,
    domain: str | None = None,
    teleportation_factors: dict[str, float] | None = None,  # NEW
) -> dict[str, float]:
```

**Replace the body** with power iteration over `self._edges` and `self._nodes`, building the adjacency structure on-the-fly and running the iteration loop described in B.2.

Estimated: ~60 LOC (replacing ~30 LOC of BFS).

### C.5 `src/qortex/core/backend.py` -- Protocol and Memgraph Changes

**File**: `/Users/peleke/Documents/Projects/qortex-track-c/src/qortex/core/backend.py`

**Protocol change** (line 178-189): Add `teleportation_factors` parameter:

```python
@abstractmethod
def personalized_pagerank(
    self,
    source_nodes: list[str],
    damping_factor: float = 0.5,       # Changed default
    max_iterations: int = 100,
    domain: str | None = None,
    teleportation_factors: dict[str, float] | None = None,  # NEW
) -> dict[str, float]:
```

**MemgraphBackend change** (lines 618-662): Accept `teleportation_factors`, apply post-multiplication approximation (Option A from B.2):

```python
def personalized_pagerank(
    self,
    source_nodes: list[str],
    damping_factor: float = 0.5,
    max_iterations: int = 100,
    domain: str | None = None,
    teleportation_factors: dict[str, float] | None = None,
) -> dict[str, float]:
    # ... existing MAGE call ...
    raw_scores = {r["id"]: r["rank"] for r in records if r.get("id")}

    # Post-multiply by teleportation factors (approximation)
    if teleportation_factors:
        for node_id in raw_scores:
            if node_id in teleportation_factors:
                raw_scores[node_id] *= teleportation_factors[node_id]

    return raw_scores
```

**Also add SYNONYM to RelationType** in `models.py`:

```python
class RelationType(StrEnum):
    # ... existing ...
    SYNONYM = "synonym"
```

### C.6 `src/qortex/mcp/server.py` -- mode="hipporag" in qortex_query

**File**: `/Users/peleke/Documents/Projects/qortex-track-c/src/qortex/mcp/server.py`

The MCP server is currently a stub (entirely commented out). When it is implemented (Phase 5), the `qortex_query` tool should accept a `mode` parameter:

```python
@server.tool()
def qortex_query(
    context: str,
    domains: list[str] | None = None,
    top_k: int = 10,
    mode: str = "auto",  # "auto" | "vec" | "hipporag"
) -> list[dict]:
    """Retrieve relevant rules for a context.

    Args:
        mode: Retrieval mode.
            "auto" - Use HippoRAG if available, fall back to vec
            "vec" - Pure vector similarity (Phase 0)
            "hipporag" - Full HippoRAG with PPR (Phase 4)
    """
    if mode == "hipporag" or (mode == "auto" and hipporag_adapter):
        adapter = hipporag_adapter
    else:
        adapter = vec_adapter

    return adapter.retrieve(context, domains, top_k)
```

This is a small change (~20 LOC) but depends on the MCP server being implemented first (Phase 5).

### C.7 `tests/test_hippocampus.py` -- Test Suite

**File**: `/Users/peleke/Documents/Projects/qortex-track-c/tests/test_hippocampus.py` (new file)

Tests to write:

```python
"""Tests for HippoRAG-style retrieval (Phase 4)."""

class TestSeedNodeFinding:
    """Test _find_seed_nodes with vector index."""

    def test_finds_seeds_by_embedding_similarity(self): ...
    def test_filters_by_domain(self): ...
    def test_filters_by_threshold(self): ...
    def test_fallback_to_string_matching(self): ...
    def test_returns_empty_when_no_matches(self): ...

class TestPPR:
    """Test Personalized PageRank implementation."""

    def test_seed_nodes_have_highest_scores(self): ...
    def test_connected_nodes_score_higher_than_distant(self): ...
    def test_bridge_nodes_score_high(self):
        """Node connecting two seeds should score highly."""
        ...
    def test_convergence(self):
        """Scores should stabilize within max_iterations."""
        ...
    def test_damping_factor_effect(self):
        """Lower damping = more concentrated around seeds."""
        ...
    def test_teleportation_factors_modulate_scores(self): ...

class TestTeleportationFactors:
    """Test feedback -> factor update loop."""

    def test_accepted_increases_factor(self): ...
    def test_rejected_decreases_factor(self): ...
    def test_partial_mildly_increases(self): ...
    def test_factor_clamped_to_bounds(self): ...
    def test_persistence_roundtrip(self): ...
    def test_default_factor_is_one(self): ...

class TestRuleCollection:
    """Test _collect_rules from activated concepts."""

    def test_rules_linked_to_high_scoring_concepts_returned(self): ...
    def test_rules_ranked_by_concept_score(self): ...
    def test_top_k_limits_output(self): ...
    def test_rules_from_unactivated_concepts_excluded(self): ...

class TestEndToEnd:
    """Integration test: query -> seeds -> PPR -> rules."""

    def test_full_retrieval_pipeline(self):
        """Build small graph, query, get relevant rules."""
        ...
    def test_cross_domain_retrieval(self): ...
    def test_feedback_improves_subsequent_queries(self): ...

class TestHippoRAGAdapter:
    """Test the adapter protocol implementation."""

    def test_implements_retrieval_adapter(self): ...
    def test_retrieve_delegates_to_hippocampus(self): ...
    def test_feedback_delegates_to_hippocampus(self): ...
```

Estimated: ~300 LOC of tests.

---

## D. Open Questions and Design Decisions

### D.1 What to Keep from HippoRAG vs. Adapt

| HippoRAG Feature | Keep/Adapt/Skip | Rationale |
|---|---|---|
| PPR for pattern completion | **Keep** | Core algorithm, well-proven |
| Knowledge graph as index | **Keep** | Already built in qortex |
| Embedding-based seed finding | **Keep** | Requires Phase 0 vector layer |
| Synonym edge detection | **Adapt** | Add as post-ingestion step or enrichment |
| Query-to-triple matching (v2) | **Skip for MVP** | Requires embedding triples separately; add in Phase 4.1 |
| Recognition memory / LLM filtering (v2) | **Skip for MVP** | Adds latency; threshold filtering is sufficient initially |
| Passage nodes (v2) | **Skip** | qortex retrieves rules, not passages |
| Schema-less relations | **Adapt** | Keep qortex's typed relations; they are strictly more informative |
| Teleportation from feedback | **Novel addition** | Not in HippoRAG; qortex's contribution |

### D.2 Where We Diverge from the Paper

1. **Retrieval target**: HippoRAG retrieves *passages* to feed to an LLM for QA. qortex retrieves *rules* to feed to coding agents for guidance. This means we skip passage nodes and instead follow concept -> rule links.

2. **Teleportation modulation**: HippoRAG uses static teleportation (seed scores from query matching). qortex adds reward-modulated teleportation factors that evolve with consumer feedback. This is a novel contribution.

3. **Typed edges**: HippoRAG uses schema-less relation edges. qortex uses typed RelationTypes. For PPR, we initially treat all edge types equally (matching HippoRAG), but typed edges open future possibilities like weighting REQUIRES edges more heavily than SIMILAR_TO edges.

4. **Multi-domain**: HippoRAG operates on a single corpus. qortex has domain isolation with cross-domain bridging. PPR walks the full graph (all domains) but results can be filtered by domain.

5. **No OpenIE at query time**: HippoRAG v1 runs NER on the query; v2 embeds the query against triples. qortex embeds the full query against node embeddings. This is simpler and avoids the need for a separate NER model.

### D.3 Minimum Viable HippoRAG vs. Nice-to-Have

**MVP (Phase 4.0)**:
- Embedding-based seed node finding (depends on Phase 0)
- Real power iteration PPR in InMemoryBackend
- Rule collection from activated concepts
- TeleportationFactors with JSON persistence
- HippoRAGAdapter implementing RetrievalAdapter protocol
- Damping factor changed from 0.85 to 0.5
- SYNONYM added to RelationType
- Test suite

**Phase 4.1 (Nice-to-Have)**:
- Query-to-triple matching (embed triples, not just nodes)
- LLM recognition memory for seed filtering
- Edge-type-weighted adjacency matrix (REQUIRES has different weight than SIMILAR_TO)
- Automatic synonym detection during ingestion (post-processing step)
- Teleportation factor visualization in CLI (`qortex inspect factors`)
- Factor decay over time (factors slowly regress to 1.0 if not reinforced)
- MemgraphBackend: native weighted PPR via Cypher power iteration

### D.4 Remaining Questions

1. **Embedding model choice**: Phase 0 defines the `EmbeddingModel` protocol. Which concrete model should be the default? Options: sentence-transformers (local, free), OpenAI ada-002 (API, accurate), or NV-Embed-v2 (what HippoRAG v2 uses, but large). Recommendation: Start with sentence-transformers for testing, make it pluggable.

2. **Synonym detection threshold**: HippoRAG uses a cosine similarity threshold (tau) for synonym edges. What value? HippoRAG does not publish the exact threshold. Recommendation: Start with tau=0.85, tune empirically.

3. **When to compute embeddings**: During ingestion (sync, blocks ingest) or as a background post-processing step (async, requires eventual consistency)? Recommendation: During ingestion for MVP (simpler), move to async in Phase 4.1 if ingestion latency becomes a problem.

4. **Teleportation factor granularity**: Per-concept or per-rule? Per-concept is more general (one concept affects many rules) but per-rule would be more precise. Recommendation: Per-concept (matches the PPR model where teleportation is over nodes).

5. **Integration with causal module**: qortex-track-c has a `causal/` module with credit assignment. Should teleportation factor updates flow through the causal DAG? Recommendation: Not for MVP. The causal module and teleportation factors can be integrated in a future phase where credit assignment propagates through both the causal DAG and the KG.

---

## References

- [HippoRAG v2 (arXiv:2502.14802)](https://arxiv.org/abs/2502.14802) - "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"
- [HippoRAG v1 (arXiv:2405.14831)](https://arxiv.org/abs/2405.14831) - "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models" (NeurIPS 2024)
- [HippoRAG GitHub Repository](https://github.com/OSU-NLP-Group/HippoRAG)
- [Rossi & Gleich 2012](https://arxiv.org/abs/1203.6098) - "Dynamic PageRank Using Evolving Teleportation"
- [Rossi & Gleich 2014](https://arxiv.org/abs/1211.4266) - "A Dynamical System for PageRank with Time-Dependent Teleportation"
- [HippoRAG v2 Analysis (MarkTechPost)](https://www.marktechpost.com/2025/03/03/hipporag-2-advancing-long-term-memory-and-contextual-retrieval-in-large-language-models/)
- [HippoRAG v2 Analysis (EmergentMind)](https://www.emergentmind.com/papers/2502.14802)

---

## File Summary

| File | Action | Estimated LOC |
|---|---|---|
| `src/qortex/hippocampus/retrieval.py` | **Major rewrite** | ~200 LOC (from ~224) |
| `src/qortex/hippocampus/factors.py` | **New file** | ~80 LOC |
| `src/qortex/hippocampus/adapter.py` | **New file** | ~40 LOC |
| `src/qortex/core/memory.py` | **Replace PPR method** | ~60 LOC changed |
| `src/qortex/core/backend.py` | **Add teleportation_factors param** | ~20 LOC changed |
| `src/qortex/core/models.py` | **Add SYNONYM to RelationType** | ~2 LOC |
| `src/qortex/mcp/server.py` | **Add mode param** (when MCP implemented) | ~20 LOC |
| `tests/test_hippocampus.py` | **New file** | ~300 LOC |
| **Total** | | **~720 LOC** |
