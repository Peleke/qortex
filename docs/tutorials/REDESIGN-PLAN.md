# Qortex Theory Tutorials Redesign: From Pharma to Software Design

## Executive Summary

**The Minitheory: "The Dangerous Setter Cascade"**

Replace the aspirin/warfarin/bleeding-risk drug interaction example with a software design example: **Dangerous Setters leading to Invalid State leading to Production Crash**.

This minitheory uses content from Chapters 5 and 6 of "Software Design for Python Programmers":
- **Dangerous Setters** (Ch5 S5.5): setter methods that can put objects into invalid states
- **Properties** (Ch5 S5.2): controlled access via @property decorators
- **Immutability** (Ch5 S5.4): objects that cannot be modified after construction
- **Programming by Contract** (Ch6 S6.3): preconditions, postconditions, class invariants
- **Principle of Least Astonishment** (Ch6 S6.1): no surprises for code users
- **Encapsulation** (Ch5): the foundational concept everything connects to

**The War Story Setup (replaces Mrs. Chen in the hospital):**

> You're on-call. 3am. PagerDuty fires. The recommendation engine is returning garbage: users seeing products from completely wrong categories. You check the logs. No exceptions. No errors. The service is confidently, politely, *expensively* wrong.
>
> You trace it to a `Product` object that somehow has `category="Electronics"` but `subcategory="Fresh Produce"`. That's... impossible. The subcategory REQUIRES a valid parent category.
>
> The junior dev on the team added a `set_subcategory()` method last sprint. It worked great in isolation. But combined with the existing `set_category()` method, you can now put a Product into an invalid state by calling them in the wrong order. Or calling set_subcategory when category is None. Or a dozen other paths that "shouldn't happen" but did.
>
> Your million-dollar recommendation engine couldn't answer a simple question: "Is this product valid?" because the validity check looks at category and subcategory independently. No single check says "Electronics can't have Fresh Produce subcategories." The dangerous setter created a crack in reality that your embeddings couldn't see.

**Why This Works:**

1. **Life-affecting stakes**: Not life-or-death, but career-affecting. Devs viscerally understand "3am PagerDuty + broken prod"
2. **Confidently wrong**: Same pattern as pharma: the system has all the data, doesn't see the connection
3. **Multi-hop required**: `DangerousSetter -> InvalidState -> BrokenInvariant -> ProductionCrash`
4. **Vector similarity fails**: An embedding of "set_subcategory" has no similarity to "production crash"
5. **Graph traversal succeeds**: The knowledge graph connects: `DangerousSetter --CHALLENGES--> Properties --IMPLEMENTS--> Encapsulation`, and `Invariant --PREVENTS--> InvalidState`

---

## Part-by-Part Redesign

---

## Part 1: The Multi-Hop Problem

**Teaching Concept (unchanged):** Vector similarity finds documents that "look like" your query. Multi-hop questions need documents that are "connected to" your query through concepts the embedding never learned.

**War Story (new):**

The 3am PagerDuty story above. A junior dev adds a setter. Tests pass. Deploy. Crash. The embedding-based "code quality assistant" you built says the setter is fine because it's semantically similar to other valid setters. It can't see that this PARTICULAR setter, in THIS codebase, with THESE existing methods, creates a path to invalid state.

**The Failure Pattern (reframed):**

> You asked your code assistant: "Is this `set_subcategory` method safe to add?"
>
> The system searched for similar methods. Found 47 setter examples. All looked fine. "This setter follows common patterns. Ship it."
>
> What it missed:
> - `set_subcategory` + existing `set_category` = invalid state path
> - No single document mentions both methods together
> - The danger emerges from the COMBINATION, not from either method alone

**The Fix (preview):**

```
(Dangerous Setter) --CHALLENGES--> (Encapsulation)
(Dangerous Setter) --ENABLES--> (Invalid State)
(Invalid State) --VIOLATES--> (Class Invariant)
(Class Invariant) --PREVENTS--> (Production Bug)
```

When you query about the new setter, you traverse the graph and discover it connects to "Production Bug" through a three-hop chain.

**Falsifiable Claim:**

> Vector similarity will find 0 documents connecting "set_subcategory" to "production crash" when queried directly.
> Graph traversal will find at least 1 path connecting them via intermediate concepts.

---

## Part 2: Knowledge Graphs 101

**Teaching Concept (unchanged):** Concepts, edges, semantic types, domains.

**Example (new):**

Back to the setter problem. What do we know about `DangerousSetter`?

```
(DangerousSetter) --IS_A--> (Setter Method)
(DangerousSetter) --ENABLES--> (Invalid State)
(DangerousSetter) --CONTRADICTS--> (Immutability)
(DangerousSetter) --CHALLENGES--> (Encapsulation)
```

And about `ClassInvariant`:

```
(ClassInvariant) --PREVENTS--> (Invalid State)
(ClassInvariant) --REQUIRES--> (Precondition)
(ClassInvariant) --REQUIRES--> (Postcondition)
```

Look at that. `DangerousSetter` ENABLES `Invalid State`, and `ClassInvariant` PREVENTS it. The tension is visible in the structure.

**Your First Knowledge Graph (code example):**

```python
from qortex.core.models import ConceptNode, ConceptEdge, RelationType
from qortex.core.memory import InMemoryBackend

backend = InMemoryBackend()
backend.connect()
backend.create_domain("software_design", "Design principles from SD4PP")

backend.add_node(ConceptNode(
    id="dangerous_setters",
    name="Dangerous Setters",
    description="Setter methods that can put an object into an invalid state",
    domain="software_design",
    source_id="ch5",
))

backend.add_node(ConceptNode(
    id="class_invariant",
    name="Class Invariant",
    description="A condition that must be true for all objects before and after method calls",
    domain="software_design",
    source_id="ch6",
))

backend.add_edge(ConceptEdge(
    source_id="dangerous_setters",
    target_id="class_invariant",
    relation_type=RelationType.CHALLENGES,
))
```

**Semantic Types Table (update examples):**

| Type | Meaning | Example |
|------|---------|---------|
| REQUIRES | A needs B | Properties REQUIRES Encapsulation |
| CONTRADICTS | A and B conflict | Dangerous Setters CONTRADICTS Immutability |
| REFINES | A is a specific form of B | Law of Demeter REFINES Principle of Least Knowledge |
| IMPLEMENTS | A is a concrete form of B | @property decorator IMPLEMENTS Encapsulation |
| CHALLENGES | A introduces problems for B | Dangerous Setters CHALLENGES Class Invariant |

---

## Part 3: The Projection Pipeline

**Teaching Concept (unchanged):** Graph -> Rules via Source -> Enricher -> Target. The graph is the primary representation; rules are projections of it.

**War Story Opening:**

> Two weeks after the 3am incident. You've built a knowledge graph of software design concepts. 200 nodes. 450 edges. The CTO asks: "How do I prevent this from happening again?"
>
> She doesn't speak "graph." She speaks "rules." She wants guardrails for the code review process.

**The Insight:**

Transform:
```
(Dangerous Setters) --CHALLENGES--> (Class Invariant)
```

Into:
```yaml
- rule: "When adding a setter method, verify it cannot violate any class invariant"
  category: safety
  severity: high
```

**Edge Rule Templates (examples with new domain):**

| Edge Type | Variant | Example Rule |
|-----------|---------|--------------|
| CHALLENGES | warning | "Dangerous Setters may undermine Class Invariant. Review carefully." |
| REQUIRES | conditional | "When implementing Properties, ensure Encapsulation is in place." |
| CONTRADICTS | imperative | "Do not combine Dangerous Setters with Immutability patterns." |

**Code Example (update):**

```python
from qortex.projectors.projection import Projection
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.enrichers.template import TemplateEnricher
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

projection = Projection(
    source=FlatRuleSource(backend=backend),
    enricher=TemplateEnricher(domain="software_design"),
    target=BuildlogSeedTarget(persona_name="code_review_guardrails"),
)

result = projection.project(domains=["software_design"])
```

**Output Example (new):**

```yaml
persona: code_review_guardrails
version: 1
rules:
  - rule: "Dangerous Setters challenges Class Invariant; ensure setter cannot create invalid state"
    category: software_design
    provenance:
      id: derived:dangerous_setters->class_invariant:warning
      domain: software_design
      derivation: derived
      confidence: 0.85
      relation_type: challenges
      template_id: challenges:warning
```

---

## Part 4: The Consumer Loop

**Teaching Concept (unchanged):** Rules are hypotheses. The consumer loop tests them via feedback (Thompson Sampling, Repeated Mistake Rate).

**War Story Opening:**

> You've projected 150 rules from your software design knowledge graph. They're sitting in a YAML file. The CTO is impressed. Then she asks the hard question: "Do these actually prevent bugs?"
>
> You don't know. You HOPE they help. But hope isn't engineering.

**The Loop (reframed for software):**

```
qortex projects rules
    ↓
buildlog ingests seeds
    ↓
AI code reviewer uses rules in PR reviews
    ↓
Developer logs when rule caught a real bug (reward)
Developer logs when rule was a false positive (penalty)
    ↓
Thompson Sampling updates posteriors
    ↓
High-confidence rules surface more often
    ↓
Bug Prevention Rate measured
```

**Concrete Example:**

> Rule: "When adding a setter, verify it cannot violate class invariant"
>
> Week 1: Surfaced 12 times. Caught 3 real bugs. 2 false positives. 7 neutral.
> Posterior: α=4, β=3 (more useful than not, but noisy)
>
> Week 4: Surfaced 47 times. Caught 15 bugs. 5 false positives. 27 neutral.
> Posterior: α=16, β=6 (clearly useful, surface more often)

**Falsifiable Claim:**

> If the rule doesn't reduce the bug rate after 50 surfacings with α/(α+β) < 0.3, deprecate it.

---

## Part 5: Pattern Completion

**Teaching Concept (unchanged):** Personalized PageRank as algorithmic equivalent of hippocampal pattern completion.

**War Story Opening:**

> A new developer asks: "What should I watch out for when adding a setter to a class with existing validation?"
>
> Your system needs to:
> 1. Match "setter" to `Setter Method` and `Dangerous Setters`
> 2. Match "validation" to `Precondition`, `Postcondition`, `Class Invariant`
> 3. Discover that setters connect to "Invalid State" which connects to "Class Invariant"
> 4. Return rules about all these connected concepts

The grandmother's perfume analogy. Replace with:

> See the word "singleton" in code. What happens?
>
> You don't just think "singleton." You think of global state. Thread safety. Testing nightmares. That one legacy codebase where every class was a singleton for no reason.
>
> That's pattern completion. The partial cue activates a network.

**Algorithm Example (new context):**

```
Query: "adding setter to validated class"

Step 1: Match to graph nodes
   - setter → Setter Method, Dangerous Setters
   - validated → Class Invariant, Precondition

Step 2: Random walks from these seeds
   - Dangerous Setters → connects to → Invalid State
   - Invalid State → VIOLATES → Class Invariant
   - Class Invariant was already a seed (reinforced!)

Step 3: After many walks, nodes have scores
   - Invalid State: high score (connects both query concepts)
   - Class Invariant: high score (seed + reinforced)
   - Dangerous Setters: high score (seed)
   - Encapsulation: medium score (hub node)
   - Law of Demeter: low score (barely connected)

Step 4: Retrieve rules from high-scoring nodes
   - "Setters may create invalid state paths..."
   - "Class invariants must hold before and after..."
```

**The Party Analogy (adapted):**

> You're looking for someone at a tech conference. You know two things: they work on validation and they've had production incidents from state bugs.
>
> You could ask everyone "do you work on validation AND have you had state bugs?" (vector similarity)
>
> Or you could find the validation folks and ask "who here has had state bugs?", find the incident responders and ask "any of you work on validation?" The overlap is your answer. (pattern completion)

---

## Part 6: HippoRAG First Principles

**Teaching Concept (unchanged):** The full algorithm: indexing extracts discrete representations, retrieval spreads activation.

**Indexing Example (new):**

```
Chapter 5: "Hide class implementations"
    ↓
[Extract triples via LLM]
    ↓
("Dangerous Setters", "CHALLENGES", "Encapsulation")
("Immutability", "CONTRADICTS", "Dangerous Setters")
("Properties", "IMPLEMENTS", "Encapsulation")
    ↓
[Build knowledge graph]
    ↓
Nodes: Dangerous Setters, Encapsulation, Immutability, ...
Edges: CHALLENGES, CONTRADICTS, IMPLEMENTS, ...
    ↓
[Link nodes to source passages]
    ↓
Dangerous Setters → Section 5.5, paragraphs 3-7
    ↓
Ready for retrieval
```

**Retrieval Example (new):**

```
Query: "can I add a set_subcategory method to Product?"
    ↓
[Extract query entities]
    ↓
["setter", "method", "Product"]
    ↓
[Match to graph nodes]
    ↓
Setter Method (0.95)
Dangerous Setters (0.82)
    ↓
[Run PPR from matched nodes]
    ↓
High: Dangerous Setters, Invalid State, Class Invariant
    ↓
[Retrieve passages from high-scoring nodes]
    ↓
"Setter methods that can put an object into an invalid state by modifying fields individually" (Section 5.5)
```

**Why This Beats Standard RAG (table updated):**

| Query | Standard RAG | HippoRAG |
|-------|--------------|----------|
| "Is set_subcategory safe?" | Returns examples of other setters | Returns warnings about dangerous setters + class invariant connections |
| "Why did the product become invalid?" | Finds nothing (no doc mentions this specific bug) | Traverses Dangerous Setters → Invalid State → Class Invariant |

---

## New Diagrams Needed

1. **Part 1 opener:** Mermaid diagram showing the setter → invalid state → crash chain
2. **Part 2:** Graph visualization with Encapsulation as the hub, connecting to Properties, Dangerous Setters, Law of Demeter, Immutability
3. **Part 3:** Source → Enricher → Target pipeline diagram with software design examples
4. **Part 5:** PPR visualization showing activation spreading from "setter" and "validation" seed nodes
5. **Part 6:** Two-phase diagram adapted for book ingestion

---

## Code Snippets to Update

All existing code examples should use:
- Domain: `"software_design"` instead of `"pharmacology"`
- Concepts: `Encapsulation`, `Dangerous Setters`, `Properties`, `Class Invariant`, `Immutability`
- Source ID: `"ch5"` or `"ch6"`

---

## Transition Recaps Between Parts

**Part 1 → Part 2:**
> You've seen the problem: the setter looked fine in isolation, but combined with other methods, it created a path to chaos. The two-million-dollar system couldn't connect the dots. Now let's build the structure that can.

**Part 2 → Part 3:**
> You've got concepts and edges. Encapsulation connects to Properties connects to Dangerous Setters. The shape is visible. But a graph sitting in memory doesn't stop bad PRs from merging. You need to extract actionable rules.

**Part 3 → Part 4:**
> You've projected 150 rules. Impressive YAML. But are they actually helpful? Or did you just generate text that sounds authoritative but changes nothing? Time to close the loop.

**Part 4 → Part 5:**
> The feedback loop works. Rules that help get surfaced more. But you're still doing keyword matching to find relevant rules. What if a developer asks about "adding methods to a validated class" and none of your rules mention those exact words?

**Part 5 → Part 6:**
> You've learned the algorithmic trick: spread activation, find what's connected. Now let's put it all together: from book chapter to retrievable knowledge.

---

## Falsifiable Claims Throughout

| Part | Claim | Threshold |
|------|-------|-----------|
| 1 | Vector search returns 0 docs connecting "setter" to "crash" | 0 relevant docs |
| 2 | Graph structure reveals Dangerous Setters → Invalid State → Crash path | Path exists |
| 3 | Template derivation produces ≥10 rules from the 8 concepts + 7 edges | ≥10 rules |
| 4 | After 50 surfacings, useful rules have α/(α+β) > 0.5 | >50% utility rate |
| 5 | PPR finds Class Invariant from {"setter", "validation"} seeds | Non-zero score |
| 6 | Multi-hop retrieval answers "is this setter safe?" when standard RAG fails | Correct answer |

---

## Critical Files for Implementation

- `docs/tutorials/part1-multi-hop-problem.md`: Core narrative rewrite; war story lives here
- `docs/tutorials/part2-knowledge-graphs.md`: Concept/edge examples to replace with software design
- `tests/test_e2e_book_to_buildlog.py`: Existing fixture data to reference (CONCEPTS, EDGES, EXPLICIT_RULES from Ch5)
- `aegir/.claude/skills/lesson-generator/SKILL.md`: Voice guide to follow strictly
- `data/books/ch05_5_Hide_class_implementations.txt`: Source material for examples (Dangerous Setters, Properties, Immutability)
