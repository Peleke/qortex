# Part 2: Knowledge Graphs 101

Back to the drug interaction problem. You've decided to build a knowledge graph instead of just embedding documents.

You start with aspirin. What do you know about it?

- Aspirin is a drug
- Aspirin is an NSAID
- Aspirin is an anticoagulant
- Aspirin treats pain
- Aspirin treats inflammation
- Aspirin increases bleeding risk

Each of these is a **relationship** between aspirin and some other concept. Let's write them as triples:

```
(Aspirin) --IS_A--> (NSAID)
(Aspirin) --IS_A--> (Anticoagulant)
(Aspirin) --TREATS--> (Pain)
(Aspirin) --TREATS--> (Inflammation)
(Aspirin) --INCREASES_RISK--> (Bleeding)
```

Now warfarin:

```
(Warfarin) --IS_A--> (Anticoagulant)
(Warfarin) --TREATS--> (Blood Clots)
(Warfarin) --INCREASES_RISK--> (Bleeding)
```

Look at that. Aspirin and warfarin both connect to "Anticoagulant" and both connect to "Bleeding." The dangerous interaction is *visible in the structure*.

## Concepts and edges

A **concept** (node) is a thing: a drug, a condition, a property, a procedure.

An **edge** connects two concepts with a semantic relationship.

The relationship type matters. "Aspirin TREATS Pain" is different from "Aspirin CAUSES Pain." The verb carries meaning.

## Semantic types

qortex uses 10 relationship types:

| Type | Meaning | Example |
|------|---------|---------|
| REQUIRES | A needs B | Circuit Breaker REQUIRES Timeout |
| CONTRADICTS | A and B conflict | Retry CONTRADICTS Fail-Fast |
| REFINES | A is a specific form of B | Ibuprofen REFINES NSAID |
| IMPLEMENTS | A is a concrete form of B | Redis IMPLEMENTS Cache |
| PART_OF | A is a component of B | Handler PART_OF Middleware |
| USES | A depends on B | Service USES Database |
| SIMILAR_TO | A and B are analogous | Aspirin SIMILAR_TO Ibuprofen |
| ALTERNATIVE_TO | A can substitute for B | Tylenol ALTERNATIVE_TO Aspirin |
| SUPPORTS | A provides evidence for B | Study SUPPORTS Efficacy |
| CHALLENGES | A provides counter-evidence | Side Effect CHALLENGES Safety |

Rich typing is what enables automated rule derivation later. If you know the relationship *type*, you can generate rules mechanically.

## Domains

Concepts belong to **domains**, isolated subgraphs like schemas in a database.

You might have:
- A "pharmacology" domain with drug concepts
- An "error_handling" domain with software patterns
- A "legal" domain with compliance rules

Domains don't interact by default. This prevents cross-contamination when you ingest multiple sources.

## Your first knowledge graph

```python
from qortex.core.models import ConceptNode, ConceptEdge, RelationType
from qortex.core.memory import InMemoryBackend

backend = InMemoryBackend()
backend.connect()
backend.create_domain("pharmacology", "Drug interactions and effects")

# Add concepts
backend.add_node(ConceptNode(
    id="aspirin",
    name="Aspirin",
    description="NSAID with anticoagulant properties",
    domain="pharmacology",
    source_id="drug_database",
))

backend.add_node(ConceptNode(
    id="anticoagulant",
    name="Anticoagulant",
    description="Substance that prevents blood clotting",
    domain="pharmacology",
    source_id="drug_database",
))

# Add relationship
backend.add_edge(ConceptEdge(
    source_id="aspirin",
    target_id="anticoagulant",
    relation_type=RelationType.REFINES,  # Aspirin is a type of anticoagulant
))
```

You now have structure. Concepts connected by typed relationships.

But a graph sitting in memory doesn't help anyone. You need to *extract actionable rules* from it.

## What you learned

- Knowledge graphs store concepts (nodes) and relationships (edges)
- Edge types carry semantic meaning (REQUIRES vs CONTRADICTS)
- 10 relationship types enable automated rule derivation
- Domains isolate subgraphs to prevent cross-contamination

## Next

[Part 3: The Projection Pipeline](part3-projection-pipeline.md): How do you turn a graph into actionable rules?
