# Part 3: The Projection Pipeline

You have a knowledge graph with 200 drug concepts and 450 relationships. Great.

Your clinical decision support system doesn't speak "graph." It speaks "rules." You need to transform:

```
(Aspirin) --REFINES--> (Anticoagulant)
(Warfarin) --REFINES--> (Anticoagulant)
```

Into:

```
"When prescribing Aspirin, check if patient is on other Anticoagulants (e.g., Warfarin)"
```

And you need to do it systematically, for every relevant edge, in consistent styles.

## The insight

The graph is the **primary representation**. Rules are just one **projection** of it.

Different consumers need different projections:
- A clinical system needs warning-style rules
- A training manual needs explanatory rules
- An audit log needs formal rules

Same graph, different outputs. That's the projection pattern.

## Source → Enricher → Target

qortex structures projections as a pipeline:

```
Graph ──→ [Source] ──→ [Enricher] ──→ [Target] ──→ Output
              │              │              │
         Extract rules    Add context    Serialize
         (explicit +      (antipattern,  (YAML, JSON,
          derived)        rationale)     Markdown)
```

**Source**: Extracts rules from the graph. Both explicit rules (stated in your data) and derived rules (generated from edges using templates).

**Enricher**: Adds context, antipatterns, and rationale. Can be template-based (fast, deterministic) or LLM-based (rich, contextual).

**Target**: Serializes to an output format. buildlog YAML, flat JSON, Markdown docs, whatever the consumer needs.

## Edge rule templates

qortex has 30 built-in templates: 3 variants for each of the 10 relationship types.

| Variant | Style | Example |
|---------|-------|---------|
| imperative | Direct command | "Ensure Aspirin is not combined with Warfarin" |
| conditional | When/then | "When prescribing Aspirin, verify no anticoagulant conflicts" |
| warning | Caution | "Combining Aspirin with Warfarin may increase bleeding risk" |

The templates are mechanical. Given an edge type and two concept names, you get a rule. No LLM needed.

## Running a projection

```python
from qortex.projectors.projection import Projection
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.enrichers.template import TemplateEnricher
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

projection = Projection(
    source=FlatRuleSource(backend=backend),
    enricher=TemplateEnricher(domain="pharmacology"),
    target=BuildlogSeedTarget(persona_name="drug_safety_rules"),
)

result = projection.project(domains=["pharmacology"])
# result is a dict in the universal schema format
```

One line to go from graph to actionable rules.

## The output

```yaml
persona: drug_safety_rules
version: 1
rules:
  - rule: "Aspirin refines Anticoagulant; verify compatibility with other anticoagulants"
    category: pharmacology
    provenance:
      id: derived:aspirin->anticoagulant:warning
      domain: pharmacology
      derivation: derived
      confidence: 0.95
      relation_type: refines
      template_id: refines:warning
metadata:
  source: qortex
  rule_count: 1
```

Notice the **provenance** block. It tracks where the rule came from: which edge, which template, what confidence. Full audit trail.

## Why buildlog isn't special

The `BuildlogSeedTarget` is just one target. You could write a `MarkdownDocTarget` or `JSONAPITarget` the same way.

buildlog happens to be the first consumer we built for. But the architecture doesn't privilege it. Any system that understands the universal schema can consume qortex projections.

## What you learned

- The graph is the primary representation; rules are projections of it
- Projections follow Source → Enricher → Target pattern
- 30 edge rule templates generate rules mechanically from edges
- Provenance tracks the full derivation chain
- buildlog is one consumer, not a privileged one

## Next

[Part 4: The Consumer Loop](part4-consumer-loop.md): Who uses these rules, and how do you know if they work?
