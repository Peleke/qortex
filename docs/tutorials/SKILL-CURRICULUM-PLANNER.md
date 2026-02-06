# Skill: Curriculum Planner

> **Purpose**: Run an interactive elicitation to design a tutorial series, producing chapter specs that can be passed to the Chapter Generator skill.
> **Output**: Structured outline file + individual chapter specs ready for generation.
> **Invokes**: Chapter Generator skill (for each chapter, or as batch).

---

## Trigger Phrases

- "Plan a tutorial series on [topic]"
- "Design a curriculum for [audience] learning [skill]"
- "I want to teach [concept] - help me plan the chapters"
- "/curriculum-plan [topic]"
- "Elicit a tutorial structure for [domain]"

---

## Elicitation Protocol

### Dimension 1: AUDIENCE

**Questions:**
- Who is the reader? (curious engineer, optimizer, architect, beginner, expert)
- What do they already know? (anchors for new concepts)
- What's their goal? (explore, build, optimize, understand)

**Extract:**
- Persona name (e.g., "curious engineer")
- Prior knowledge anchors
- End-state capability

### Dimension 2: NARRATIVE ARC

**Questions:**
- What's the opening hook? (What problem creates cognitive tension?)
- What's the climax? (The surprising insight or reversal?)
- What's the cliffhanger? (What question leads to the next series?)

**Extract:**
- Opening war story / scenario
- The "aha" surprise
- Bridge to future content

### Dimension 3: CHAPTER BREAKDOWN

**Questions:**
- What are the 4-8 major teaching beats?
- Which chapters go deep (exercises, theory backfill)?
- Which chapters are transitional (light, keep moving)?

**Extract:**
- Chapter list with titles
- Depth markers (deep/light)
- Dependencies between chapters

### Dimension 4: DATA & EXAMPLES

**Questions:**
- What real data do we have?
- What supplements are needed for teaching?
- What running example carries through?

**Extract:**
- Data source path
- Supplement list (with plausibility notes)
- Running example description

### Dimension 5: VOICE & STYLE

**Questions:**
- What's the tone? (aegir-style irreverent, formal, conversational)
- Any specific voice rules? (no em-dashes, personality level)
- What's the teaching philosophy? (code-first, theory-first, story-first)

**Extract:**
- Voice guide reference (e.g., aegir SKILL.md)
- Specific style rules
- Philosophy statement

---

## Elicitation Flow

```
1. OPEN
   "Let's plan a tutorial series. Who's the reader and what are they trying to learn?"

2. ANCHOR
   "Where does this start? What does the reader have/know when they begin?"

3. ARC
   "What's the opening hook - the 'wtf this is wrong' moment?"
   "What's the surprise they'll discover?"
   "Where does it lead them wanting more?"

4. STRUCTURE
   "Let me propose [N] chapters. Which feel deep vs light?"
   [Interactive refinement]

5. DATA
   "What real data do we use? What do we need to add?"

6. CONFIRM
   "Here's the full structure. Ready to generate chapter specs?"
```

---

## Output: Curriculum Outline

```yaml
# CURRICULUM-OUTLINE.yaml

metadata:
  title: "Part 3: Exploring Your Knowledge Graph"
  audience: "curious engineer"
  estimated_time: "90 minutes"
  voice_guide: "/path/to/aegir/SKILL.md"
  data_source: "Ch5 extraction"

narrative_arc:
  opening_hook: |
    "wtf why only 3 nodes"
  climax: |
    "Properties > Encapsulation?! The hub isn't what I expected"
  cliffhanger: |
    "How do I rank paths? We need... Personalized PageRank"

chapters:
  - number: "3.1"
    title: "What Do I Actually Have?"
    depth: deep
    opens_with: "3 nodes wtf"
    closes_with: "47 concepts but only 12 have edges"

  - number: "3.2"
    title: "Counting and Grouping"
    depth: light
    opens_with: "Which relation types dominate?"
    closes_with: "REQUIRES is 40%"

  # ... more chapters

supplements:
  - concept: "ProductionBug"
    reason: "Endpoint for path examples"
    plausibility: "High: natural consequence of InvalidState"
  - edge: "InvalidState -[:CAUSES]-> ProductionBug"
    reason: "Complete the causal chain"
    plausibility: "High: direct implication"

style_rules:
  - "Story-first opening"
  - "Code-before-formula"
  - "Visual every 2-3 cells"
  - "No em-dashes"
  - "Recap aggressively"
```

---

## Output: Chapter Specs (per chapter)

For each chapter, generate a spec file that the Chapter Generator can consume:

```yaml
# chapter-3.1-spec.yaml

chapter:
  number: "3.1"
  title: "What Do I Actually Have?"
  depth: deep

  opening_hook: |
    You just ingested Chapter 5 of "Software Design for Python Programmers."
    The CLI said "47 concepts, 83 edges, 7 rules." Cool.

    Now you want to find everything about setters.

    MATCH (c:Concept) WHERE c.name CONTAINS 'setter' RETURN c

    3 nodes. That's it.

    But you KNOW that chapter had 15 pages on dangerous setters...

  learning_objectives:
    - "Basic MATCH/RETURN - The query skeleton"
    - "Property access - c.name, c.description, c.domain"
    - "CONTAINS vs exact match - Why their query was too narrow"
    - "Case sensitivity - toLower() for robust matching"
    - "LIMIT and ordering - Not drowning in output"

  aha_moments:
    - "Oh, I was only searching name, not description"
    - "The concepts are there, they just have different names"
    - "There are 47 concepts but only 32 have edges"

  key_queries:
    - description: "Their first attempt (fails)"
      cypher: "MATCH (c:Concept) WHERE c.name CONTAINS 'setter' RETURN c"
      expected: "3 nodes"

    - description: "Fixed: search description too"
      cypher: |
        MATCH (c:Concept)
        WHERE toLower(c.name) CONTAINS 'setter'
           OR toLower(c.description) CONTAINS 'setter'
        RETURN c.name, c.description
      expected: "12 nodes"

  closing_bridge: |
    47 concepts. But only 32 have edges. That means 15 are orphans.
    And you still don't know what KINDS of relationships exist. Let's count.

  visual_hooks:
    - type: "bar_chart"
      data: "concepts with edges vs orphans"

  falsifiable_claim:
    statement: "Searching name-only misses >50% of relevant concepts"
    test: "Query both, compare counts"
```

---

## Recursive Invocation

After generating specs, the Curriculum Planner can invoke Chapter Generator:

```
# Generate all chapters
for spec in chapter-specs/*.yaml:
  /chapter-gen $spec

# Or generate one at a time with review
/chapter-gen chapter-3.1-spec.yaml
[review output]
/chapter-gen chapter-3.2-spec.yaml
...
```

---

## Workflow Diagram

```
User: "Plan a tutorial on graph exploration"
    ↓
[Curriculum Planner: Elicitation]
    ↓
CURRICULUM-OUTLINE.yaml + chapter-*.spec.yaml
    ↓
[Chapter Generator: for each spec]
    ↓
3.1-what-do-i-have.ipynb
3.2-counting-grouping.ipynb
...
```

---

## Quality Checklist (for the planner)

- [ ] All 5 elicitation dimensions covered
- [ ] Narrative arc has tension (open, climax, cliffhanger)
- [ ] Chapters have clear depth markers
- [ ] Supplements are plausible and noted
- [ ] Voice rules are explicit
- [ ] Each chapter spec is self-contained (can generate independently)
- [ ] Cross-references to related materials included
