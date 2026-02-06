# Skill: Chapter Generator

> **Purpose**: Generate a single tutorial chapter from a structured outline.
> **Invoked by**: Curriculum planner skill or directly with chapter spec.
> **Output**: Jupyter notebook (.ipynb) or Markdown file following aegir voice.

---

## Trigger Phrases

- "Generate chapter 3.1 from the outline"
- "Create the notebook for [chapter title]"
- "Write chapter [N] following the spec"
- "/chapter-gen [outline-path] [chapter-number]"

---

## Required Input

The skill expects a chapter specification with these fields:

```yaml
chapter:
  number: "3.1"
  title: "What Do I Actually Have?"
  depth: deep  # or "light"

  opening_hook: |
    You just ingested Chapter 5... [the war story]

  learning_objectives:
    - Basic MATCH/RETURN - The query skeleton
    - Property access - c.name, c.description
    - CONTAINS vs exact match

  aha_moments:
    - "Oh, I was only searching name, not description"
    - "There are 47 concepts but only 12 have edges"

  key_queries:
    - description: "Their first attempt (fails)"
      cypher: "MATCH (c:Concept) WHERE c.name CONTAINS 'setter' RETURN c"
      expected_result: "3 nodes"
    - description: "Fixed: search description too"
      cypher: |
        MATCH (c:Concept)
        WHERE toLower(c.name) CONTAINS 'setter'
           OR toLower(c.description) CONTAINS 'setter'
        RETURN c.name, c.description
      expected_result: "12 nodes"

  closing_bridge: |
    47 concepts. But only 32 have edges...

  visual_hooks:
    - type: bar_chart
      description: "concepts with edges vs orphans"

  falsifiable_claim:
    statement: "Searching name-only misses >50% of relevant concepts"
    test: "Query both, compare counts"

  jargon_introduced:
    - term: "MATCH/RETURN"
      earned_by: "here's how you ask questions"
```

---

## Generation Process

### Phase 1: Structure

1. Create notebook skeleton with aegir cell structure
2. Insert metadata cell (title, arc, prerequisites, time estimate)
3. Insert imports cell (if needed for visualizations)

### Phase 2: Narrative Opening

1. Insert hero image placeholder (or generate with ComfyUI if available)
2. Write opening hook following aegir story-first pattern
3. Ensure cognitive tension ("this is wrong!" energy)

### Phase 3: Content Cells

For each learning objective:

1. **Recap** - What we just learned / where we are
2. **Scenario** - Concrete example motivating the concept
3. **Code cell** - The query/code to try
4. **Expected output** - What they should see
5. **Aha moment** - The insight that lands
6. **Bridge** - Transition to next concept

### Phase 4: Exercises

If `depth: deep`:
- Include 2-3 exercises with:
  - Problem statement
  - Hints (for early chapters)
  - Solution cell (collapsed)
  - Verification cell

If `depth: light`:
- Include 1 quick exercise or skip

### Phase 5: Closing

1. Insert closing bridge (transition to next chapter)
2. Add "What You Learned" summary
3. Add "Next Up" link

### Phase 6: Visual Hooks

Insert visual hooks every 2-3 markdown cells:
- Generate matplotlib/seaborn code for charts
- Insert mermaid diagrams for flows
- Add "run this and play" interactive cells

---

## Voice Rules (from aegir)

- **Story-first**: Open with incident, not abstraction
- **Code-before-formula**: Compute it → verify it → name it
- **Recap aggressively**: Assume reader forgot 3 cells ago
- **Jargon-earning**: Never use a term before earning it
- **Personality**: Irreverent, direct, "you absolute walnut" energy when appropriate
- **No em-dashes**: Use colons, commas, parentheses
- **Visual every 2-3 cells**: Never go long without something to look at

---

## Output Format

### Jupyter Notebook (.ipynb)

```
notebooks/
  part3-graph-exploration/
    3.1-what-do-i-have/
      3.1-what-do-i-have.ipynb
      hero-opening.png
    3.2-counting-grouping/
      ...
```

### Markdown (.md)

```
docs/tutorials/
  part3-graph-exploration/
    3.1-what-do-i-have.md
    3.2-counting-grouping.md
    ...
```

---

## Example Invocation

```
/chapter-gen docs/tutorials/PART3-GRAPH-EXPLORATION-OUTLINE.md 3.1
```

Or with inline spec:

```
Generate chapter 3.1 "What Do I Actually Have?" following this spec:

opening_hook: |
  You just ingested Chapter 5...

[rest of spec]
```

---

## Quality Checklist

- [ ] Opens with story/hook, not abstract framing
- [ ] Every code cell runs without error
- [ ] Aha moments land after building intuition
- [ ] Visual hook every 2-3 markdown cells
- [ ] Jargon introduced only after earning
- [ ] Closing bridges to next chapter
- [ ] Exercises have solutions (if deep chapter)
- [ ] No em-dashes
- [ ] Personality shows through
