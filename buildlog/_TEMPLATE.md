# Build Journal: [TITLE]

**Date:** [YYYY-MM-DD]
**Duration:** [X hours]
**Status:** [Complete | Partial | Blocked]

---

## The Goal

[1-2 paragraphs: What are we building? Why does it matter? What problem does it solve?]

---

## What We Built

### Architecture

```
[ASCII diagram of the system/component]
```

### Components

| Component | Status | Notes |
|-----------|--------|-------|
| [Name] | [Working/Pending/Blocked] | [Brief note] |

---

## The Journey

### [Phase/Hour 1]: [Title]

**What we tried:**
[Approach taken]

**What happened:**
[Actual result, including errors]

```
[Actual error message or output]
```

**The fix:**
[What resolved it]

**Lesson:**
[One-liner takeaway]

---

### [Phase/Hour 2]: [Title]

[Repeat pattern...]

---

## Test Results

### [Test Name]

**Command:**
```bash
[Actual command run]
```

**Response:**
```json
[Actual response received]
```

**Result:** [Pass/Fail + brief assessment]

---

## Code Samples

### [Component/Function Name]

```[language]
[Key code snippet - not entire file, just the important parts]
```

[Brief explanation of why this code matters]

---

## What's Left

- [ ] [Pending task 1]
- [ ] [Pending task 2]
- [ ] [Blocker or dependency]

---

## Cost/Performance Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| [Time/Cost/Memory] | [Value] | [Context] |

---

## AI Experience Reflection

### What Worked Well

[Specific things about the collaboration that were effective]

### What Was Frustrating

[Pain points, unclear instructions, context loss]

### Communication Notes

[Observations about pace, tone, information density, interruptions]

---

## Improvements

*Actionable learnings for future work. This section accumulates knowledge that can improve agent practices over time.*

### Architectural

[Things to do differently in system design, abstractions, patterns]
- [e.g., "Should have used a plugin architecture from the start instead of hardcoding backends"]
- [e.g., "The retry logic belongs in a shared util, not duplicated across services"]

### Workflow

[Better approaches to the development process itself]
- [e.g., "Run the type checker before committing, not after"]
- [e.g., "Should have written the integration test first to clarify the API contract"]

### Tool Usage

[More effective ways to use available tools and capabilities]
- [e.g., "Use grep with -C flag for context instead of reading entire files"]
- [e.g., "The batch endpoint would have been faster than individual requests"]

### Domain Knowledge

[Things learned about the specific technology/domain that apply broadly]
- [e.g., "ComfyUI node IDs must be strings, not integers"]
- [e.g., "Supabase storage returns 400 not 404 for missing files"]

---

## Files Changed

```
path/to/
├── changed/
│   └── file.ext      # Brief description
└── another.ext       # Brief description
```

---

*Next entry: [Preview of what's coming]*
