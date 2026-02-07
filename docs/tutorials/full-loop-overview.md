# Buildlog Case Study: The Full Loop

This case study walks through the complete bidirectional knowledge flow between qortex and buildlog. Design pattern knowledge from books influences AI agent behavior. Agent mistakes feed back into the knowledge graph. The loop closes.

## The Architecture

```
                    BIDIRECTIONAL KNOWLEDGE FLOW
----------------------------------------------------------------

   FORWARD FLOW (Knowledge to Agents)
   ==================================

   Book Chapter
      | qortex ingest file --backend anthropic
      v
   Memgraph Knowledge Graph (concepts, edges, rules)
      | qortex project buildlog --domain X
      v
   YAML Seed File (universal schema)
      | buildlog ingest-seeds
      v
   Agent Personas (test_terrorist, security_karen, etc.)


   BACKWARD FLOW (Agents to Knowledge)
   ====================================

   Agent makes mistakes during work
      | buildlog emits mistake_manifest
      v
   ~/.buildlog/emissions/pending/
      | transform to qortex manifest
      v
   Memgraph (experiential domain)
      | cross-domain edges
      v
   Mistakes linked to relevant design patterns
```

## What This Demonstrates

1. **Forward flow**: Book chapters become agent rules. Knowledge extracted from design pattern literature shapes how AI agents review code.

2. **Backward flow**: Agent mistakes become graph nodes. Errors get linked to the design patterns that might prevent them.

3. **Cross-domain analytics**: PageRank and other graph algorithms reveal which concepts are most central, which patterns address which blind spots.

## Prerequisites

- Docker running with Memgraph + Lab
- qortex installed (`uv sync` in the repo)
- buildlog-template configured
- Anthropic API key in `.env.local`

## The Two Parts

**[Part I: Forward Flow](full-loop-forward.md)** — From book chapters to agent personas. We ingest a design patterns chapter, explore the resulting graph, project rules to buildlog format, and load them into the agent system.

**[Part II: Backward Flow](full-loop-backward.md)** — From agent mistakes to graph insights. We ingest 192 mistake manifests, link them to design pattern concepts, and discover which knowledge domains need reinforcement.
