# Buildlog Integration

[Buildlog](https://peleke.github.io/buildlog/) is qortex's first consumer. It uses projected rules to power AI code review personas.

## The Integration

```
qortex (knowledge graph)     buildlog (agent system)
         │                            │
         │  qortex project buildlog   │
         │  ───────────────────────>  │
         │      YAML seed files       │
         │                            │
         │  mistake emissions         │
         │  <───────────────────────  │
         │      JSON manifests        │
```

**Forward flow**: Book chapters become agent rules. `qortex project buildlog` emits YAML seed files that buildlog ingests into reviewer personas.

**Backward flow**: Agent mistakes become graph nodes. Buildlog emits mistake manifests that qortex can ingest and link to design patterns.

## Quick Start

```bash
# 1. Project rules from qortex
uv run qortex project buildlog \
  --domain implementation_hiding \
  -p qortex_impl_hiding \
  -o ~/.qortex/seeds/pending/qortex_impl_hiding.yaml

# 2. Ingest into buildlog
cd path/to/buildlog-template
uv run buildlog ingest-seeds
```

## Full Documentation

See the [Buildlog Integration Guide](https://peleke.github.io/buildlog/guides/qortex-integration/) for:

- Seed file format and schema
- Emission data structures
- Attribution and reward system
- Analysis workflows

## Case Study

For a complete walkthrough of both flows, see the [Buildlog Case Study](../tutorials/full-loop-overview.md).
