# qortex

Knowledge graph ingestion engine for automated rule generation.

## What It Does

qortex transforms unstructured content (books, docs, PDFs) into a knowledge graph, then projects actionable rules for AI agents and other consumers.

```
📚 Sources    →    🧠 qortex    →    📋 Rules    →    🤖 Consumers
(PDF/MD/Text)      (Knowledge Graph)   (Universal Schema)  (buildlog, agents, CI)
```

## Core Features

- **Content Ingestion**: PDF, Markdown, and text into structured knowledge graphs
- **Rich Type System**: 10 semantic relation types (REQUIRES, CONTRADICTS, REFINES, etc.)
- **Rule Derivation**: 30 edge templates generate rules from concept relationships
- **Rule Enrichment**: Add context, antipatterns, rationale via templates or LLM
- **Universal Schema**: JSON Schema artifacts for any-language validation
- **Consumer Interop**: Hybrid pull/push protocol for rule distribution

## Quick Start

```bash
# Install
pip install qortex

# Or with all optional dependencies
pip install qortex[all]
```

```python
from qortex.core.memory import InMemoryBackend
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget
from qortex.projectors.projection import Projection

# Set up backend
backend = InMemoryBackend()
backend.connect()

# ... add concepts, edges, rules ...

# Project rules
projection = Projection(
    source=FlatRuleSource(backend=backend),
    target=BuildlogSeedTarget(persona_name="my_rules"),
)
result = projection.project(domains=["my_domain"])
```

Or use the CLI:

```bash
# Project rules to interop pending directory
qortex project buildlog --domain my_domain --pending

# Check interop status
qortex interop status
```

## Architecture

- **GraphBackend**: Protocol with InMemoryBackend (testing) and MemgraphBackend (production)
- **Projection Pipeline**: Source → Enricher → Target composition
- **Consumer Interop**: Pending directory + signal log for any consumer

## Documentation

Full documentation: https://peleke.github.io/qortex/

## Roadmap

- [x] Phase 1: KG core, projection pipeline, consumer interop
- [ ] Phase 2: HippoRAG-style cross-domain retrieval (PPR-based pattern completion)
- [ ] Phase 3: Causal DAG for confidence feedback loops

See [Issues](https://github.com/Peleke/qortex/issues) for detailed roadmap.

## License

MIT
