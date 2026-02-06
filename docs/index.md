# qortex

**Knowledge graph ingestion engine for automated rule generation.**

qortex transforms unstructured content (books, docs, PDFs) into a knowledge graph, then projects actionable rules for AI agents, buildlog, and other consumers.

![qortex pipeline](images/diagrams/pipeline.svg)

## Features

- **Flexible Ingestion**: PDF, Markdown, and text sources into a unified knowledge graph
- **Rich Type System**: 10 semantic relation types with 30 edge rule templates
- **Projection Pipeline**: Source → Enricher → Target architecture for rule generation
- **Universal Schema**: JSON Schema artifacts for any-language validation
- **Consumer Interop**: Shared directory protocol for seamless integration with buildlog and other tools
- **Multiple Backends**: InMemory (testing), Memgraph (production with MAGE algorithms)

## Quick Example

```python
from qortex.core.memory import InMemoryBackend
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget
from qortex.projectors.projection import Projection

# Connect to backend
backend = InMemoryBackend()
backend.connect()

# Project rules to buildlog format
projection = Projection(
    source=FlatRuleSource(backend=backend),
    target=BuildlogSeedTarget(persona_name="my_rules"),
)
result = projection.project(domains=["my_domain"])
```

Or use the CLI:

```bash
# Project rules to the interop pending directory
qortex project buildlog --domain my_domain --pending

# Check interop status
qortex interop status
```

## Installation

```bash
pip install qortex

# With Memgraph support
pip install qortex[memgraph]

# With all optional dependencies
pip install qortex[all]
```

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } **Getting Started**

    ---

    Install qortex and project your first rules in under 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-graph:{ .lg .middle } **Core Concepts**

    ---

    Understand domains, concepts, edges, and rules

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

-   :material-pipe:{ .lg .middle } **Projection Pipeline**

    ---

    Learn the Source → Enricher → Target architecture

    [:octicons-arrow-right-24: Projecting Rules](guides/projecting-rules.md)

-   :material-connection:{ .lg .middle } **Consumer Integration**

    ---

    Connect qortex to buildlog, MCP servers, and other consumers

    [:octicons-arrow-right-24: Integration Guide](guides/consumer-integration.md)

</div>

## License

MIT License. See [LICENSE](https://github.com/Peleke/qortex/blob/main/LICENSE) for details.
