# Changelog

All notable changes to qortex are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-02-13

### Added

- **`qortex-observe` standalone package**: Extracted observability layer into `packages/qortex-observe/`, installable independently as `qortex-observe` (#73)
- **Declarative metric schema**: All 36 metrics defined in a single `metrics_schema.py` registry with types, labels, and custom histogram buckets
- **Unified metrics pipeline**: OTel as sole metric backend. One set of event handlers in `metrics_handlers.py` creates OTel instruments from the schema
- **PrometheusMetricReader**: Replaces the old `prometheus_client` subscriber. Serves `/metrics` on port 9464 using OTel's built-in Prometheus exporter
- **Full trace hierarchy**: `@traced` decorator on all MemgraphBackend operations (14 methods), vec layer (embeddings, index ops), and learning (select, observe, credit deltas). Jaeger shows complete parent-child span trees
- **PPR span attributes**: `ppr.node_count`, `ppr.edge_count`, `ppr.iterations`, `ppr.converged`, `ppr.final_diff`, `ppr.latency_ms` on every PageRank execution
- **Embedding model tracing**: `vec.embed.sentence_transformer`, `vec.embed.openai`, `vec.embed.ollama` spans with model name, batch size, and external I/O marking
- **Cached embedding tracing**: `vec.embed.cached` spans with cache hit/miss/batch_size attributes
- **Selective trace sampling**: `SelectiveSpanProcessor` always exports errors and slow spans; normal spans sampled at configurable rate (default 10%)
- **Trace sampling env vars**: `QORTEX_OTEL_TRACE_SAMPLE_RATE` (default 0.1) and `QORTEX_OTEL_TRACE_LATENCY_THRESHOLD_MS` (default 100.0)
- **Live stack validation**: `scripts/validate_live_stack.py` verifies metrics in Prometheus, traces in Jaeger, and dashboards in Grafana (132 checks)

### Removed

- **`prometheus.py` subscriber**: Replaced by unified OTel metrics pipeline with PrometheusMetricReader
- **Direct `prometheus_client` dependency**: All Prometheus exposition now via `opentelemetry-exporter-prometheus`

### Changed

- `emitter.configure()` now wires the unified metrics pipeline (schema -> factory -> handlers) instead of separate OTel + Prometheus subscribers
- OTel subscriber (`otel.py`) slimmed to traces-only; all metric definitions and handlers moved to dedicated modules
- Global `MeterProvider` set via `set_meter_provider()` so `force_flush()` works from external code
- CI workflow installs `observe` extra in all jobs (lint, type check, test)
- Observability guide updated with new architecture diagram, tracing section, and env var documentation

## [0.4.0] - 2026-02-13

### Added

- **`qortex_compare` tool**: Side-by-side comparison of graph-enhanced vs cosine-only retrieval with diff, rank changes, and rule surfacing (#99)
- **`qortex_stats` tool**: Knowledge coverage, learning progress, activity counters, and persistence info (#99)
- **`Learner.batch_observe()`**: Bulk observation wrapper for direct Python consumers (#98)
- **`Learner.top_arms()`**: Top-k arms by posterior mean, sorted descending (#98)
- **`Learner.decay_arm()`**: Shrink learned signal toward prior with configurable decay factor (#98)
- **Activity counters**: Server-level query/feedback tracking for stats tool (#99)

### Changed

- 9 MCP tool docstrings rewritten to remove internal jargon (PPR, teleportation, FlatRuleSource, Phase refs) (#99)
- README overhauled with product-focused structure: install, what happens next, comparison table, framework adapters (#99)
- Module docstring updated to list all 33 tools by category (#99)
- pyproject.toml description updated to "Knowledge graph that learns from every interaction" (#99)
- docs/index.md aligned with README tone (#100)
- docs/installation.md updated from 4 to 12 dependency groups (#100)
- docs/api.md expanded with learning tools, Learner API, compare, stats, type tables (#100)
- docs/quickstart.md added compare and stats examples (#100)

## [0.3.4] - 2026-02-12

### Added

- **Credit Propagation**: CreditAssigner wired into feedback loop; causal DAG propagates rewards to upstream concepts behind `QORTEX_CREDIT_PROPAGATION` flag (#81)
- **Centralized Feature Flags**: `FeatureFlags` dataclass in `flags.py` with YAML config + env var overrides (#81)
- **Learning Event Subscribers**: structlog and JSONL subscribers now handle `LearningSelectionMade`, `LearningObservationRecorded`, `LearningPosteriorUpdated` events (#79)
- **LearningStore Protocol**: `SqliteLearningStore` (ACID, default) and `JsonLearningStore` backends with context partitioning (#75)
- **Learning Module**: Thompson Sampling bandit with 6 MCP tools and full observability (#64)
- **Credit Propagation Dashboard**: Grafana panels, Mermaid diagrams, observability guide (#85)
- **Dashboard Mermaid Explainers**: Every dashboard section now opens with a Mermaid flowchart and signal table (Healthy/Investigate thresholds); observability guide updated with Learning & Bandits and Credit Propagation sections (#95)

### Fixed

- structlog logger no longer nukes external log handlers (#84)
- Observability guide prose cleanup (bragi gauntlet: removed marketing fragments, em-dash substitutes, dismissive framing)

### Changed

- README and docs/index.md feature descriptions rewritten for precision over marketing

## [Unreleased]

### Added

- **CodeExample Entity** for few-shot retrieval and code grounding
  - Associate code snippets with concepts and rules
  - SQLA-compatible structure for direct deserialization
  - Antipattern support for contrastive learning
  - Automatic extraction from source text during ingestion
  - See Issue #19 for full plan

- **Extraction Density Improvements**
  - 4.3x edge density improvement (0.097 -> 0.418 edges/concept)
  - Enhanced relation extraction prompts with disambiguation
  - Density guidance in system prompts
  - Rate limit handling with exponential backoff
  - Concept limiting (100/call) to stay under token limits

- **Edge Pruning Pipeline**
  - 6-step deterministic pruning (evidence, confidence, Jaccard, competing, isolated, layer tagging)
  - `qortex prune manifest` and `qortex prune stats` CLI commands
  - Configurable confidence floor and evidence thresholds

- **Manifest Save/Load**
  - `qortex ingest file --save-manifest` to persist extraction results
  - Auto-save on graph connection failure
  - `qortex ingest load` to reload manifests without re-extraction

- **Consumer Interop Protocol** (Track G)
  - Hybrid pull/push model for seed distribution
  - `write_seed_to_pending()` for publishing seeds
  - `append_signal()` / `read_signals()` for event streaming
  - Path traversal protection and filename sanitization
  - JSON Schema artifacts for any-language validation (`seed.v1.schema.json`, `event.v1.schema.json`)
  - CLI commands: `qortex interop init/status/pending/signals/schema/validate/config`

- **Universal Rule Set Schema** (Track F)
  - Flat `persona` (string, not dict)
  - Integer `version` (not semver string)
  - `rule` key (not `text`) for rule text
  - `provenance` block with full derivation metadata
  - Template metadata threading for derived rules

- **MemgraphBackend** (Track F)
  - Production backend using neo4j driver
  - Cypher query support
  - Personalized PageRank via MAGE
  - Checkpoint/restore functionality

- **Rule Enrichment System** (Track C)
  - `EnrichmentPipeline` with automatic fallback
  - `TemplateEnrichmentFallback` for deterministic enrichment
  - `AnthropicEnrichmentBackend` for LLM enrichment
  - Re-enrichment support with version tracking

- **Edge Rule Templates** (Track A)
  - 30 templates (3 variants x 10 relation types)
  - Template registry with reset for test isolation
  - `select_template()` for edge-based selection

- **Projection Pipeline** (Track B)
  - `FlatRuleSource` for explicit + derived rules
  - `TemplateEnricher` for fast enrichment
  - `BuildlogSeedTarget` for universal schema output
  - `Projection` orchestrator composing Source → Enricher → Target

- **CLI Commands**
  - `qortex infra up/down/status` - Infrastructure management
  - `qortex ingest file <path>` - Content ingestion with LLM extraction
  - `qortex ingest load <manifest>` - Load saved manifest
  - `qortex prune manifest/stats` - Edge pruning and analysis
  - `qortex project buildlog/flat/json` - Rule projection
  - `qortex inspect domains/rules/stats` - Graph inspection
  - `qortex viz open/query` - Visualization and Cypher

- **InMemoryBackend** (Track A)
  - Full GraphBackend protocol implementation
  - Domain isolation
  - Pattern-based node search

### Changed

- **Schema Changes** (breaking for existing consumers)
  - `persona` changed from dict to flat string
  - `version` changed from string to integer
  - Rule text key changed from `text` to `rule`
  - Provenance fields moved into `provenance` block

### Fixed

- Timezone comparison in `read_signals()` now normalizes both timestamps
- Registry reset for test isolation in templates module
- JSON parse error handling in LLM output parsing

## [0.1.0] - 2026-02-01

### Added

- Initial project scaffolding
- Core data models (ConceptNode, ConceptEdge, Rule, Domain)
- RelationType enum with 10 semantic types
- Basic CLI structure with typer

---

## Migration Guide

### From 0.1.0 to 0.2.0

If you have existing consumers of qortex output:

1. **Update persona handling**:
   ```python
   # Old
   persona_name = seed["persona"]["name"]

   # New
   persona_name = seed["persona"]
   ```

2. **Update version comparison**:
   ```python
   # Old
   if seed["version"] == "1.0.0":

   # New
   if seed["version"] == 1:
   ```

3. **Update rule access**:
   ```python
   # Old
   rule_text = rule["text"]
   rule_id = rule["id"]
   rule_domain = rule["domain"]

   # New
   rule_text = rule["rule"]
   rule_id = rule["provenance"]["id"]
   rule_domain = rule["provenance"]["domain"]
   ```

4. **Validate with new schema**:
   ```bash
   qortex interop schema --output ./schemas/
   # Use schemas/seed.v1.schema.json for validation
   ```
