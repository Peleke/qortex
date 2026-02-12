# Changelog

All notable changes to qortex are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.4] - 2026-02-12

### Added

- **Credit Propagation**: CreditAssigner wired into feedback loop; causal DAG propagates rewards to upstream concepts behind `QORTEX_CREDIT_PROPAGATION` flag (#81)
- **Centralized Feature Flags**: `FeatureFlags` dataclass in `flags.py` with YAML config + env var overrides (#81)
- **Learning Event Subscribers**: structlog and JSONL subscribers now handle `LearningSelectionMade`, `LearningObservationRecorded`, `LearningPosteriorUpdated` events (#79)
- **LearningStore Protocol**: `SqliteLearningStore` (ACID, default) and `JsonLearningStore` backends with context partitioning (#75)
- **Learning Module**: Thompson Sampling bandit with 6 MCP tools and full observability (#64)
- **Credit Propagation Dashboard**: Grafana panels, Mermaid diagrams, observability guide (#85)

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
