"""Tests for buildlog emission artifact ingestion."""

import json
import sqlite3
from pathlib import Path

from qortex.core.models import RelationType
from qortex.ingest_emissions import (
    _classify_artifact,
    _extract_concept,
    _extract_edges,
    _extract_learned_rules,
    _map_relation_type,
    aggregate_emissions,
    bridge_gauntlet_rules,
    build_manifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_emission(dir_path: Path, filename: str, data: dict) -> Path:
    """Write an emission artifact JSON file."""
    p = dir_path / filename
    p.write_text(json.dumps(data))
    return p


def _make_mistake_manifest(mistake_id: str = "mistake-test-001", edges: list | None = None) -> dict:
    return {
        "source_id": "buildlog:test123",
        "domain": "experiential",
        "concepts": [
            {
                "name": f"mistake:{mistake_id}",
                "domain": "experiential",
                "properties": {
                    "error_class": "logic_error",
                    "description": "Off-by-one in loop",
                    "timestamp": "2026-02-14T10:00:00+00:00",
                    "was_repeat": False,
                    "session_id": "session-test-001",
                },
                "source_id": "buildlog:test123",
            }
        ],
        "edges": edges or [],
        "rules": [],
        "metadata": {
            "source": "buildlog",
            "source_version": "0.17.0",
            "emitted_at": "2026-02-14T10:00:00+00:00",
            "project_id": "test123",
            "mistake_id": mistake_id,
        },
    }


def _make_session_summary(session_id: str = "session-test-001") -> dict:
    return {
        "source_id": "buildlog:test123",
        "domain": "experiential",
        "concepts": [
            {
                "name": f"session:{session_id}",
                "domain": "experiential",
                "properties": {
                    "started_at": "2026-02-14T10:00:00+00:00",
                    "duration_minutes": 45.0,
                    "mistakes_logged": 3,
                    "repeated_mistakes": 1,
                    "outcome": "accepted",
                    "rules_count": 5,
                    "ended_at": "2026-02-14T10:45:00+00:00",
                },
                "source_id": "buildlog:test123",
            }
        ],
        "edges": [
            {
                "source_id": f"session:{session_id}",
                "target_id": "rule-a",
                "relation_type": "uses",
                "properties": {"type": "rule_in_session"},
                "confidence": 1.0,
            },
            {
                "source_id": f"session:{session_id}",
                "target_id": "mistake:mistake-test-001",
                "relation_type": "contains",
                "properties": {"type": "mistake_in_session"},
                "confidence": 1.0,
            },
        ],
        "rules": [],
        "metadata": {
            "source": "buildlog",
            "emitted_at": "2026-02-14T10:45:00+00:00",
            "project_id": "test123",
            "session_id": session_id,
        },
    }


def _make_reward_signal(reward_id: str = "rew-test-001") -> dict:
    return {
        "source_id": "buildlog:test123",
        "domain": "experiential",
        "concepts": [
            {
                "name": f"reward:{reward_id}",
                "domain": "experiential",
                "properties": {
                    "outcome": "accepted",
                    "reward_value": 1.0,
                    "timestamp": "2026-02-14T11:00:00+00:00",
                },
                "source_id": "buildlog:test123",
            }
        ],
        "edges": [],
        "rules": [],
        "metadata": {
            "source": "buildlog",
            "emitted_at": "2026-02-14T11:00:00+00:00",
            "project_id": "test123",
            "reward_id": reward_id,
        },
    }


def _make_learned_rules() -> dict:
    return {
        "persona": "buildlog_test",
        "version": 1,
        "rules": [
            {
                "rule": "Always define interfaces before implementations",
                "category": "architectural",
                "provenance": {
                    "id": "bl:arch-001",
                    "domain": "experiential",
                    "derivation": "explicit",
                    "confidence": 0.8,
                },
            },
            {
                "rule": "Run tests before committing",
                "category": "process",
                "provenance": {
                    "id": "bl:proc-001",
                    "domain": "experiential",
                    "derivation": "explicit",
                    "confidence": 0.9,
                },
            },
        ],
        "metadata": {
            "source": "buildlog",
            "source_version": "0.14.1",
            "projected_at": "2026-02-14T12:00:00+00:00",
            "rule_count": 2,
        },
    }


# ---------------------------------------------------------------------------
# Unit tests: classification
# ---------------------------------------------------------------------------


class TestClassifyArtifact:
    def test_mistake_manifest(self):
        assert _classify_artifact("mistake_manifest_abc_20260214.json") == "mistake_manifest"

    def test_reward_signal(self):
        assert _classify_artifact("reward_signal_abc_20260214.json") == "reward_signal"

    def test_session_summary(self):
        assert _classify_artifact("session_summary_abc_20260214.json") == "session_summary"

    def test_learned_rules(self):
        assert _classify_artifact("learned_rules_abc_20260214.json") == "learned_rules"

    def test_unknown(self):
        assert _classify_artifact("something_else.json") == "unknown"


# ---------------------------------------------------------------------------
# Unit tests: relation mapping
# ---------------------------------------------------------------------------


class TestMapRelationType:
    def test_valid_types(self):
        for rt in RelationType:
            assert _map_relation_type(rt.value) == rt

    def test_invalid_type(self):
        assert _map_relation_type("invented_relation") is None

    def test_empty_string(self):
        assert _map_relation_type("") is None


# ---------------------------------------------------------------------------
# Unit tests: concept extraction
# ---------------------------------------------------------------------------


class TestExtractConcept:
    def test_mistake_concept(self):
        raw = {
            "name": "mistake:m1",
            "domain": "experiential",
            "properties": {"error_class": "typo", "description": "Variable name typo"},
            "source_id": "buildlog:abc",
        }
        concept = _extract_concept(raw, "mistake_manifest")
        assert concept is not None
        assert concept.id == "mistake:m1"
        assert concept.name == "mistake:m1"
        assert "typo" in concept.description.lower()
        assert concept.domain == "experiential"

    def test_session_concept(self):
        raw = {
            "name": "session:s1",
            "domain": "experiential",
            "properties": {"duration_minutes": 30.0, "mistakes_logged": 2, "outcome": "accepted"},
            "source_id": "buildlog:abc",
        }
        concept = _extract_concept(raw, "session_summary")
        assert concept is not None
        assert "30min" in concept.description
        assert "accepted" in concept.description

    def test_reward_concept(self):
        raw = {
            "name": "reward:r1",
            "domain": "experiential",
            "properties": {"outcome": "accepted", "reward_value": 1.0},
            "source_id": "buildlog:abc",
        }
        concept = _extract_concept(raw, "reward_signal")
        assert concept is not None
        assert "accepted" in concept.description

    def test_empty_name_returns_none(self):
        assert _extract_concept({"name": ""}, "mistake_manifest") is None
        assert _extract_concept({}, "mistake_manifest") is None


# ---------------------------------------------------------------------------
# Unit tests: edge extraction
# ---------------------------------------------------------------------------


class TestExtractEdges:
    def test_valid_edges(self):
        raw = [
            {"source_id": "a", "target_id": "b", "relation_type": "uses", "confidence": 0.9, "properties": {}},
            {"source_id": "c", "target_id": "d", "relation_type": "supports", "confidence": 1.0, "properties": {"note": "x"}},
        ]
        edges = _extract_edges(raw, "session_summary")
        assert len(edges) == 2
        assert edges[0] == ("a", "b", "uses", 0.9, {})
        assert edges[1][4] == {"note": "x"}

    def test_skips_incomplete(self):
        raw = [
            {"source_id": "a", "target_id": "", "relation_type": "uses"},
            {"source_id": "", "target_id": "b", "relation_type": "uses"},
            {"source_id": "a", "target_id": "b", "relation_type": ""},
        ]
        assert _extract_edges(raw, "test") == []


# ---------------------------------------------------------------------------
# Unit tests: learned rules extraction
# ---------------------------------------------------------------------------


class TestExtractLearnedRules:
    def test_extracts_rules(self):
        data = _make_learned_rules()
        rules = _extract_learned_rules(data)
        assert len(rules) == 2
        assert rules[0].text == "Always define interfaces before implementations"
        assert rules[0].category == "architectural"
        assert rules[0].confidence == 0.8
        assert rules[1].text == "Run tests before committing"

    def test_skips_empty_text(self):
        data = {"rules": [{"rule": "", "provenance": {"id": "x", "domain": "y"}}]}
        assert _extract_learned_rules(data) == []


# ---------------------------------------------------------------------------
# Integration: aggregation
# ---------------------------------------------------------------------------


class TestAggregateEmissions:
    def test_aggregate_all_types(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()

        _write_emission(processed, "mistake_manifest_abc_001.json", _make_mistake_manifest())
        _write_emission(processed, "session_summary_abc_001.json", _make_session_summary())
        _write_emission(processed, "reward_signal_abc_001.json", _make_reward_signal())
        _write_emission(processed, "learned_rules_abc_001.json", _make_learned_rules())

        result = aggregate_emissions(emissions_dir=tmp_path)
        assert result.files_processed == 4
        assert result.files_failed == 0
        assert result.by_type["mistake_manifest"] == 1
        assert result.by_type["session_summary"] == 1
        assert result.by_type["reward_signal"] == 1
        assert result.by_type["learned_rules"] == 1

        # Concepts: mistake + session + reward + stub nodes for edge targets
        assert len(result.concepts) >= 3
        assert "mistake:mistake-test-001" in result.concepts
        assert "session:session-test-001" in result.concepts
        assert "reward:rew-test-001" in result.concepts

        # Edges: 2 from session summary (uses + contains)
        assert len(result.edges) == 2

        # Rules: 2 from learned_rules
        assert len(result.rules) == 2

    def test_dedup_concepts(self, tmp_path):
        """Same concept in multiple artifacts is only stored once."""
        processed = tmp_path / "processed"
        processed.mkdir()

        # Two manifests with same mistake concept
        _write_emission(processed, "mistake_manifest_a_001.json", _make_mistake_manifest("m1"))
        _write_emission(processed, "mistake_manifest_b_001.json", _make_mistake_manifest("m1"))

        result = aggregate_emissions(emissions_dir=tmp_path)
        assert result.files_processed == 2
        # Same concept id → only 1
        mistake_concepts = [c for c in result.concepts.values() if c.id == "mistake:m1"]
        assert len(mistake_concepts) == 1

    def test_dedup_edges(self, tmp_path):
        """Same edge (source, target, relation) is only stored once."""
        processed = tmp_path / "processed"
        processed.mkdir()

        # Two session summaries with same edges
        _write_emission(processed, "session_summary_a_001.json", _make_session_summary("s1"))
        _write_emission(processed, "session_summary_b_001.json", _make_session_summary("s1"))

        result = aggregate_emissions(emissions_dir=tmp_path)
        assert len(result.edges) == 2  # Not 4 — deduped

    def test_empty_dir(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()

        result = aggregate_emissions(emissions_dir=tmp_path)
        assert result.files_processed == 0
        assert len(result.concepts) == 0

    def test_bad_json_skipped(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        (processed / "mistake_manifest_bad_001.json").write_text("not json{{{")
        _write_emission(processed, "reward_signal_good_001.json", _make_reward_signal())

        result = aggregate_emissions(emissions_dir=tmp_path)
        assert result.files_processed == 1
        assert result.files_failed == 1

    def test_include_pending(self, tmp_path):
        pending = tmp_path / "pending"
        pending.mkdir()
        _write_emission(pending, "reward_signal_abc_001.json", _make_reward_signal())

        # Default: no pending
        result = aggregate_emissions(emissions_dir=tmp_path, include_processed=False, include_pending=False)
        assert result.files_processed == 0

        # With pending
        result = aggregate_emissions(emissions_dir=tmp_path, include_processed=False, include_pending=True)
        assert result.files_processed == 1

    def test_stub_nodes_created_for_edge_targets(self, tmp_path):
        """Edge targets that don't have their own concept get stub nodes."""
        processed = tmp_path / "processed"
        processed.mkdir()
        _write_emission(processed, "session_summary_abc_001.json", _make_session_summary())

        result = aggregate_emissions(emissions_dir=tmp_path)
        # "rule-a" is an edge target with no concept — should get a stub
        assert "rule-a" in result.concepts
        assert "Reference node" in result.concepts["rule-a"].description

    def test_edges_with_invalid_relation_skipped(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        manifest = _make_mistake_manifest(edges=[
            {"source_id": "a", "target_id": "b", "relation_type": "invented_rel", "confidence": 1.0},
        ])
        _write_emission(processed, "mistake_manifest_abc_001.json", manifest)

        result = aggregate_emissions(emissions_dir=tmp_path)
        assert len(result.edges) == 0  # Invalid relation type skipped


# ---------------------------------------------------------------------------
# Integration: manifest building
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_builds_valid_manifest(self, tmp_path):
        processed = tmp_path / "processed"
        processed.mkdir()
        _write_emission(processed, "session_summary_abc_001.json", _make_session_summary())
        _write_emission(processed, "learned_rules_abc_001.json", _make_learned_rules())

        result = aggregate_emissions(emissions_dir=tmp_path)
        manifest = build_manifest(result, domain="test_domain")

        assert manifest.domain == "test_domain"
        assert manifest.source.source_type == "text"
        assert manifest.source.name == "buildlog_emissions"
        assert len(manifest.concepts) > 0
        assert len(manifest.edges) == 2
        assert len(manifest.rules) == 2


# ---------------------------------------------------------------------------
# Bridge: gauntlet rules → cross-domain concepts
# ---------------------------------------------------------------------------


def _create_test_db(db_path: Path) -> None:
    """Create a minimal gauntlet_rules table for testing."""
    db = sqlite3.connect(str(db_path))
    db.execute("""
        CREATE TABLE gauntlet_rules (
            rule_id TEXT PRIMARY KEY,
            persona TEXT,
            rule TEXT,
            category TEXT,
            context TEXT,
            antipattern TEXT,
            rationale TEXT,
            tags TEXT,
            refs TEXT,
            provenance TEXT,
            version INTEGER DEFAULT 1,
            active INTEGER DEFAULT 1,
            created_at TEXT,
            updated_at TEXT,
            seed_file_hash TEXT,
            seed_filename TEXT
        )
    """)

    # Rule WITH design pattern domain (bridge candidate)
    db.execute(
        "INSERT INTO gauntlet_rules (rule_id, rule, category, provenance, active, seed_filename) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "qortex:obs-001",
            "Decouple publishers from subscribers",
            "architectural",
            json.dumps({"domain": "observer_pattern", "derivation": "explicit", "confidence": 0.9}),
            1,
            "qortex_observer.yaml",
        ),
    )

    # Rule WITHOUT design pattern domain (persona-only)
    db.execute(
        "INSERT INTO gauntlet_rules (rule_id, rule, category, provenance, active, seed_filename) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "test_terrorist:tt-001",
            "Tests must not depend on execution order",
            "isolation",
            json.dumps({}),
            1,
            "test_terrorist.yaml",
        ),
    )

    # Inactive rule with domain (should be SKIPPED)
    db.execute(
        "INSERT INTO gauntlet_rules (rule_id, rule, category, provenance, active, seed_filename) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "qortex:impl-001",
            "Hide implementation details behind interfaces",
            "encapsulation",
            json.dumps({"domain": "implementation_hiding", "confidence": 0.8}),
            0,
            "qortex_impl_hiding.yaml",
        ),
    )

    # Derived junk rule (active but derivation="derived" — should be SKIPPED)
    db.execute(
        "INSERT INTO gauntlet_rules (rule_id, rule, category, provenance, active, seed_filename) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "qortex:junk-001",
            "BaseballReporter Publisher depends on Method Name Dependency",
            "architectural",
            json.dumps({"domain": "observer_pattern", "derivation": "derived", "confidence": 0.91}),
            1,
            "qortex_observer.yaml",
        ),
    )

    db.commit()
    db.close()


class TestBridgeGauntletRules:
    def test_creates_cross_domain_concepts(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        # Should have active explicit rules, NOT inactive or derived
        concept_ids = {c.id for c in concepts}
        assert "gauntlet_rule:qortex:obs-001" in concept_ids
        assert "gauntlet_rule:test_terrorist:tt-001" in concept_ids
        # Inactive rule — SKIPPED
        assert "gauntlet_rule:qortex:impl-001" not in concept_ids
        # Derived junk rule — SKIPPED
        assert "gauntlet_rule:qortex:junk-001" not in concept_ids

    def test_bridge_rules_get_source_domain(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        obs_concept = next(c for c in concepts if c.id == "gauntlet_rule:qortex:obs-001")
        assert obs_concept.domain == "observer_pattern"  # Cross-domain!

    def test_persona_rules_stay_in_buildlog_domain(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        tt_concept = next(c for c in concepts if c.id == "gauntlet_rule:test_terrorist:tt-001")
        assert tt_concept.domain == "buildlog"

    def test_skips_inactive_rules(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        concept_ids = {c.id for c in concepts}
        assert "gauntlet_rule:qortex:impl-001" not in concept_ids

    def test_skips_derived_junk_rules(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        concept_ids = {c.id for c in concepts}
        assert "gauntlet_rule:qortex:junk-001" not in concept_ids

    def test_creates_domain_anchor_nodes(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        concept_ids = {c.id for c in concepts}
        assert "domain:observer_pattern" in concept_ids
        # implementation_hiding should NOT have anchor — its rule was inactive
        assert "domain:implementation_hiding" not in concept_ids

    def test_creates_instance_of_edges(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        instance_edges = [e for e in edges if e.relation_type == RelationType.INSTANCE_OF]
        assert len(instance_edges) == 1  # only obs-001 (impl-001 inactive, junk-001 derived)

    def test_creates_belongs_to_edges_for_personas(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path)

        concepts, edges = bridge_gauntlet_rules(db_path=db_path)

        belongs_edges = [e for e in edges if e.relation_type == RelationType.BELONGS_TO]
        assert len(belongs_edges) >= 1
        # test_terrorist rule belongs_to test_terrorist persona
        tt_edge = next(e for e in belongs_edges if "test_terrorist:tt-001" in e.source_id)
        assert "persona:test_terrorist" in tt_edge.target_id

    def test_missing_db_returns_empty(self, tmp_path):
        concepts, edges = bridge_gauntlet_rules(db_path=tmp_path / "nonexistent.db")
        assert concepts == []
        assert edges == []
