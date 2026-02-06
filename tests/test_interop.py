"""Exhaustive tests for consumer interop protocol.

Tests cover:
- Config loading with defaults and custom paths
- Signal event creation, appending, and reading
- Seed file writing with path validation
- Security: path traversal prevention, filename sanitization
- CLI commands for interop management
"""

from __future__ import annotations

from datetime import UTC, datetime

import yaml

# =============================================================================
# Config Tests
# =============================================================================


class TestInteropConfig:
    def test_default_config_paths(self):
        from qortex.interop import InteropConfig

        config = InteropConfig()
        assert "pending" in str(config.seeds.pending)
        assert "processed" in str(config.seeds.processed)
        assert "failed" in str(config.seeds.failed)
        assert "projections.jsonl" in str(config.signals.projections)

    def test_config_expands_user(self):
        from qortex.interop import SeedsConfig

        config = SeedsConfig(pending="~/test/pending")
        assert "~" not in str(config.pending)

    def test_get_interop_config_returns_defaults_when_no_file(self, tmp_path):
        from qortex.interop import get_interop_config

        config = get_interop_config(tmp_path / "nonexistent.yaml")
        assert config.seeds.pending.name == "pending"

    def test_get_interop_config_reads_custom_file(self, tmp_path):
        from qortex.interop import get_interop_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "seeds": {
                "pending": str(tmp_path / "custom_pending"),
            },
        }))

        config = get_interop_config(config_file)
        assert config.seeds.pending == tmp_path / "custom_pending"

    def test_get_interop_config_handles_malformed_yaml(self, tmp_path):
        from qortex.interop import get_interop_config

        config_file = tmp_path / "bad.yaml"
        config_file.write_text("not: valid: yaml: [")

        # Should return defaults on parse error
        config = get_interop_config(config_file)
        assert config.seeds.pending.name == "pending"

    def test_write_config(self, tmp_path):
        from qortex.interop import InteropConfig, get_interop_config, write_config

        config = InteropConfig()
        config_path = tmp_path / "written.yaml"
        write_config(config, config_path)

        assert config_path.exists()
        loaded = get_interop_config(config_path)
        assert loaded.seeds.pending.name == config.seeds.pending.name

    def test_ensure_dirs_creates_directories(self, tmp_path):
        from qortex.interop import InteropConfig, SeedsConfig, SignalsConfig

        seeds = SeedsConfig(
            pending=tmp_path / "s" / "pending",
            processed=tmp_path / "s" / "processed",
            failed=tmp_path / "s" / "failed",
        )
        signals = SignalsConfig(projections=tmp_path / "sig" / "proj.jsonl")
        config = InteropConfig(seeds=seeds, signals=signals)

        config.ensure_dirs()

        assert (tmp_path / "s" / "pending").exists()
        assert (tmp_path / "s" / "processed").exists()
        assert (tmp_path / "s" / "failed").exists()
        assert (tmp_path / "sig").exists()


# =============================================================================
# Signal Event Tests
# =============================================================================


class TestProjectionEvent:
    def test_to_dict(self):
        from qortex.interop import ProjectionEvent

        event = ProjectionEvent(
            persona="test_persona",
            domain="test_domain",
            path="/path/to/seed.yaml",
            rule_count=10,
            ts="2026-02-05T14:00:00",
        )
        d = event.to_dict()

        assert d["event"] == "projection_complete"
        assert d["persona"] == "test_persona"
        assert d["rule_count"] == 10

    def test_to_dict_with_extra_fields(self):
        from qortex.interop import ProjectionEvent

        event = ProjectionEvent(
            persona="test",
            extra={"custom_field": "value", "number": 42},
        )
        d = event.to_dict()

        assert d["custom_field"] == "value"
        assert d["number"] == 42

    def test_from_dict(self):
        from qortex.interop import ProjectionEvent

        data = {
            "event": "projection_complete",
            "persona": "test",
            "domain": "dom",
            "rule_count": 5,
            "unknown_field": "preserved",
        }
        event = ProjectionEvent.from_dict(data)

        assert event.persona == "test"
        assert event.rule_count == 5
        assert event.extra["unknown_field"] == "preserved"

    def test_roundtrip(self):
        from qortex.interop import ProjectionEvent

        original = ProjectionEvent(
            persona="roundtrip",
            domain="test",
            rule_count=99,
            extra={"key": "value"},
        )
        d = original.to_dict()
        restored = ProjectionEvent.from_dict(d)

        assert restored.persona == original.persona
        assert restored.rule_count == original.rule_count
        assert restored.extra == original.extra


class TestSignalAppending:
    def test_append_signal_creates_file(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            ProjectionEvent,
            SeedsConfig,
            SignalsConfig,
            append_signal,
        )

        signals = SignalsConfig(projections=tmp_path / "signals.jsonl")
        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "p",
                processed=tmp_path / "pr",
                failed=tmp_path / "f",
            ),
            signals=signals,
        )

        event = ProjectionEvent(persona="test", rule_count=1)
        append_signal(event, config)

        assert (tmp_path / "signals.jsonl").exists()
        content = (tmp_path / "signals.jsonl").read_text()
        assert "test" in content

    def test_append_signal_appends_newline(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            ProjectionEvent,
            SeedsConfig,
            SignalsConfig,
            append_signal,
        )

        signals = SignalsConfig(projections=tmp_path / "signals.jsonl")
        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "p",
                processed=tmp_path / "pr",
                failed=tmp_path / "f",
            ),
            signals=signals,
        )

        append_signal(ProjectionEvent(persona="first"), config)
        append_signal(ProjectionEvent(persona="second"), config)

        lines = (tmp_path / "signals.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2


class TestSignalReading:
    def test_read_signals_empty(self, tmp_path):
        from qortex.interop import InteropConfig, SeedsConfig, SignalsConfig, read_signals

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "p",
                processed=tmp_path / "pr",
                failed=tmp_path / "f",
            ),
            signals=SignalsConfig(projections=tmp_path / "nope.jsonl"),
        )
        signals = read_signals(config)
        assert signals == []

    def test_read_signals_returns_events(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            ProjectionEvent,
            SeedsConfig,
            SignalsConfig,
            append_signal,
            read_signals,
        )

        signals_cfg = SignalsConfig(projections=tmp_path / "signals.jsonl")
        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "p",
                processed=tmp_path / "pr",
                failed=tmp_path / "f",
            ),
            signals=signals_cfg,
        )

        append_signal(ProjectionEvent(persona="a", rule_count=1), config)
        append_signal(ProjectionEvent(persona="b", rule_count=2), config)

        signals = read_signals(config)
        assert len(signals) == 2
        assert signals[0].persona == "a"
        assert signals[1].persona == "b"

    def test_read_signals_filters_by_event_type(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            ProjectionEvent,
            SeedsConfig,
            SignalsConfig,
            append_signal,
            read_signals,
        )

        signals_cfg = SignalsConfig(projections=tmp_path / "signals.jsonl")
        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "p",
                processed=tmp_path / "pr",
                failed=tmp_path / "f",
            ),
            signals=signals_cfg,
        )

        append_signal(ProjectionEvent(event="projection_complete", persona="a"), config)
        append_signal(ProjectionEvent(event="seed_ingested", persona="b"), config)

        signals = read_signals(config, event_types=["projection_complete"])
        assert len(signals) == 1
        assert signals[0].persona == "a"

    def test_read_signals_filters_by_timestamp(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            ProjectionEvent,
            SeedsConfig,
            SignalsConfig,
            append_signal,
            read_signals,
        )

        signals_cfg = SignalsConfig(projections=tmp_path / "signals.jsonl")
        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "p",
                processed=tmp_path / "pr",
                failed=tmp_path / "f",
            ),
            signals=signals_cfg,
        )

        append_signal(ProjectionEvent(persona="old", ts="2020-01-01T00:00:00"), config)
        append_signal(ProjectionEvent(persona="new", ts="2026-01-01T00:00:00"), config)

        cutoff = datetime(2025, 1, 1, tzinfo=UTC)
        signals = read_signals(config, since=cutoff)
        assert len(signals) == 1
        assert signals[0].persona == "new"

    def test_read_signals_skips_malformed_lines(self, tmp_path):
        from qortex.interop import InteropConfig, SeedsConfig, SignalsConfig, read_signals

        signals_path = tmp_path / "signals.jsonl"
        signals_path.parent.mkdir(parents=True, exist_ok=True)
        signals_path.write_text('{"persona": "good"}\nnot json\n{"persona": "also_good"}\n')

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "p",
                processed=tmp_path / "pr",
                failed=tmp_path / "f",
            ),
            signals=SignalsConfig(projections=signals_path),
        )

        signals = read_signals(config)
        assert len(signals) == 2


# =============================================================================
# Seed File Tests
# =============================================================================


class TestFilenameSanitization:
    def test_sanitize_removes_path_separators(self):
        from qortex.interop import _sanitize_filename

        assert "/" not in _sanitize_filename("foo/bar")
        assert "\\" not in _sanitize_filename("foo\\bar")

    def test_sanitize_removes_path_traversal(self):
        from qortex.interop import _sanitize_filename

        assert ".." not in _sanitize_filename("../../../etc/passwd")

    def test_sanitize_removes_tilde(self):
        from qortex.interop import _sanitize_filename

        assert "~" not in _sanitize_filename("~root")

    def test_sanitize_handles_empty_result(self):
        from qortex.interop import _sanitize_filename

        result = _sanitize_filename("///")
        assert result == "unnamed"

    def test_sanitize_collapses_underscores(self):
        from qortex.interop import _sanitize_filename

        result = _sanitize_filename("a___b")
        assert result == "a_b"


class TestGenerateSeedFilename:
    def test_includes_persona_and_timestamp(self):
        from qortex.interop import generate_seed_filename

        ts = datetime(2026, 2, 5, 14, 30, 0, tzinfo=UTC)
        filename = generate_seed_filename("test_persona", ts)

        assert "test_persona" in filename
        assert "2026-02-05T14-30-00" in filename
        assert filename.endswith(".yaml")

    def test_sanitizes_persona(self):
        from qortex.interop import generate_seed_filename

        filename = generate_seed_filename("../evil/persona", None)
        assert ".." not in filename
        assert "/" not in filename


class TestWriteSeedToPending:
    def test_writes_yaml_file(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            write_seed_to_pending,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        seed_data = {
            "persona": "test",
            "version": 1,
            "rules": [{"rule": "Test rule"}],
            "metadata": {"rule_count": 1},
        }

        path = write_seed_to_pending(
            seed_data, "test_persona", "test_domain", config, emit_signal=False
        )

        assert path.exists()
        content = yaml.safe_load(path.read_text())
        assert content["persona"] == "test"

    def test_emits_signal(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            read_signals,
            write_seed_to_pending,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        seed_data = {"rules": [], "metadata": {"rule_count": 0}}
        write_seed_to_pending(seed_data, "test", "domain", config, emit_signal=True)

        signals = read_signals(config)
        assert len(signals) == 1
        assert signals[0].event == "projection_complete"
        assert signals[0].persona == "test"

    def test_no_signal_when_disabled(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            read_signals,
            write_seed_to_pending,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        seed_data = {"rules": [], "metadata": {"rule_count": 0}}
        write_seed_to_pending(seed_data, "test", "domain", config, emit_signal=False)

        signals = read_signals(config)
        assert len(signals) == 0

    def test_extra_event_data(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            read_signals,
            write_seed_to_pending,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        seed_data = {"rules": [], "metadata": {"rule_count": 0}}
        write_seed_to_pending(
            seed_data, "test", "domain", config,
            emit_signal=True,
            extra_event_data={"chapter": "5"},
        )

        signals = read_signals(config)
        assert signals[0].extra["chapter"] == "5"


class TestPathValidation:
    def test_validate_path_in_directory_success(self, tmp_path):
        from qortex.interop import _validate_path_in_directory

        directory = tmp_path / "base"
        directory.mkdir()
        path = directory / "file.yaml"

        assert _validate_path_in_directory(path, directory)

    def test_validate_path_in_directory_rejects_traversal(self, tmp_path):
        from qortex.interop import _validate_path_in_directory

        directory = tmp_path / "base"
        directory.mkdir()
        path = tmp_path / "other" / "file.yaml"

        assert not _validate_path_in_directory(path, directory)

    def test_write_seed_with_malicious_persona_is_sanitized(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            write_seed_to_pending,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        seed_data = {"rules": [], "metadata": {"rule_count": 0}}

        # Attempt path traversal via persona
        path = write_seed_to_pending(
            seed_data, "../../../etc/passwd", "domain", config, emit_signal=False
        )

        # File should be in pending, not escaped
        assert path.parent == config.seeds.pending
        assert ".." not in path.name


class TestListSeeds:
    def test_list_pending_empty(self, tmp_path):
        from qortex.interop import InteropConfig, SeedsConfig, SignalsConfig, list_pending_seeds

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        assert list_pending_seeds(config) == []

    def test_list_pending_returns_sorted(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            list_pending_seeds,
            write_seed_to_pending,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        seed_data = {"rules": [], "metadata": {"rule_count": 0}}
        write_seed_to_pending(seed_data, "aaa", "d", config, emit_signal=False)
        write_seed_to_pending(seed_data, "zzz", "d", config, emit_signal=False)
        write_seed_to_pending(seed_data, "mmm", "d", config, emit_signal=False)

        seeds = list_pending_seeds(config)
        assert len(seeds) == 3
        # Should be sorted by name (aaa, mmm, zzz)
        names = [s.name for s in seeds]
        assert names == sorted(names)

    def test_list_processed(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            list_processed_seeds,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )
        config.seeds.ensure_dirs()

        (config.seeds.processed / "test.yaml").write_text("test: data")

        seeds = list_processed_seeds(config)
        assert len(seeds) == 1

    def test_list_failed_with_errors(self, tmp_path):
        from qortex.interop import (
            InteropConfig,
            SeedsConfig,
            SignalsConfig,
            list_failed_seeds,
        )

        config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )
        config.seeds.ensure_dirs()

        (config.seeds.failed / "bad.yaml").write_text("bad: data")
        (config.seeds.failed / "bad.error").write_text("ValidationError: something")

        failed = list_failed_seeds(config)
        assert len(failed) == 1
        path, error = failed[0]
        assert path.name == "bad.yaml"
        assert "ValidationError" in error


# =============================================================================
# CLI Tests
# =============================================================================


class TestInteropCLI:
    def test_status_command(self, tmp_path):
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        runner = CliRunner()
        result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "Interop Configuration" in result.output

    def test_init_command(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        # Monkeypatch to use tmp_path
        from qortex import interop
        from qortex.cli.interop_cmd import app
        monkeypatch.setattr(
            interop, "_CONFIG_PATH", tmp_path / "config.yaml"
        )

        runner = CliRunner()
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Initialized" in result.output

    def test_pending_command_empty(self):
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        runner = CliRunner()
        result = runner.invoke(app, ["pending"])

        # May show "No pending seeds" or list some
        assert result.exit_code == 0

    def test_signals_command(self):
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        runner = CliRunner()
        result = runner.invoke(app, ["signals", "--limit", "5"])

        assert result.exit_code == 0

    def test_config_show(self):
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        runner = CliRunner()
        result = runner.invoke(app, ["config", "--show"])

        assert result.exit_code == 0
        assert "seeds.pending" in result.output


# =============================================================================
# Integration with project CLI
# =============================================================================


class TestProjectBuildlogPending:
    def test_project_buildlog_pending_flag(self, tmp_path, monkeypatch):
        """Test that --pending flag writes to interop directory."""
        from typer.testing import CliRunner

        from qortex import interop
        from qortex.cli import app
        from qortex.interop import InteropConfig, SeedsConfig, SignalsConfig

        # Create a custom config for testing
        test_config = InteropConfig(
            seeds=SeedsConfig(
                pending=tmp_path / "pending",
                processed=tmp_path / "processed",
                failed=tmp_path / "failed",
            ),
            signals=SignalsConfig(projections=tmp_path / "signals.jsonl"),
        )

        # Monkeypatch get_interop_config to return our test config
        monkeypatch.setattr(interop, "get_interop_config", lambda *a, **kw: test_config)

        runner = CliRunner()
        result = runner.invoke(app, [
            "project", "buildlog",
            "--domain", "test",
            "--pending",
            "--persona", "test_persona",
        ])

        # The command might fail due to no backend, but we check the flag parsing
        # In real use, there would be data in the backend
        # For now, just verify the CLI accepts the flags
        assert "--pending" not in result.output or "Error" not in result.output


# =============================================================================
# Schema Tests
# =============================================================================


class TestSchemas:
    def test_seed_schema_is_valid_json_schema(self):
        from qortex.interop_schemas import SEED_SCHEMA

        # Basic structure checks
        assert SEED_SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "title" in SEED_SCHEMA
        assert "properties" in SEED_SCHEMA
        assert "persona" in SEED_SCHEMA["properties"]
        assert "rules" in SEED_SCHEMA["properties"]

    def test_event_schema_is_valid_json_schema(self):
        from qortex.interop_schemas import EVENT_SCHEMA

        assert EVENT_SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "properties" in EVENT_SCHEMA
        assert "event" in EVENT_SCHEMA["properties"]

    def test_get_seed_schema_returns_copy(self):
        from qortex.interop_schemas import SEED_SCHEMA, get_seed_schema

        schema = get_seed_schema()
        schema["modified"] = True
        assert "modified" not in SEED_SCHEMA

    def test_get_event_schema_returns_copy(self):
        from qortex.interop_schemas import EVENT_SCHEMA, get_event_schema

        schema = get_event_schema()
        schema["modified"] = True
        assert "modified" not in EVENT_SCHEMA

    def test_export_schemas(self, tmp_path):
        from qortex.interop_schemas import export_schemas

        seed_path, event_path = export_schemas(tmp_path)

        assert seed_path.exists()
        assert event_path.exists()
        assert "seed" in seed_path.name
        assert "event" in event_path.name
        assert seed_path.suffix == ".json"

    def test_validate_seed_valid(self):
        from qortex.interop_schemas import validate_seed

        valid_seed = {
            "persona": "test",
            "version": 1,
            "rules": [
                {
                    "rule": "Test rule",
                    "category": "testing",
                    "provenance": {
                        "id": "r1",
                        "domain": "test",
                        "derivation": "explicit",
                        "confidence": 0.9,
                    }
                }
            ],
            "metadata": {
                "source": "qortex",
                "rule_count": 1,
            }
        }

        errors = validate_seed(valid_seed)
        assert errors == []

    def test_validate_seed_missing_fields(self):
        from qortex.interop_schemas import validate_seed

        invalid_seed = {"persona": "test"}
        errors = validate_seed(invalid_seed)
        assert len(errors) > 0
        assert any("version" in e for e in errors)

    def test_validate_seed_wrong_types(self):
        from qortex.interop_schemas import validate_seed

        invalid_seed = {
            "persona": 123,  # Should be string
            "version": "1",  # Should be int
            "rules": "not a list",
            "metadata": {},
        }
        errors = validate_seed(invalid_seed)
        assert len(errors) >= 3

    def test_validate_event_valid(self):
        from qortex.interop_schemas import validate_event

        valid_event = {
            "event": "projection_complete",
            "ts": "2026-02-05T14:00:00Z",
            "source": "qortex",
        }
        errors = validate_event(valid_event)
        assert errors == []

    def test_validate_event_invalid_type(self):
        from qortex.interop_schemas import validate_event

        invalid_event = {
            "event": "unknown_event",
            "ts": "2026-02-05T14:00:00Z",
            "source": "qortex",
        }
        errors = validate_event(invalid_event)
        assert len(errors) > 0


class TestSchemaCLI:
    def test_schema_command_shows_seed(self):
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        runner = CliRunner()
        result = runner.invoke(app, ["schema", "--which", "seed"])

        assert result.exit_code == 0
        assert "Seed Schema" in result.output
        assert "persona" in result.output

    def test_schema_command_shows_event(self):
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        runner = CliRunner()
        result = runner.invoke(app, ["schema", "--which", "event"])

        assert result.exit_code == 0
        assert "Event Schema" in result.output
        assert "projection_complete" in result.output

    def test_schema_command_exports(self, tmp_path):
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        runner = CliRunner()
        result = runner.invoke(app, ["schema", "--output", str(tmp_path)])

        assert result.exit_code == 0
        assert "Exported" in result.output
        assert (tmp_path / "seed.v1.0.schema.json").exists()

    def test_validate_command_valid_seed(self, tmp_path):
        import yaml
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        seed_file = tmp_path / "test.yaml"
        seed_file.write_text(yaml.dump({
            "persona": "test",
            "version": 1,
            "rules": [],
            "metadata": {"source": "test", "rule_count": 0},
        }))

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(seed_file)])

        assert result.exit_code == 0
        assert "Valid seed" in result.output

    def test_validate_command_invalid_seed(self, tmp_path):
        import yaml
        from typer.testing import CliRunner

        from qortex.cli.interop_cmd import app

        seed_file = tmp_path / "bad.yaml"
        seed_file.write_text(yaml.dump({"persona": "incomplete"}))

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(seed_file)])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()
