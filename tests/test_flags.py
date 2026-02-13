"""Tests for centralized feature flags system."""

from __future__ import annotations

import pytest
import yaml

from qortex.flags import FeatureFlags, get_flags, reset_flags


@pytest.fixture(autouse=True)
def _reset():
    """Reset singleton between tests."""
    reset_flags()
    yield
    reset_flags()


class TestFeatureFlagsDefaults:
    """Default values match expectations."""

    def test_defaults(self):
        flags = FeatureFlags()
        assert flags.graph is True
        assert flags.teleportation is False
        assert flags.online_edges is True
        assert flags.enrichment is True
        assert flags.learning is True
        assert flags.causal is True
        assert flags.credit_propagation is False

    def test_to_dict(self):
        flags = FeatureFlags()
        d = flags.to_dict()
        assert isinstance(d, dict)
        assert d["teleportation"] is False
        assert d["credit_propagation"] is False
        assert d["graph"] is True

    def test_is_enabled(self):
        flags = FeatureFlags(teleportation=True)
        assert flags.is_enabled("teleportation") is True
        assert flags.is_enabled("credit_propagation") is False
        assert flags.is_enabled("nonexistent") is False


class TestYAMLLoading:
    """Load flags from YAML config file."""

    def test_load_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": True, "credit_propagation": True}))

        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is True
        assert flags.credit_propagation is True

    def test_load_partial_yaml(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": True}))

        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is True
        assert flags.credit_propagation is False  # default

    def test_load_missing_yaml_uses_defaults(self, tmp_path):
        flags = FeatureFlags.load(tmp_path / "nonexistent.yaml")
        assert flags.teleportation is False
        assert flags.credit_propagation is False

    def test_load_empty_yaml(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text("")

        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is False

    def test_load_non_dict_yaml(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text("just a string")

        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is False

    def test_string_values_in_yaml(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": "on", "credit_propagation": "yes"}))

        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is True
        assert flags.credit_propagation is True

    def test_unknown_keys_ignored(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": True, "unknown_flag": True}))

        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is True

    def test_disable_graph_layer(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({
            "graph": False,
            "enrichment": False,
            "learning": False,
            "causal": False,
            "online_edges": False,
        }))

        flags = FeatureFlags.load(yaml_file)
        assert flags.graph is False
        assert flags.enrichment is False
        assert flags.learning is False
        assert flags.causal is False
        assert flags.online_edges is False


class TestEnvVarOverrides:
    """Env vars override YAML and defaults."""

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("QORTEX_TELEPORTATION", "on")
        flags = FeatureFlags.load()
        assert flags.teleportation is True

    def test_env_overrides_yaml(self, monkeypatch, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": False}))

        monkeypatch.setenv("QORTEX_TELEPORTATION", "1")
        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is True

    def test_env_false_overrides_yaml_true(self, monkeypatch, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": True}))

        monkeypatch.setenv("QORTEX_TELEPORTATION", "off")
        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is False

    def test_env_truthy_values(self, monkeypatch):
        for val in ("1", "true", "on", "yes", "True", "ON", "YES"):
            monkeypatch.setenv("QORTEX_TELEPORTATION", val)
            flags = FeatureFlags.load()
            assert flags.teleportation is True, f"Expected True for {val!r}"

    def test_env_falsy_values(self, monkeypatch):
        for val in ("0", "false", "off", "no", "anything"):
            monkeypatch.setenv("QORTEX_TELEPORTATION", val)
            flags = FeatureFlags.load()
            assert flags.teleportation is False, f"Expected False for {val!r}"

    def test_env_can_disable_defaults(self, monkeypatch):
        monkeypatch.setenv("QORTEX_GRAPH", "off")
        monkeypatch.setenv("QORTEX_LEARNING", "off")
        flags = FeatureFlags.load()
        assert flags.graph is False
        assert flags.learning is False


class TestPriority:
    """env var > YAML > default."""

    def test_full_priority_chain(self, monkeypatch, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({
            "teleportation": True,
            "credit_propagation": True,
        }))

        # env overrides yaml for teleportation only
        monkeypatch.setenv("QORTEX_TELEPORTATION", "off")

        flags = FeatureFlags.load(yaml_file)
        assert flags.teleportation is False  # env wins over yaml
        assert flags.credit_propagation is True  # yaml wins over default


class TestSingleton:
    """get_flags() returns cached singleton, reset_flags() clears it."""

    def test_singleton(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": True}))

        f1 = get_flags(yaml_file)
        f2 = get_flags()
        assert f1 is f2
        assert f1.teleportation is True

    def test_reset(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({"teleportation": True}))

        f1 = get_flags(yaml_file)
        assert f1.teleportation is True

        reset_flags()
        # After reset, loading without path uses default (no file)
        f2 = get_flags(tmp_path / "nonexistent.yaml")
        assert f2 is not f1
        assert f2.teleportation is False


class TestVecOnlyMode:
    """Everything off except vec store (which has no flag — always on)."""

    def test_vec_only_config(self, tmp_path):
        yaml_file = tmp_path / "flags.yaml"
        yaml_file.write_text(yaml.dump({
            "graph": False,
            "teleportation": False,
            "online_edges": False,
            "enrichment": False,
            "learning": False,
            "causal": False,
            "credit_propagation": False,
        }))

        flags = FeatureFlags.load(yaml_file)
        # Everything is off
        for f_name, val in flags.to_dict().items():
            assert val is False, f"{f_name} should be False in vec-only mode"
