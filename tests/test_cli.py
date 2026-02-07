"""Exhaustive tests for the qortex CLI.

Tests all commands: infra, ingest, project, inspect, viz.
Uses typer.testing.CliRunner for isolated CLI testing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from qortex.cli import app
from qortex.cli._config import QortexConfig, get_config

runner = CliRunner()


def _memgraph_available() -> bool:
    """Check if Memgraph is reachable."""
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("localhost", 7687))
        s.close()
        return True
    except Exception:
        return False


MEMGRAPH_RUNNING = _memgraph_available()
skip_if_memgraph_running = pytest.mark.skipif(
    MEMGRAPH_RUNNING, reason="Test expects Memgraph to be down but it's running"
)


# =========================================================================
# App structure
# =========================================================================


class TestAppStructure:
    def test_app_has_all_subcommands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "infra" in result.output
        assert "ingest" in result.output
        assert "project" in result.output
        assert "inspect" in result.output
        assert "viz" in result.output

    def test_infra_help(self):
        result = runner.invoke(app, ["infra", "--help"])
        assert result.exit_code == 0
        assert "up" in result.output
        assert "down" in result.output
        assert "status" in result.output

    def test_project_help(self):
        result = runner.invoke(app, ["project", "--help"])
        assert result.exit_code == 0
        assert "buildlog" in result.output
        assert "flat" in result.output
        assert "json" in result.output

    def test_inspect_help(self):
        result = runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "domains" in result.output
        assert "rules" in result.output
        assert "stats" in result.output

    def test_viz_help(self):
        result = runner.invoke(app, ["viz", "--help"])
        assert result.exit_code == 0
        assert "open" in result.output
        assert "query" in result.output

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # typer with no_args_is_help=True returns exit code 0 on some versions, 2 on others
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "Commands" in result.output


# =========================================================================
# Config
# =========================================================================


class TestConfig:
    def test_default_config(self):
        config = QortexConfig()
        assert config.memgraph_host == "localhost"
        assert config.memgraph_port == 7687
        # Default matches docker-compose defaults
        assert config.memgraph_credentials.user == "memgraph"
        assert config.memgraph_credentials.auth_tuple == ("memgraph", "memgraph")
        assert config.lab_port == 3000

    def test_config_from_env(self):
        with patch.dict(
            os.environ,
            {
                "QORTEX_MEMGRAPH_HOST": "custom-host",
                "QORTEX_MEMGRAPH_PORT": "9999",
                "QORTEX_LAB_PORT": "4000",
            },
        ):
            config = QortexConfig()
            assert config.memgraph_host == "custom-host"
            assert config.memgraph_port == 9999
            assert config.lab_port == 4000

    def test_get_config(self):
        config = get_config()
        assert isinstance(config, QortexConfig)

    def test_invalid_port_gives_helpful_error(self):
        with (
            patch.dict(os.environ, {"QORTEX_MEMGRAPH_PORT": "not_a_number"}),
            pytest.raises(SystemExit),
        ):
            QortexConfig()

    def test_invalid_lab_port_gives_helpful_error(self):
        with patch.dict(os.environ, {"QORTEX_LAB_PORT": "abc"}), pytest.raises(SystemExit):
            QortexConfig()

    def test_compose_file_default(self):
        config = QortexConfig()
        assert config.compose_file == "docker/docker-compose.yml"

    def test_compose_file_from_env(self):
        with patch.dict(os.environ, {"QORTEX_COMPOSE_FILE": "/custom/compose.yml"}):
            config = QortexConfig()
            assert config.compose_file == "/custom/compose.yml"


# =========================================================================
# Infra commands
# =========================================================================


class TestInfraUp:
    @patch("qortex.cli.infra.subprocess.run")
    def test_up_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Started\n", stderr="")
        result = runner.invoke(app, ["infra", "up"])
        assert result.exit_code == 0
        assert "Starting infrastructure" in result.output
        call_args = mock_run.call_args[0][0]
        assert "docker" in call_args
        assert "compose" in call_args
        assert "up" in call_args
        assert "-d" in call_args

    @patch("qortex.cli.infra.subprocess.run")
    def test_up_no_detach(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = runner.invoke(app, ["infra", "up", "--no-detach"])
        assert result.exit_code == 0
        call_args = mock_run.call_args[0][0]
        assert "-d" not in call_args

    @patch("qortex.cli.infra.subprocess.run")
    def test_up_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Connection refused")
        result = runner.invoke(app, ["infra", "up"])
        assert result.exit_code == 1
        assert "Failed to start" in result.output

    @patch("qortex.cli.infra.subprocess.run")
    def test_up_shows_endpoints(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = runner.invoke(app, ["infra", "up"])
        assert "7687" in result.output
        assert "3000" in result.output


class TestInfraDown:
    @patch("qortex.cli.infra.subprocess.run")
    def test_down_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = runner.invoke(app, ["infra", "down"])
        assert result.exit_code == 0
        assert "stopped" in result.output.lower()

    @patch("qortex.cli.infra.subprocess.run")
    def test_down_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Error")
        result = runner.invoke(app, ["infra", "down"])
        assert result.exit_code == 1


class TestInfraStatus:
    @patch("qortex.cli.infra.subprocess.run")
    def test_status_no_containers(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        result = runner.invoke(app, ["infra", "status"])
        assert result.exit_code == 0
        assert "No containers" in result.output or "Not reachable" in result.output

    @patch("qortex.cli.infra.subprocess.run")
    def test_status_with_containers(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="qortex-memgraph  running\n", stderr=""
        )
        result = runner.invoke(app, ["infra", "status"])
        assert result.exit_code == 0
        assert "Containers" in result.output


# =========================================================================
# Ingest command
# =========================================================================


class TestIngest:
    @skip_if_memgraph_running
    def test_ingest_requires_memgraph(self):
        result = runner.invoke(app, ["ingest", "file", "/nonexistent/file.md"])
        assert result.exit_code == 1

    def test_ingest_help(self):
        result = runner.invoke(app, ["ingest", "file", "--help"])
        assert result.exit_code == 0
        assert "path" in result.output.lower()


# =========================================================================
# Project commands
# =========================================================================


def _mock_empty_backend():
    """Return an empty InMemoryBackend for testing."""
    from qortex.core.memory import InMemoryBackend

    backend = InMemoryBackend()
    backend.connect()
    return backend


class TestProjectBuildlog:
    def test_buildlog_empty_graph(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "buildlog"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert parsed["rules"] == []
        assert parsed["persona"] == "qortex"

    def test_buildlog_custom_persona(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "buildlog", "--persona", "custom"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert parsed["persona"] == "custom"

    def test_buildlog_no_enrich(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "buildlog", "--no-enrich"])
        assert result.exit_code == 0

    def test_buildlog_to_file(self, tmp_path):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            output = tmp_path / "seed.yml"
            result = runner.invoke(app, ["project", "buildlog", "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        parsed = yaml.safe_load(output.read_text())
        assert "rules" in parsed

    def test_buildlog_domain_filter(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "buildlog", "--domain", "test"])
        assert result.exit_code == 0


class TestProjectFlat:
    def test_flat_empty_graph(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "flat"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert parsed["rules"] == []

    def test_flat_to_file(self, tmp_path):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            output = tmp_path / "rules.yml"
            result = runner.invoke(app, ["project", "flat", "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()

    def test_flat_no_enrich(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "flat", "--no-enrich"])
        assert result.exit_code == 0


class TestProjectJSON:
    def test_json_empty_graph(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["rules"] == []

    def test_json_to_file(self, tmp_path):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            output = tmp_path / "rules.json"
            result = runner.invoke(app, ["project", "json", "--output", str(output)])
        assert result.exit_code == 0
        assert output.exists()
        parsed = json.loads(output.read_text())
        assert "rules" in parsed

    def test_json_no_enrich(self):
        with patch("qortex.cli.project._get_backend", _mock_empty_backend):
            result = runner.invoke(app, ["project", "json", "--no-enrich"])
        assert result.exit_code == 0


# =========================================================================
# Project commands with populated graph
# =========================================================================


class TestProjectWithData:
    """Verify rules flow through the CLI when the graph has data."""

    def _populate_backend(self, backend):
        from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType

        backend.create_domain("test_domain")
        n1 = ConceptNode(
            id="c1",
            name="Retry",
            description="Retry pattern",
            domain="test_domain",
            source_id="ch1",
        )
        n2 = ConceptNode(
            id="c2",
            name="Timeout",
            description="Timeout config",
            domain="test_domain",
            source_id="ch1",
        )
        backend.add_node(n1)
        backend.add_node(n2)
        edge = ConceptEdge(
            source_id="c1",
            target_id="c2",
            relation_type=RelationType.REQUIRES,
            confidence=0.9,
        )
        backend.add_edge(edge)
        rule = ExplicitRule(
            id="r1",
            text="Always configure timeouts with retries",
            domain="test_domain",
            source_id="ch1",
            confidence=0.95,
        )
        backend.add_rule(rule)

    @patch("qortex.cli.project._get_backend")
    def test_buildlog_with_data(self, mock_get_backend):
        from qortex.core.memory import InMemoryBackend

        backend = InMemoryBackend()
        backend.connect()
        self._populate_backend(backend)
        mock_get_backend.return_value = backend

        result = runner.invoke(app, ["project", "buildlog"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert len(parsed["rules"]) >= 1

    @patch("qortex.cli.project._get_backend")
    def test_flat_with_data(self, mock_get_backend):
        from qortex.core.memory import InMemoryBackend

        backend = InMemoryBackend()
        backend.connect()
        self._populate_backend(backend)
        mock_get_backend.return_value = backend

        result = runner.invoke(app, ["project", "flat"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert len(parsed["rules"]) >= 1

    @patch("qortex.cli.project._get_backend")
    def test_json_with_data(self, mock_get_backend):
        from qortex.core.memory import InMemoryBackend

        backend = InMemoryBackend()
        backend.connect()
        self._populate_backend(backend)
        mock_get_backend.return_value = backend

        result = runner.invoke(app, ["project", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed["rules"]) >= 1


# =========================================================================
# Inspect commands (require Memgraph)
# =========================================================================


class TestInspect:
    @skip_if_memgraph_running
    def test_domains_requires_memgraph(self):
        result = runner.invoke(app, ["inspect", "domains"])
        assert result.exit_code == 1

    @skip_if_memgraph_running
    def test_rules_requires_memgraph(self):
        result = runner.invoke(app, ["inspect", "rules"])
        assert result.exit_code == 1

    @skip_if_memgraph_running
    def test_stats_requires_memgraph(self):
        result = runner.invoke(app, ["inspect", "stats"])
        assert result.exit_code == 1

    def test_domains_help(self):
        result = runner.invoke(app, ["inspect", "domains", "--help"])
        assert result.exit_code == 0

    def test_rules_help(self):
        result = runner.invoke(app, ["inspect", "rules", "--help"])
        assert result.exit_code == 0

    def test_stats_help(self):
        result = runner.invoke(app, ["inspect", "stats", "--help"])
        assert result.exit_code == 0


# =========================================================================
# Viz commands
# =========================================================================


class TestVizOpen:
    @patch("qortex.cli.viz.webbrowser.open")
    def test_open_lab(self, mock_open):
        result = runner.invoke(app, ["viz", "open"])
        assert result.exit_code == 0
        assert "Memgraph Lab" in result.output
        mock_open.assert_called_once_with("http://localhost:3000")

    @patch("qortex.cli.viz.webbrowser.open")
    def test_open_lab_custom_port(self, mock_open):
        with patch.dict(os.environ, {"QORTEX_LAB_PORT": "4000"}):
            result = runner.invoke(app, ["viz", "open"])
            assert result.exit_code == 0
            mock_open.assert_called_once_with("http://localhost:4000")


class TestVizQuery:
    @skip_if_memgraph_running
    def test_query_requires_memgraph(self):
        result = runner.invoke(app, ["viz", "query", "MATCH (n) RETURN n"])
        assert result.exit_code == 1

    def test_query_help(self):
        result = runner.invoke(app, ["viz", "query", "--help"])
        assert result.exit_code == 0


# =========================================================================
# Docker compose file validation
# =========================================================================


class TestDockerCompose:
    def test_compose_file_is_valid_yaml(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        assert "services" in parsed

    def test_compose_has_memgraph_service(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        assert "memgraph" in parsed["services"]

    def test_compose_has_lab_service(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        assert "memgraph-lab" in parsed["services"]

    def test_compose_memgraph_healthcheck(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        mg = parsed["services"]["memgraph"]
        assert "healthcheck" in mg
        assert "start_period" in mg["healthcheck"]

    def test_compose_lab_depends_on_healthy(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        lab = parsed["services"]["memgraph-lab"]
        assert lab["depends_on"]["memgraph"]["condition"] == "service_healthy"

    def test_compose_no_deprecated_version(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        assert "version" not in parsed

    def test_compose_ports(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        mg_ports = parsed["services"]["memgraph"]["ports"]
        assert "7687:7687" in mg_ports
        lab_ports = parsed["services"]["memgraph-lab"]["ports"]
        assert "3000:3000" in lab_ports

    def test_compose_volumes(self):
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        parsed = yaml.safe_load(compose_path.read_text())
        assert "memgraph_data" in parsed["volumes"]
        assert "memgraph_log" in parsed["volumes"]


# =========================================================================
# require_memgraph decorator
# =========================================================================


class TestRequireMemgraph:
    @skip_if_memgraph_running
    def test_decorator_fails_without_memgraph(self):
        from click.exceptions import Exit as ClickExit

        from qortex.cli._errors import require_memgraph

        @require_memgraph
        def dummy_cmd():
            return "should not reach here"

        with pytest.raises((SystemExit, ClickExit)):
            dummy_cmd()


# =========================================================================
# Pyproject.toml validation
# =========================================================================


class TestPyprojectToml:
    def test_has_entry_point(self):
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["scripts"]["qortex"] == "qortex.cli:main"

    def test_has_typer_dependency(self):
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        deps = data["project"]["dependencies"]
        assert any("typer" in d for d in deps)

    def test_has_pyyaml_dependency(self):
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        deps = data["project"]["dependencies"]
        assert any("pyyaml" in d.lower() for d in deps)
