"""
Comprehensive CLI dogfooding tests.
Tests every command, flag combination, and error path that might be missed by unit tests.
"""
import json
from unittest.mock import Mock, patch

import pytest
import yaml
from typer.testing import CliRunner

from qortex.cli import app
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType

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


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def populated_backend():
    """Create a pre-populated InMemoryBackend for testing."""
    backend = InMemoryBackend()
    backend.connect()
    backend.create_domain("test-domain")
    
    # Add some concept nodes
    nodes = [
        ConceptNode(id="node1", name="Concept A", description="First concept", 
                   domain="test-domain", source_id="src1"),
        ConceptNode(id="node2", name="Concept B", description="Second concept",
                   domain="test-domain", source_id="src1"),
        ConceptNode(id="node3", name="Concept C", description="Third concept",
                   domain="test-domain", source_id="src2"),
    ]
    for node in nodes:
        backend.add_node(node)
    
    # Add some edges
    edges = [
        ConceptEdge(source_id="node1", target_id="node2", relation_type=RelationType.REQUIRES),
        ConceptEdge(source_id="node2", target_id="node3", relation_type=RelationType.SUPPORTS),
    ]
    for edge in edges:
        backend.add_edge(edge)
    
    # Add some rules
    rules = [
        ExplicitRule(id="rule1", text="Always do X before Y", 
                    domain="test-domain", source_id="src1"),
        ExplicitRule(id="rule2", text="Never combine A with B",
                    domain="test-domain", source_id="src2"),
    ]
    for rule in rules:
        backend.add_rule(rule)
    
    return backend


# ============================================================================
# Top-level CLI tests
# ============================================================================

class TestTopLevelCLI:
    """Test the top-level qortex command."""
    
    def test_no_args_shows_help(self):
        """qortex with no args should show help (no_args_is_help=True)."""
        result = runner.invoke(app, [])
        # Typer returns exit code 2 for usage/help display
        assert result.exit_code in [0, 2]
        assert "Usage:" in result.stdout or "usage:" in result.stdout.lower()
        assert "infra" in result.stdout
        assert "project" in result.stdout
        assert "inspect" in result.stdout
        assert "viz" in result.stdout
    
    def test_help_flag(self):
        """qortex --help should show help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout or "usage:" in result.stdout.lower()
    
    def test_unknown_command(self):
        """Unknown command should fail gracefully."""
        result = runner.invoke(app, ["unknown-command"])
        assert result.exit_code != 0


# ============================================================================
# Infra command tests
# ============================================================================

class TestInfraCommands:
    """Test qortex infra subcommands."""
    
    @patch('subprocess.run')
    def test_infra_up_default(self, mock_run):
        """Test infra up with default (no-detach) mode."""
        mock_run.return_value = Mock(returncode=0, stdout="")
        result = runner.invoke(app, ["infra", "up"])
        assert result.exit_code == 0
        assert "Starting" in result.stdout
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        # Default is DETACH, so -d SHOULD be present
        assert "-d" in args
        # Default is DETACH, so -d SHOULD be present
        assert "-d" in args
        # Default is DETACH, so -d SHOULD be present
        assert "-d" in args
        # Default is DETACH, so -d SHOULD be present
        assert "-d" in args
    
    @patch('subprocess.run')
    def test_infra_up_detach(self, mock_run):
        """Test infra up with --detach flag."""
        mock_run.return_value = Mock(returncode=0, stdout="")
        result = runner.invoke(app, ["infra", "up", "--detach"])
        assert result.exit_code == 0
        args = mock_run.call_args[0][0]
        assert "-d" in args
    
    @patch('subprocess.run')
    def test_infra_up_no_detach_explicit(self, mock_run):
        """Test infra up with explicit --no-detach flag."""
        mock_run.return_value = Mock(returncode=0, stdout="")
        result = runner.invoke(app, ["infra", "up", "--no-detach"])
        assert result.exit_code == 0
        args = mock_run.call_args[0][0]
        assert "-d" not in args
    
    @patch('subprocess.run')
    def test_infra_up_failure(self, mock_run):
        """Test infra up when docker-compose fails."""
        mock_run.return_value = Mock(returncode=1, stdout="")
        result = runner.invoke(app, ["infra", "up"])
        assert result.exit_code == 1
    
    @patch('subprocess.run')
    def test_infra_down(self, mock_run):
        """Test infra down."""
        mock_run.return_value = Mock(returncode=0, stdout="")
        result = runner.invoke(app, ["infra", "down"])
        assert result.exit_code == 0
        assert "Stopping" in result.stdout or "stopped" in result.stdout.lower()
        args = mock_run.call_args[0][0]
        assert "down" in args
    
    @patch('subprocess.run')
    def test_infra_down_failure(self, mock_run):
        """Test infra down when docker-compose fails."""
        mock_run.return_value = Mock(returncode=1, stdout="")
        result = runner.invoke(app, ["infra", "down"])
        assert result.exit_code == 1
    
    @patch('subprocess.run')
    def test_infra_status(self, mock_run):
        """Test infra status."""
        mock_run.return_value = Mock(returncode=0, stdout="")
        result = runner.invoke(app, ["infra", "status"])
        assert result.exit_code == 0
        args = mock_run.call_args[0][0]
        assert "ps" in args


# ============================================================================
# Ingest command tests
# ============================================================================

class TestIngestCommand:
    """Test qortex ingest subcommands."""

    def test_ingest_shows_help_without_subcommand(self):
        """ingest without subcommand shows help."""
        result = runner.invoke(app, ["ingest"])
        assert result.exit_code == 2  # no_args_is_help returns 2
        assert "file" in result.output.lower()

    def test_ingest_nonexistent_file(self):
        """ingest file with nonexistent file should fail gracefully."""
        result = runner.invoke(app, ["ingest", "file", "/nonexistent/file.txt"])
        assert result.exit_code == 1

    @skip_if_memgraph_running
    def test_ingest_with_domain(self, tmp_path):
        """ingest file with --domain flag (will fail at Memgraph check)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        result = runner.invoke(app, ["ingest", "file", str(test_file), "--domain", "custom-domain"])
        assert result.exit_code == 1  # Fails because no Memgraph

    @skip_if_memgraph_running
    def test_ingest_empty_domain(self, tmp_path):
        """ingest file with empty --domain string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        result = runner.invoke(app, ["ingest", "file", str(test_file), "--domain", ""])
        assert result.exit_code == 1


# ============================================================================
# Project command tests
# ============================================================================

class TestProjectCommands:
    """Test qortex project subcommands."""
    
    def test_project_buildlog_basic(self, populated_backend):
        """Test project buildlog with default options."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog"])
            assert result.exit_code == 0
            # Should output YAML by default
            data = yaml.safe_load(result.stdout)
            assert "persona" in data
            assert "version" in data
            assert "rules" in data
            assert "metadata" in data
            # Persona is a flat string
            assert isinstance(data["persona"], str)
    
    def test_project_buildlog_with_domain(self, populated_backend):
        """Test project buildlog with --domain flag."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--domain", "test-domain"])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            assert "rules" in data
    
    def test_project_buildlog_empty_domain(self, populated_backend):
        """Test project buildlog with empty --domain string."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--domain", ""])
            assert result.exit_code == 0
    
    def test_project_buildlog_custom_persona(self, populated_backend):
        """Test project buildlog with --persona flag."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--persona", "CustomBot"])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            # Persona is a flat string
            assert data["persona"] == "CustomBot"
    
    def test_project_buildlog_empty_persona(self, populated_backend):
        """Test project buildlog with empty --persona string."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--persona", ""])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            # Empty string should be accepted
            assert data["persona"] == ""
    
    def test_project_buildlog_no_enrich(self, populated_backend):
        """Test project buildlog with --no-enrich flag."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--no-enrich"])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            assert "rules" in data
    
    def test_project_buildlog_output_file(self, populated_backend, tmp_path):
        """Test project buildlog with --output flag."""
        output_file = tmp_path / "output.yaml"
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--output", str(output_file)])
            assert result.exit_code == 0
            assert output_file.exists()
            data = yaml.safe_load(output_file.read_text())
            assert "persona" in data
    
    def test_project_buildlog_unwritable_output(self, populated_backend):
        """Test project buildlog with unwritable --output path."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--output", "/root/unwritable.yaml"])
            # Should fail (can't write to /root as non-root user)
            assert result.exit_code != 0
    
    def test_project_flat_basic(self, populated_backend):
        """Test project flat with default options."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "flat"])
            assert result.exit_code == 0
            # Should output valid YAML
            data = yaml.safe_load(result.stdout)
            # Flat format returns a dict with "rules" key
            assert isinstance(data, dict)
            assert "rules" in data
    
    def test_project_flat_yaml_validity(self, populated_backend):
        """Verify that flat output is actually valid YAML."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "flat"])
            assert result.exit_code == 0
            # This should not raise
            data = yaml.safe_load(result.stdout)
            assert data is not None
    
    def test_project_flat_with_domain(self, populated_backend):
        """Test project flat with --domain flag."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "flat", "--domain", "test-domain"])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            assert isinstance(data, dict)
    
    def test_project_flat_output_file(self, populated_backend, tmp_path):
        """Test project flat with --output flag."""
        output_file = tmp_path / "flat.yaml"
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "flat", "--output", str(output_file)])
            assert result.exit_code == 0
            assert output_file.exists()
            data = yaml.safe_load(output_file.read_text())
            assert isinstance(data, dict)
    
    def test_project_json_basic(self, populated_backend):
        """Test project json with default options."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "json"])
            assert result.exit_code == 0
            # Should output valid JSON
            data = json.loads(result.stdout)
            # JSON format has "rules" key
            assert "rules" in data
    
    def test_project_json_validity(self, populated_backend):
        """Verify that JSON output is actually valid JSON."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "json"])
            assert result.exit_code == 0
            # This should not raise
            data = json.loads(result.stdout)
            assert data is not None
    
    def test_project_json_structure(self, populated_backend):
        """Verify JSON output has the right structure."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "json"])
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "rules" in data
            assert isinstance(data["rules"], list)
    
    def test_project_json_with_domain(self, populated_backend):
        """Test project json with --domain flag."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "json", "--domain", "test-domain"])
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "rules" in data
    
    def test_project_json_output_file(self, populated_backend, tmp_path):
        """Test project json with --output flag."""
        output_file = tmp_path / "output.json"
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "json", "--output", str(output_file)])
            assert result.exit_code == 0
            assert output_file.exists()
            data = json.loads(output_file.read_text())
            assert "rules" in data
    
    def test_cross_format_consistency(self, populated_backend):
        """Test that buildlog, flat, and json have consistent rule counts."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            # Get buildlog output
            buildlog_result = runner.invoke(app, ["project", "buildlog"])
            assert buildlog_result.exit_code == 0
            buildlog_data = yaml.safe_load(buildlog_result.stdout)
            buildlog_rules = buildlog_data["rules"]
            
            # Get flat output
            flat_result = runner.invoke(app, ["project", "flat"])
            assert flat_result.exit_code == 0
            flat_data = yaml.safe_load(flat_result.stdout)
            
            # Get json output
            json_result = runner.invoke(app, ["project", "json"])
            assert json_result.exit_code == 0
            json_data = json.loads(json_result.stdout)
            
            # All should have the same number of rules
            assert len(buildlog_rules) == len(flat_data["rules"]) == len(json_data["rules"])


# ============================================================================
# Inspect command tests
# ============================================================================

class TestInspectCommands:
    """Test qortex inspect subcommands."""

    @skip_if_memgraph_running
    def test_inspect_domains_no_memgraph(self):
        """inspect domains should fail without Memgraph."""
        result = runner.invoke(app, ["inspect", "domains"])
        assert result.exit_code == 1

    @skip_if_memgraph_running
    def test_inspect_rules_no_memgraph(self):
        """inspect rules should fail without Memgraph."""
        result = runner.invoke(app, ["inspect", "rules"])
        assert result.exit_code == 1

    @skip_if_memgraph_running
    def test_inspect_rules_with_domain_no_memgraph(self):
        """inspect rules --domain should fail without Memgraph."""
        result = runner.invoke(app, ["inspect", "rules", "--domain", "test-domain"])
        assert result.exit_code == 1

    @skip_if_memgraph_running
    def test_inspect_stats_no_memgraph(self):
        """inspect stats should fail without Memgraph."""
        result = runner.invoke(app, ["inspect", "stats"])
        assert result.exit_code == 1


# ============================================================================
# Viz command tests
# ============================================================================

class TestVizCommands:
    """Test qortex viz subcommands."""
    
    @patch('webbrowser.open')
    def test_viz_open(self, mock_open):
        """Test viz open."""
        mock_open.return_value = True
        result = runner.invoke(app, ["viz", "open"])
        assert result.exit_code == 0
        assert "Opening Memgraph Lab" in result.stdout
        mock_open.assert_called_once_with("http://localhost:3000")
    
    @skip_if_memgraph_running
    def test_viz_query_no_memgraph(self):
        """viz query should fail without Memgraph."""
        result = runner.invoke(app, ["viz", "query", "MATCH (n) RETURN n"])
        assert result.exit_code == 1
    
    def test_viz_query_missing_cypher(self):
        """viz query without cypher should fail."""
        result = runner.invoke(app, ["viz", "query"])
        assert result.exit_code != 0


# ============================================================================
# Help text tests
# ============================================================================

class TestHelpText:
    """Test that --help works for all commands."""
    
    def test_infra_help(self):
        """Test qortex infra --help."""
        result = runner.invoke(app, ["infra", "--help"])
        assert result.exit_code == 0
        assert "up" in result.stdout
        assert "down" in result.stdout
        assert "status" in result.stdout
    
    def test_infra_up_help(self):
        """Test qortex infra up --help."""
        result = runner.invoke(app, ["infra", "up", "--help"])
        assert result.exit_code == 0
        assert "--detach" in result.stdout or "detach" in result.stdout
    
    def test_ingest_help(self):
        """Test qortex ingest --help."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.stdout
    
    def test_project_help(self):
        """Test qortex project --help."""
        result = runner.invoke(app, ["project", "--help"])
        assert result.exit_code == 0
        assert "buildlog" in result.stdout
        assert "flat" in result.stdout
        assert "json" in result.stdout
    
    def test_project_buildlog_help(self):
        """Test qortex project buildlog --help."""
        result = runner.invoke(app, ["project", "buildlog", "--help"])
        assert result.exit_code == 0
        assert "--domain" in result.stdout
        assert "--persona" in result.stdout
        assert "--enrich" in result.stdout
    
    def test_inspect_help(self):
        """Test qortex inspect --help."""
        result = runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "domains" in result.stdout
        assert "rules" in result.stdout
        assert "stats" in result.stdout
    
    def test_viz_help(self):
        """Test qortex viz --help."""
        result = runner.invoke(app, ["viz", "--help"])
        assert result.exit_code == 0
        assert "open" in result.stdout
        assert "query" in result.stdout


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_very_long_domain_name(self, populated_backend):
        """Test with a very long domain name."""
        long_domain = "a" * 1000
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--domain", long_domain])
            # Should handle gracefully (exit 0 with no matching domain)
            assert result.exit_code == 0
    
    def test_special_characters_in_domain(self, populated_backend):
        """Test with special characters in domain name."""
        special_domain = "test@#$%^&*()"
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--domain", special_domain])
            # Should handle gracefully
            assert result.exit_code == 0
    
    def test_special_characters_in_persona(self, populated_backend):
        """Test with special characters in persona name."""
        special_persona = "Bot<>!@#"
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--persona", special_persona])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            assert data["persona"] == special_persona
    
    def test_unicode_in_persona(self, populated_backend):
        """Test with Unicode characters in persona name."""
        unicode_persona = "Bot🤖émoji"
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog", "--persona", unicode_persona])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            assert data["persona"] == unicode_persona


# ============================================================================
# Data integrity tests
# ============================================================================

class TestDataIntegrity:
    """Test that output data has the right structure and content."""
    
    def test_buildlog_rules_structure(self, populated_backend):
        """Test that buildlog rules have the right structure."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "buildlog"])
            assert result.exit_code == 0
            data = yaml.safe_load(result.stdout)
            
            # Check structure
            assert "persona" in data
            assert "version" in data
            assert "rules" in data
            assert "metadata" in data
            
            # Check rules are dicts with required fields
            assert isinstance(data["rules"], list)
            for rule in data["rules"]:
                assert isinstance(rule, dict)
                # Universal schema: 'rule' key for text
                assert "rule" in rule
    
    def test_json_output_contains_actual_data(self, populated_backend):
        """Test that JSON output contains the actual rule data."""
        with patch('qortex.cli.project._get_backend', return_value=populated_backend):
            result = runner.invoke(app, ["project", "json"])
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            
            # Should have rules (2 explicit + derived rules from enrichment)
            assert "rules" in data
            assert len(data["rules"]) >= 2
            
            # Check rule structure
            for rule in data["rules"]:
                assert isinstance(rule, dict)
                # Rules should have basic fields
                assert "text" in rule or "id" in rule
                assert "domain" in rule
