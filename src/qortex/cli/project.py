"""CLI commands for projecting rules from the knowledge graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from qortex.cli._config import get_config
from qortex.cli._errors import handle_error

app = typer.Typer(help="Project rules from the knowledge graph.")


def _get_backend():
    """Get the appropriate backend (InMemory for now, Memgraph when available)."""
    from qortex.core.memory import InMemoryBackend

    backend = InMemoryBackend()
    backend.connect()
    return backend


def _run_projection(
    target_obj: Any,
    domain: str | None,
    enrich: bool,
) -> Any:
    """Shared projection logic for all project commands."""
    from qortex.projectors.projection import Projection
    from qortex.projectors.sources.flat import FlatRuleSource

    backend = _get_backend()
    source = FlatRuleSource(backend=backend)

    enricher = None
    if enrich:
        from qortex.projectors.enrichers.template import TemplateEnricher

        enricher = TemplateEnricher(domain=domain or "general")

    projection = Projection(source=source, enricher=enricher, target=target_obj)
    domains = [domain] if domain else None
    return projection.project(domains=domains)


def _write_output(result: str, output: Path | None, label: str) -> None:
    """Write string result to file or stdout."""
    if output:
        output.write_text(result)
        typer.echo(f"Wrote {label} to {output}")
    else:
        typer.echo(result)


@app.command()
def buildlog(
    domain: str = typer.Option(None, "--domain", "-d", help="Limit to domain"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    enrich: bool = typer.Option(True, "--enrich/--no-enrich", help="Run enrichment"),
    persona: str = typer.Option("qortex", "--persona", "-p", help="Buildlog persona name"),
) -> None:
    """Project rules to buildlog seed YAML format."""
    from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

    import yaml

    target = BuildlogSeedTarget(persona_name=persona)
    result = _run_projection(target, domain, enrich)
    yaml_str = yaml.dump(result, default_flow_style=False, sort_keys=False)

    if output:
        output.write_text(yaml_str)
        typer.echo(f"Wrote buildlog seed to {output} ({result['metadata']['rule_count']} rules)")
    else:
        typer.echo(yaml_str)


@app.command()
def flat(
    domain: str = typer.Option(None, "--domain", "-d", help="Limit to domain"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    enrich: bool = typer.Option(True, "--enrich/--no-enrich", help="Run enrichment"),
) -> None:
    """Project rules to flat YAML format."""
    from qortex.projectors.targets.flat_yaml import FlatYAMLTarget

    target = FlatYAMLTarget()
    result = _run_projection(target, domain, enrich)
    _write_output(result, output, "flat YAML")


@app.command(name="json")
def json_cmd(
    domain: str = typer.Option(None, "--domain", "-d", help="Limit to domain"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    enrich: bool = typer.Option(True, "--enrich/--no-enrich", help="Run enrichment"),
) -> None:
    """Project rules to JSON format."""
    from qortex.projectors.targets.flat_json import FlatJSONTarget

    target = FlatJSONTarget()
    result = _run_projection(target, domain, enrich)
    _write_output(result, output, "JSON")
