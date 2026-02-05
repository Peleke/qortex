"""CLI commands for projecting rules from the knowledge graph."""

from __future__ import annotations

from pathlib import Path

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


@app.command()
def buildlog(
    domain: str = typer.Option(None, "--domain", "-d", help="Limit to domain"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    enrich: bool = typer.Option(True, "--enrich/--no-enrich", help="Run enrichment"),
    persona: str = typer.Option("qortex", "--persona", "-p", help="Buildlog persona name"),
) -> None:
    """Project rules to buildlog seed YAML format."""
    from qortex.projectors.models import ProjectionFilter
    from qortex.projectors.projection import Projection
    from qortex.projectors.sources.flat import FlatRuleSource
    from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

    backend = _get_backend()
    source = FlatRuleSource(backend=backend)
    target = BuildlogSeedTarget(persona_name=persona)

    enricher = None
    if enrich:
        from qortex.projectors.enrichers.template import TemplateEnricher

        enricher = TemplateEnricher(domain=domain or "general")

    projection = Projection(source=source, enricher=enricher, target=target)
    domains = [domain] if domain else None
    result = projection.project(domains=domains)

    import yaml

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
    from qortex.projectors.projection import Projection
    from qortex.projectors.sources.flat import FlatRuleSource
    from qortex.projectors.targets.flat_yaml import FlatYAMLTarget

    backend = _get_backend()
    source = FlatRuleSource(backend=backend)
    target = FlatYAMLTarget()

    enricher = None
    if enrich:
        from qortex.projectors.enrichers.template import TemplateEnricher

        enricher = TemplateEnricher(domain=domain or "general")

    projection = Projection(source=source, enricher=enricher, target=target)
    domains = [domain] if domain else None
    result = projection.project(domains=domains)

    if output:
        output.write_text(result)
        typer.echo(f"Wrote flat YAML to {output}")
    else:
        typer.echo(result)


@app.command(name="json")
def json_cmd(
    domain: str = typer.Option(None, "--domain", "-d", help="Limit to domain"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file path"),
    enrich: bool = typer.Option(True, "--enrich/--no-enrich", help="Run enrichment"),
) -> None:
    """Project rules to JSON format."""
    from qortex.projectors.projection import Projection
    from qortex.projectors.sources.flat import FlatRuleSource
    from qortex.projectors.targets.flat_json import FlatJSONTarget

    backend = _get_backend()
    source = FlatRuleSource(backend=backend)
    target = FlatJSONTarget()

    enricher = None
    if enrich:
        from qortex.projectors.enrichers.template import TemplateEnricher

        enricher = TemplateEnricher(domain=domain or "general")

    projection = Projection(source=source, enricher=enricher, target=target)
    domains = [domain] if domain else None
    result = projection.project(domains=domains)

    if output:
        output.write_text(result)
        typer.echo(f"Wrote JSON to {output}")
    else:
        typer.echo(result)
