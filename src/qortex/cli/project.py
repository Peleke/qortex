"""CLI commands for projecting rules from the knowledge graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from qortex.cli._config import get_config

app = typer.Typer(help="Project rules from the knowledge graph.")


def _get_backend():
    """Get the appropriate backend. Tries Memgraph, falls back to InMemory."""
    config = get_config()
    try:
        from qortex.core.backend import MemgraphBackend, MemgraphCredentials

        creds = MemgraphCredentials.from_tuple(config.memgraph_credentials.auth_tuple)
        backend = MemgraphBackend(
            host=config.memgraph_host,
            port=config.memgraph_port,
            credentials=creds,
        )
        backend.connect()
        return backend
    except Exception as e:
        import sys

        print(
            f"Warning: Could not connect to Memgraph ({e}), using empty InMemoryBackend",
            file=sys.stderr,
        )
        from qortex.core.memory import InMemoryBackend

        backend = InMemoryBackend()
        backend.connect()
        return backend


def _run_projection(
    target_obj: Any,
    domain: str | None,
    enrich: bool,
    include_edge_derived: bool = True,
) -> Any:
    """Shared projection logic for all project commands."""
    from qortex.projectors.models import ProjectionFilter
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
    filters = ProjectionFilter(include_edge_derived=include_edge_derived)
    return projection.project(domains=domains, filters=filters)


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
    pending: bool = typer.Option(False, "--pending", help="Write to interop pending directory"),
    emit: bool = typer.Option(True, "--emit/--no-emit", help="Emit signal event (with --pending)"),
    no_edges: bool = typer.Option(
        False, "--no-edges", help="Exclude edge-derived rules (DAG relationship templates)"
    ),
) -> None:
    """Project rules to buildlog seed YAML format.

    Use --pending to write to the shared interop directory for consumers.
    Use --output to write to a specific file path.
    Use --no-edges to exclude auto-generated DAG relationship rules.
    If neither is specified, prints to stdout.
    """
    import yaml

    from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

    target = BuildlogSeedTarget(persona_name=persona)
    result = _run_projection(target, domain, enrich, include_edge_derived=not no_edges)

    if pending:
        # Write to interop pending directory
        from qortex.interop import write_seed_to_pending

        seed_path = write_seed_to_pending(
            seed_data=result,
            persona=persona,
            domain=domain or "general",
            emit_signal=emit,
        )
        typer.echo(f"Wrote seed to {seed_path} ({result['metadata']['rule_count']} rules)")
        if emit:
            typer.echo("Signal emitted to projections.jsonl")
    elif output:
        yaml_str = yaml.dump(result, default_flow_style=False, sort_keys=False)
        output.write_text(yaml_str)
        typer.echo(f"Wrote buildlog seed to {output} ({result['metadata']['rule_count']} rules)")
    else:
        yaml_str = yaml.dump(result, default_flow_style=False, sort_keys=False)
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
