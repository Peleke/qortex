"""CLI commands for projecting rules from the knowledge graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from qortex.cli._config import get_config
from qortex.cli._errors import handle_error

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


@app.command()
def skillipedia(
    domain: str = typer.Option(None, "--domain", "-d", help="Limit to domain"),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory for MDX files"),
    enrich: bool = typer.Option(True, "--enrich/--no-enrich", help="Run enrichment"),
    group_by: str = typer.Option(
        "per_rule", "--group-by", "-g", help="Grouping: per_rule or per_domain"
    ),
    no_edges: bool = typer.Option(
        False, "--no-edges", help="Exclude edge-derived rules"
    ),
) -> None:
    """Project rules to Skillipedia MDX format.

    Generates MDX files with YAML frontmatter suitable for Astro sites.

    Examples:
        qortex project skillipedia --output ./content/
        qortex project skillipedia --domain skill:ax-rubric --output ./content/
        qortex project skillipedia --group-by per_domain --output ./content/
    """
    from qortex.projectors.targets.skillipedia import SkillipediaTarget

    target = SkillipediaTarget(group_by=group_by)
    result = _run_projection(target, domain, enrich, include_edge_derived=not no_edges)

    if not result:
        typer.echo("No rules to project.")
        return

    if output:
        output.mkdir(parents=True, exist_ok=True)
        for page in result:
            page_path = output / page["path"]
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_text(page["content"])
        typer.echo(f"Wrote {len(result)} MDX file(s) to {output}/")
    else:
        for page in result:
            typer.echo(f"\n--- {page['path']} ---")
            typer.echo(page["content"][:500])
            if len(page["content"]) > 500:
                typer.echo(f"  ... ({len(page['content'])} chars total)")


@app.command()
def skill(
    domain: str = typer.Option(None, "--domain", "-d", help="Limit to domain"),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory"),
    format: str = typer.Option(
        "claude-code", "--format", "-f", help="Format: claude-code or openclaw"
    ),
    name: str = typer.Option(
        None, "--name", "-n", help="Override skill name (single-skill mode)"
    ),
    enrich: bool = typer.Option(True, "--enrich/--no-enrich", help="Run enrichment"),
    no_edges: bool = typer.Option(
        False, "--no-edges", help="Exclude edge-derived rules"
    ),
) -> None:
    """Project rules to SKILL.md format (Claude Code or OpenClaw).

    Re-emits knowledge graph rules as installable SKILL.md files.

    Examples:
        qortex project skill --domain skill:ax-rubric --output ./emit/
        qortex project skill --format openclaw --output ./emit/
        qortex project skill --name my-skill --format claude-code --output ./emit/
    """
    if format == "claude-code":
        from qortex.projectors.targets.claude_code_skill import ClaudeCodeSkillTarget

        target = ClaudeCodeSkillTarget(skill_name=name)
    elif format == "openclaw":
        from qortex.projectors.targets.openclaw_skill import OpenClawSkillTarget

        target = OpenClawSkillTarget(skill_name=name)
    else:
        handle_error(f"Unknown format: {format}. Use 'claude-code' or 'openclaw'.")

    result = _run_projection(target, domain, enrich, include_edge_derived=not no_edges)

    if not result:
        typer.echo("No rules to project.")
        return

    if output:
        output.mkdir(parents=True, exist_ok=True)
        for item in result:
            item_path = output / item["path"]
            item_path.parent.mkdir(parents=True, exist_ok=True)
            item_path.write_text(item["content"])
        typer.echo(
            f"Wrote {len(result)} SKILL.md file(s) to {output}/ ({format} format)"
        )
    else:
        for item in result:
            typer.echo(f"\n--- {item['path']} ---")
            typer.echo(item["content"][:1000])
            if len(item["content"]) > 1000:
                typer.echo(f"  ... ({len(item['content'])} chars total)")
