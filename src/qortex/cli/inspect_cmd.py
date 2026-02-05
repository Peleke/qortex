"""CLI commands for inspecting the knowledge graph."""

from __future__ import annotations

import typer

from qortex.cli._errors import get_backend, require_memgraph

app = typer.Typer(help="Inspect the knowledge graph (domains, rules, stats).")


@app.command()
@require_memgraph
def domains() -> None:
    """List all domains with concept, edge, and rule counts."""
    backend = get_backend()
    domain_list = backend.list_domains()
    if not domain_list:
        typer.echo("No domains found.")
        return

    typer.echo(f"{'Domain':<25} {'Concepts':>10} {'Edges':>8} {'Rules':>8}")
    typer.echo("-" * 55)
    for d in domain_list:
        typer.echo(
            f"{d.name:<25} {d.concept_count:>10} {d.edge_count:>8} {d.rule_count:>8}"
        )


@app.command()
@require_memgraph
def rules(
    domain: str = typer.Option(None, "--domain", "-d", help="Filter by domain"),
) -> None:
    """List rules stored in the knowledge graph."""
    backend = get_backend()
    rule_list = backend.get_rules(domain)
    if not rule_list:
        typer.echo("No rules found.")
        return

    for r in rule_list:
        typer.echo(f"[{r.domain}] {r.id}: {r.text}")


@app.command()
@require_memgraph
def stats() -> None:
    """Show graph statistics (totals across all domains)."""
    backend = get_backend()
    domain_list = backend.list_domains()
    total_concepts = sum(d.concept_count for d in domain_list)
    total_edges = sum(d.edge_count for d in domain_list)
    total_rules = sum(d.rule_count for d in domain_list)

    typer.echo(f"Domains:  {len(domain_list)}")
    typer.echo(f"Concepts: {total_concepts}")
    typer.echo(f"Edges:    {total_edges}")
    typer.echo(f"Rules:    {total_rules}")
