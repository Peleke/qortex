"""CLI commands for graph visualization (Memgraph Lab)."""

from __future__ import annotations

import webbrowser

import typer

from qortex.cli._config import get_config
from qortex.cli._errors import get_backend, require_memgraph

app = typer.Typer(help="Visualize the knowledge graph.")


@app.command(name="open")
def open_lab() -> None:
    """Open Memgraph Lab in the browser."""
    config = get_config()
    url = f"http://localhost:{config.lab_port}"
    typer.echo(f"Opening Memgraph Lab at {url}")
    if not webbrowser.open(url):
        typer.echo(f"Could not open browser. Visit: {url}", err=True)


@app.command()
@require_memgraph
def query(
    cypher: str = typer.Argument(..., help="Cypher query to execute"),
) -> None:
    """Execute a Cypher query and print results."""
    backend = get_backend()
    try:
        results = list(backend.query_cypher(cypher))
        if not results:
            typer.echo("(no results)")
            return
        for row in results:
            typer.echo(row)
    except NotImplementedError:
        typer.echo(
            "Cypher queries require Memgraph backend.\n"
            "InMemoryBackend does not support Cypher.",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Query error: {e}", err=True)
        raise typer.Exit(1)
