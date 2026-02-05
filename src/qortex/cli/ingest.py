"""CLI command for ingesting content into the knowledge graph."""

from __future__ import annotations

from pathlib import Path

import typer

from qortex.cli._errors import get_backend, handle_error, require_memgraph

app = typer.Typer(help="Ingest content into the knowledge graph.")


@app.command()
@require_memgraph
def ingest(
    path: Path = typer.Argument(..., help="Path to file to ingest"),
    domain: str = typer.Option(None, "--domain", "-d", help="Domain name (default: derived from filename)"),
) -> None:
    """Ingest a file into the knowledge graph."""
    if not path.exists():
        handle_error(f"File not found: {path}")

    effective_domain = domain or path.stem.replace(" ", "_").lower()
    backend = get_backend()

    typer.echo(f"Ingesting {path} into domain '{effective_domain}'...")
    typer.echo("(Ingestion pipeline not yet integrated; see Track D for E2E)")
