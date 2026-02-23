"""CLI commands for vector migration."""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(help="Migrate vectors between backends.")


@app.command("vec")
def vec(
    source: str = typer.Option(..., "--from", help="Source backend: sqlite, pgvector, numpy"),
    batch_size: int = typer.Option(500, "--batch-size", "-b", help="Vectors per batch"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Read only, don't write"),
) -> None:
    """Migrate vectors from source into the current QORTEX_VEC backend."""
    asyncio.run(_run_migration(source, batch_size, dry_run))


async def _run_migration(source: str, batch_size: int, dry_run: bool) -> None:
    from qortex.service import QortexService

    service = QortexService.from_env()

    if service.vector_index is None:
        typer.echo("Error: no vector index configured. Check QORTEX_VEC env var.", err=True)
        raise typer.Exit(1)

    def on_progress(batches: int, vectors: int) -> None:
        typer.echo(f"  batch {batches}: {vectors} vectors read")

    typer.echo(
        f"Migrating vectors: {source} → {type(service.vector_index).__name__}"
        + (" [DRY RUN]" if dry_run else "")
    )

    result = await service.migrate_vec(
        source_type=source,
        batch_size=batch_size,
        dry_run=dry_run,
    )

    typer.echo(
        f"\nDone: {result['vectors_read']} read, {result['vectors_written']} written, "
        f"{result['batches']} batches, {result['duration_seconds']}s"
    )
