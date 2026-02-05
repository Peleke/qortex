"""CLI commands for infrastructure management (Docker, Memgraph)."""

from __future__ import annotations

import subprocess

import typer

from qortex.cli._config import get_config

app = typer.Typer(help="Manage qortex infrastructure (Memgraph, Lab).")


@app.command()
def up(
    detach: bool = typer.Option(True, "--detach/--no-detach", "-d", help="Run in background"),
) -> None:
    """Start Memgraph and Lab via docker compose."""
    config = get_config()
    cmd = ["docker", "compose", "-f", config.compose_file, "up"]
    if detach:
        cmd.append("-d")
    typer.echo(f"Starting infrastructure ({config.compose_file})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"Failed to start:\n{result.stderr}", err=True)
        raise typer.Exit(1)
    typer.echo(result.stdout)
    typer.echo(
        f"Memgraph: bolt://localhost:{config.memgraph_port}\n"
        f"Lab:      http://localhost:{config.lab_port}"
    )


@app.command()
def down() -> None:
    """Stop Memgraph and Lab."""
    config = get_config()
    cmd = ["docker", "compose", "-f", config.compose_file, "down"]
    typer.echo("Stopping infrastructure...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"Failed to stop:\n{result.stderr}", err=True)
        raise typer.Exit(1)
    typer.echo("Infrastructure stopped.")


@app.command()
def status() -> None:
    """Check Memgraph connectivity and container status."""
    config = get_config()

    # Check docker containers
    cmd = ["docker", "compose", "-f", config.compose_file, "ps", "--format", "table"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        typer.echo("Containers:")
        typer.echo(result.stdout)
    else:
        typer.echo("No containers running.")

    # Check Memgraph connectivity
    typer.echo(f"\nMemgraph ({config.memgraph_host}:{config.memgraph_port}):")
    try:
        from qortex.core.backend import MemgraphBackend

        backend = MemgraphBackend(
            host=config.memgraph_host,
            port=config.memgraph_port,
        )
        backend.connect()
        typer.echo("  Connected (OK)")
    except Exception:
        typer.echo("  Not reachable (run 'qortex infra up' to start)")
