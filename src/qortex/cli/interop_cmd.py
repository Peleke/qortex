"""CLI commands for consumer interop management."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Manage consumer interop (seed distribution).")


@app.command()
def status() -> None:
    """Show interop status: pending seeds, recent signals, directory paths."""
    from qortex.interop import (
        get_interop_config,
        list_failed_seeds,
        list_pending_seeds,
        list_processed_seeds,
        read_signals,
    )

    config = get_interop_config()

    typer.echo("Interop Configuration")
    typer.echo("=" * 40)
    typer.echo(f"  Pending:    {config.seeds.pending}")
    typer.echo(f"  Processed:  {config.seeds.processed}")
    typer.echo(f"  Failed:     {config.seeds.failed}")
    typer.echo(f"  Signals:    {config.signals.projections}")
    typer.echo()

    pending = list_pending_seeds(config)
    processed = list_processed_seeds(config)
    failed = list_failed_seeds(config)

    typer.echo("Seed Counts")
    typer.echo("-" * 40)
    typer.echo(f"  Pending:    {len(pending)}")
    typer.echo(f"  Processed:  {len(processed)}")
    typer.echo(f"  Failed:     {len(failed)}")
    typer.echo()

    if pending:
        typer.echo("Pending Seeds")
        typer.echo("-" * 40)
        for p in pending[-5:]:  # Show last 5
            typer.echo(f"  {p.name}")
        if len(pending) > 5:
            typer.echo(f"  ... and {len(pending) - 5} more")
        typer.echo()

    if failed:
        typer.echo("Failed Seeds")
        typer.echo("-" * 40)
        for p, err in failed[-3:]:  # Show last 3
            typer.echo(f"  {p.name}")
            if err:
                first_line = err.split('\n')[0][:60]
                typer.echo(f"    Error: {first_line}...")
        typer.echo()

    signals = read_signals(config)
    if signals:
        typer.echo("Recent Signals (last 5)")
        typer.echo("-" * 40)
        for sig in signals[-5:]:
            typer.echo(f"  [{sig.event}] {sig.persona} @ {sig.ts[:19]}")


@app.command()
def pending() -> None:
    """List all pending seed files."""
    from qortex.interop import get_interop_config, list_pending_seeds

    config = get_interop_config()
    seeds = list_pending_seeds(config)

    if not seeds:
        typer.echo("No pending seeds.")
        return

    typer.echo(f"Pending seeds ({len(seeds)}):")
    for seed in seeds:
        typer.echo(f"  {seed}")


@app.command()
def signals(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of signals to show"),
) -> None:
    """Show recent signals from the projection log."""
    from qortex.interop import get_interop_config, read_signals

    config = get_interop_config()
    all_signals = read_signals(config)

    if not all_signals:
        typer.echo("No signals recorded.")
        return

    recent = all_signals[-limit:]
    typer.echo(f"Recent signals (showing {len(recent)} of {len(all_signals)}):")
    for sig in recent:
        typer.echo(f"  [{sig.event}] persona={sig.persona} domain={sig.domain} rules={sig.rule_count}")
        typer.echo(f"    path: {sig.path}")
        typer.echo(f"    ts:   {sig.ts}")
        typer.echo()


@app.command()
def init() -> None:
    """Initialize interop directories."""
    from qortex.interop import get_interop_config

    config = get_interop_config()
    config.ensure_dirs()

    typer.echo("Initialized interop directories:")
    typer.echo(f"  {config.seeds.pending}")
    typer.echo(f"  {config.seeds.processed}")
    typer.echo(f"  {config.seeds.failed}")
    typer.echo(f"  {config.signals.projections.parent}")


@app.command()
def config(
    show: bool = typer.Option(True, "--show/--no-show", help="Show current config"),
    write_default: bool = typer.Option(False, "--write-default", help="Write default config file"),
) -> None:
    """Show or write interop configuration."""
    from qortex.interop import InteropConfig, get_interop_config, write_config, _CONFIG_PATH

    if write_default:
        default_config = InteropConfig()
        path = write_config(default_config)
        typer.echo(f"Wrote default config to {path}")
        return

    if show:
        current = get_interop_config()
        typer.echo(f"Config file: {_CONFIG_PATH}")
        typer.echo(f"  exists: {_CONFIG_PATH.exists()}")
        typer.echo()
        typer.echo("Current settings:")
        typer.echo(f"  seeds.pending:    {current.seeds.pending}")
        typer.echo(f"  seeds.processed:  {current.seeds.processed}")
        typer.echo(f"  seeds.failed:     {current.seeds.failed}")
        typer.echo(f"  signals.projections: {current.signals.projections}")
