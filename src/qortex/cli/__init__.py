"""Qortex CLI -- typer-based command interface.

Commands:
    qortex infra up/down/status    Infrastructure management
    qortex ingest <path>           Ingest content into KG
    qortex project buildlog/flat/json   Project rules to formats
    qortex inspect domains/rules/stats  Inspect graph contents
    qortex viz open/query          Graph visualization
"""

from __future__ import annotations

import typer

from qortex.cli import infra, ingest, inspect_cmd, project, viz

app = typer.Typer(
    name="qortex",
    help="Manage knowledge graphs: ingest content, project rules, inspect results.",
    no_args_is_help=True,
)

app.add_typer(infra.app, name="infra")
app.add_typer(ingest.app, name="ingest")
app.add_typer(project.app, name="project")
app.add_typer(inspect_cmd.app, name="inspect")
app.add_typer(viz.app, name="viz")


def main() -> None:
    """Entry point for the qortex CLI."""
    app()
