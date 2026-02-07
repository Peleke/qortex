"""Qortex CLI -- typer-based command interface.

Commands:
    qortex infra up/down/status    Infrastructure management
    qortex ingest file <path>      Ingest content into KG
    qortex project buildlog/flat/json   Project rules to formats
    qortex inspect domains/rules/stats  Inspect graph contents
    qortex viz open/query          Graph visualization
    qortex interop status/pending/signals  Consumer interop management
    qortex prune manifest/stats    Edge pruning and analysis
    qortex mcp-serve               Start MCP server (stdio or sse)
"""

from __future__ import annotations

import typer

from qortex.cli import infra, ingest, inspect_cmd, interop_cmd, project, prune, viz

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
app.add_typer(interop_cmd.app, name="interop")
app.add_typer(prune.app, name="prune")


@app.command("mcp-serve")
def mcp_serve(
    transport: str = typer.Option("stdio", help="Transport: 'stdio' or 'sse'."),
) -> None:
    """Start the qortex MCP server.

    Exposes qortex as tools any MCP client can use:
    qortex_query, qortex_feedback, qortex_ingest, qortex_domains, qortex_status.
    """
    from qortex.mcp.server import serve

    serve(transport=transport)


def main() -> None:
    """Entry point for the qortex CLI."""
    app()
