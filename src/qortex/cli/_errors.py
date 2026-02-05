"""CLI error handling and decorators."""

from __future__ import annotations

import functools
from typing import Any, Callable

import typer

from qortex.cli._config import get_config

# Module-level holder for the connected backend (set by require_memgraph)
_current_backend: Any = None


def get_backend() -> Any:
    """Get the backend set by the require_memgraph decorator."""
    return _current_backend


def require_memgraph(f: Callable) -> Callable:
    """Decorator that checks Memgraph connectivity before running a command.

    If Memgraph is not reachable, prints a helpful error and exits.
    Sets the backend accessible via get_backend().
    """

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _current_backend
        config = get_config()
        try:
            from qortex.core.backend import MemgraphBackend

            backend = MemgraphBackend(
                host=config.memgraph_host,
                port=config.memgraph_port,
            )
            backend.connect()
        except Exception:
            typer.echo(
                f"Could not connect to Memgraph at "
                f"{config.memgraph_host}:{config.memgraph_port}.\n"
                f"\n"
                f"Start it with: qortex infra up\n"
                f"Or check: qortex infra status",
                err=True,
            )
            raise typer.Exit(1)
        _current_backend = backend
        return f(*args, **kwargs)

    return wrapper


def handle_error(msg: str) -> None:
    """Print an error message and exit."""
    typer.echo(f"Error: {msg}", err=True)
    raise typer.Exit(1)
