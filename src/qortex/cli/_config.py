"""CLI configuration via environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field


def _int_env(var: str, default: int) -> int:
    """Parse an integer from an environment variable with a helpful error on bad input."""
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Error: {var}={raw!r} is not a valid integer", file=sys.stderr)
        raise SystemExit(1)


@dataclass
class QortexConfig:
    """Configuration for the qortex CLI.

    Reads from environment variables with QORTEX_ prefix.
    Falls back to sensible defaults for local development.
    """

    memgraph_host: str = field(
        default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_HOST", "localhost")
    )
    memgraph_port: int = field(
        default_factory=lambda: _int_env("QORTEX_MEMGRAPH_PORT", 7687)
    )
    memgraph_user: str = field(
        default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_USER", "qortex")
    )
    memgraph_password: str = field(
        default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_PASSWORD", "qortex")
    )
    lab_port: int = field(
        default_factory=lambda: _int_env("QORTEX_LAB_PORT", 3000)
    )
    compose_file: str = field(
        default_factory=lambda: os.environ.get(
            "QORTEX_COMPOSE_FILE", "docker/docker-compose.yml"
        )
    )
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )


def get_config() -> QortexConfig:
    """Get the current configuration."""
    return QortexConfig()
