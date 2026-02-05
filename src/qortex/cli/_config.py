"""CLI configuration via environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
        default_factory=lambda: int(os.environ.get("QORTEX_MEMGRAPH_PORT", "7687"))
    )
    memgraph_user: str = field(
        default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_USER", "qortex")
    )
    memgraph_password: str = field(
        default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_PASSWORD", "qortex")
    )
    lab_port: int = field(
        default_factory=lambda: int(os.environ.get("QORTEX_LAB_PORT", "3000"))
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
