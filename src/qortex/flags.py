"""Centralized feature flags: YAML config + env var overrides.

Priority: env var > YAML file > default.
Env vars use QORTEX_{FLAG_NAME} convention (e.g. QORTEX_TELEPORTATION=on).
YAML file default: ~/.qortex/flags.yaml
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no"}
_DEFAULT_PATH = Path("~/.qortex/flags.yaml").expanduser()


@dataclass
class FeatureFlags:
    # Core vec store (embedding + similarity) is always on.
    # Everything below layers on top and can be independently disabled.

    # Graph layer: PPR over knowledge graph (requires a GraphBackend)
    graph: bool = True
    # Interoception: teleportation factors + edge promotion buffer
    teleportation: bool = False
    online_edges: bool = True
    # Enrichment: rule extraction from graph neighborhoods
    enrichment: bool = True
    # Learning: Thompson Sampling bandit for arm selection
    learning: bool = True
    # Causal: DAG construction + d-separation + credit assignment
    causal: bool = True
    # Credit propagation: wire CreditAssigner into feedback loop
    credit_propagation: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> FeatureFlags:
        """Load flags from YAML file, then override with env vars."""
        file_path = path or _DEFAULT_PATH
        file_values: dict[str, bool] = {}

        if file_path.exists():
            raw = yaml.safe_load(file_path.read_text()) or {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if isinstance(v, bool):
                        file_values[k] = v
                    elif isinstance(v, str):
                        file_values[k] = v.lower() in _TRUTHY

        # Build kwargs: file values first, then env overrides
        kwargs: dict[str, bool] = {}
        for f in fields(cls):
            name = f.name
            env_key = f"QORTEX_{name.upper()}"

            if env_key in os.environ:
                val = os.environ[env_key].lower()
                if val in _TRUTHY:
                    kwargs[name] = True
                elif val in _FALSY:
                    kwargs[name] = False
                # Non-boolean values (e.g. QORTEX_GRAPH=memgraph) are
                # ignored — avoids collision with backend selector env vars.
            elif name in file_values:
                kwargs[name] = file_values[name]
            # else: use dataclass default

        return cls(**kwargs)

    def to_dict(self) -> dict[str, bool]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def is_enabled(self, flag_name: str) -> bool:
        return getattr(self, flag_name, False)


# Singleton
_flags: FeatureFlags | None = None
_flags_path: Path | None = None


def get_flags(path: Path | None = None) -> FeatureFlags:
    """Get the singleton FeatureFlags instance."""
    global _flags, _flags_path
    if _flags is None:
        _flags_path = path
        _flags = FeatureFlags.load(path)
    return _flags


def reset_flags() -> None:
    """Reset for testing."""
    global _flags, _flags_path
    _flags = None
    _flags_path = None
