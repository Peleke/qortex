"""Projection registry — register and retrieve sources, enrichers, targets."""

from __future__ import annotations

from typing import Any

from qortex.projectors.base import Enricher, ProjectionSource, ProjectionTarget

_sources: dict[str, type] = {}
_enrichers: dict[str, type] = {}
_targets: dict[str, type] = {}


def reset() -> None:
    """Clear all registries. Use in test fixtures for isolation."""
    _sources.clear()
    _enrichers.clear()
    _targets.clear()


def register_source(name: str, cls: type) -> None:
    _sources[name] = cls


def register_enricher(name: str, cls: type) -> None:
    _enrichers[name] = cls


def register_target(name: str, cls: type) -> None:
    _targets[name] = cls


def get_source(name: str, **kwargs: Any) -> ProjectionSource:
    if name not in _sources:
        raise KeyError(f"Unknown projection source: {name!r}. Available: {list(_sources)}")
    return _sources[name](**kwargs)


def get_enricher(name: str, **kwargs: Any) -> Enricher:
    if name not in _enrichers:
        raise KeyError(f"Unknown enricher: {name!r}. Available: {list(_enrichers)}")
    return _enrichers[name](**kwargs)


def get_target(name: str, **kwargs: Any) -> ProjectionTarget:
    if name not in _targets:
        raise KeyError(f"Unknown projection target: {name!r}. Available: {list(_targets)}")
    return _targets[name](**kwargs)


def available_sources() -> list[str]:
    return list(_sources)


def available_enrichers() -> list[str]:
    return list(_enrichers)


def available_targets() -> list[str]:
    return list(_targets)
