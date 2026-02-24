"""Parse qortex-registry.toml and provide package metadata.

Usage:
    from scripts.registry import load_registry
    packages = load_registry()
    for name, info in packages.items():
        print(f"{name} v{info.version} at {info.path}")
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


REGISTRY_PATH = Path(__file__).parent.parent / "qortex-registry.toml"


@dataclass
class PackageInfo:
    name: str
    path: Path
    version: str
    description: str
    extras: list[str] = field(default_factory=list)
    publish_tag: str = ""


@dataclass
class RegistryConfig:
    namespace: str
    pypi_url: str
    test_pypi_url: str


def load_registry(
    path: Path = REGISTRY_PATH,
) -> tuple[RegistryConfig, dict[str, PackageInfo]]:
    """Load the registry and return (config, packages)."""
    data = tomllib.loads(path.read_text())
    root = path.parent

    config = RegistryConfig(
        namespace=data["registry"]["namespace"],
        pypi_url=data["registry"]["pypi_url"],
        test_pypi_url=data["registry"]["test_pypi_url"],
    )

    packages: dict[str, PackageInfo] = {}
    for name, info in data["packages"].items():
        packages[name] = PackageInfo(
            name=name,
            path=(root / info["path"]).resolve(),
            version=info["version"],
            description=info.get("description", ""),
            extras=info.get("extras", []),
            publish_tag=info.get("publish_tag", ""),
        )

    return config, packages
