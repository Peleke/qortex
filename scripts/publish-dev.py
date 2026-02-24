#!/usr/bin/env python3
"""Publish dev builds of all qortex packages to Test PyPI.

Builds PEP 440 dev-versioned wheels (e.g., 0.7.9.dev20260222143000)
and publishes them to Test PyPI. Downstream consumers install with:

    uv pip install --extra-index-url https://test.pypi.org/simple/ \
        --prerelease=allow qortex

Usage:
    uv run scripts/publish-dev.py              # build + publish all
    uv run scripts/publish-dev.py --dry-run    # build only, don't upload
    uv run scripts/publish-dev.py --package qortex-observe  # single package
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Inline registry loading (avoid import issues when running as script)
ROOT = Path(__file__).parent.parent


def _load_registry() -> dict[str, dict]:
    """Load registry without importing the module."""
    import tomllib

    data = tomllib.loads((ROOT / "qortex-registry.toml").read_text())
    return data["packages"]


def _compute_dev_version(base_version: str) -> str:
    """Compute next dev version: bump patch + add .devYYYYMMDDHHMMSS."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", base_version)
    if not match:
        raise ValueError(f"Cannot parse version: {base_version}")
    major, minor, patch = int(match[1]), int(match[2]), int(match[3])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{major}.{minor}.{patch + 1}.dev{timestamp}"


def _patch_version(pyproject: Path, dev_version: str) -> str:
    """Temporarily rewrite version in pyproject.toml. Returns original content."""
    original = pyproject.read_text()
    patched = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{dev_version}"',
        original,
        count=1,
        flags=re.MULTILINE,
    )
    pyproject.write_text(patched)
    return original


def _build_package(pkg_path: Path, dist_dir: Path) -> list[Path]:
    """Build a wheel for the package. Returns list of built wheel paths."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["python", "-m", "build", "--wheel", "--outdir", str(dist_dir), str(pkg_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  BUILD FAILED: {result.stderr}", file=sys.stderr)
        return []
    return list(dist_dir.glob("*.whl"))


def _publish(dist_dir: Path) -> bool:
    """Publish all wheels in dist_dir to Test PyPI."""
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        print("  No wheels to publish", file=sys.stderr)
        return False

    result = subprocess.run(
        [
            "python",
            "-m",
            "twine",
            "upload",
            "--repository-url",
            "https://test.pypi.org/legacy/",
            *[str(w) for w in wheels],
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  PUBLISH FAILED: {result.stderr}", file=sys.stderr)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish dev builds to Test PyPI")
    parser.add_argument("--dry-run", action="store_true", help="Build only, don't publish")
    parser.add_argument("--package", type=str, help="Build/publish a single package")
    parser.add_argument("--ci", action="store_true", help="CI mode (no publish, just build)")
    args = parser.parse_args()

    packages = _load_registry()
    dist_dir = ROOT / "dist-dev"

    # Filter to single package if requested
    if args.package:
        if args.package not in packages:
            print(f"Unknown package: {args.package}", file=sys.stderr)
            print(f"Available: {', '.join(packages.keys())}", file=sys.stderr)
            sys.exit(1)
        packages = {args.package: packages[args.package]}

    built: list[str] = []
    failed: list[str] = []

    for name, info in packages.items():
        pkg_path = (ROOT / info["path"]).resolve()
        pyproject = pkg_path / "pyproject.toml"

        dev_version = _compute_dev_version(info["version"])
        print(f"\n{'=' * 60}")
        print(f"  {name} {info['version']} -> {dev_version}")
        print(f"  path: {pkg_path}")
        print(f"{'=' * 60}")

        # Patch version
        original = _patch_version(pyproject, dev_version)
        try:
            wheels = _build_package(pkg_path, dist_dir)
            if wheels:
                built.append(f"{name}=={dev_version}")
                for w in wheels:
                    print(f"  built: {w.name}")
            else:
                failed.append(name)
        finally:
            # Always restore original pyproject.toml
            pyproject.write_text(original)

    print(f"\n{'=' * 60}")
    print(f"  Built: {len(built)}, Failed: {len(failed)}")
    for b in built:
        print(f"    {b}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")

    if args.dry_run or args.ci:
        print("\n  Dry run / CI mode — skipping publish")
        return

    if not built:
        print("\n  Nothing to publish")
        sys.exit(1)

    print(f"\n  Publishing {len(built)} packages to Test PyPI...")
    if _publish(dist_dir):
        print("  Published successfully!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
