"""SKILL.md parser and renderers.

Handles the canonical Agent Skills format (agentskills.io) plus OpenClaw
and ClawHub extensions. Parser auto-detects format by presence of metadata
and platform-specific keys.

Renderers produce format-specific SKILL.md output for round-trip re-emission.

Format variants handled:
  - canonical: name + description (+ optional license, compatibility, allowed-tools)
  - openclaw/hybrid: canonical + metadata.openclaw (multiline YAML or single-line JSON)
  - clawhub: canonical + non-standard fields (model, category, version, keywords)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillMdDocument:
    """Parsed representation of a SKILL.md file."""

    name: str
    description: str
    body: str  # Markdown instructions (everything after frontmatter)

    # Canonical optional fields (agentskills.io spec)
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None

    # OpenClaw-specific (may be empty for canonical skills)
    homepage: str | None = None
    user_invocable: bool = True
    disable_model_invocation: bool = False
    openclaw_metadata: dict[str, Any] = field(default_factory=dict)

    # Full metadata dict (for lossless round-trip of arbitrary metadata)
    metadata: dict[str, Any] = field(default_factory=dict)

    # All raw frontmatter (for lossless round-trip of unknown fields)
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)

    # Source info
    source_path: Path | None = None
    source_format: str = "canonical"  # "canonical" | "openclaw" | "clawhub"

    @property
    def content_hash(self) -> str:
        """Content-addressable hash of name + description + body."""
        content = f"{self.name}:{self.description}:{self.body}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def skill_id(self) -> str:
        """Deterministic skill ID: skill_md:{name}:{hash}."""
        return f"skill_md:{self.name}:{self.content_hash}"


def parse_skill_md(content: str, source_path: Path | None = None) -> SkillMdDocument:
    """Parse a SKILL.md file into a SkillMdDocument.

    Handles canonical (agentskills.io), OpenClaw, and ClawHub formats.

    Args:
        content: Raw SKILL.md file content.
        source_path: Optional path for provenance tracking.

    Returns:
        Parsed SkillMdDocument.

    Raises:
        ValueError: If frontmatter is missing or name/description not found.
    """
    fm_str, body = _split_frontmatter(content)
    frontmatter = _parse_frontmatter(fm_str)
    source_format = _detect_format(frontmatter)

    name = frontmatter.get("name")
    if not name:
        raise ValueError("SKILL.md missing required 'name' field")

    description = frontmatter.get("description")
    if not description:
        raise ValueError("SKILL.md missing required 'description' field")

    # Extract metadata dict
    raw_metadata = frontmatter.get("metadata", {})
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError):
            raw_metadata = {}
    if not isinstance(raw_metadata, dict):
        raw_metadata = {}

    # Extract openclaw sub-key if present
    openclaw_meta = {}
    if isinstance(raw_metadata.get("openclaw"), dict):
        openclaw_meta = raw_metadata["openclaw"]
    elif isinstance(raw_metadata.get("clawdbot"), dict):
        # ClawHub variant uses "clawdbot" namespace
        openclaw_meta = raw_metadata["clawdbot"]

    return SkillMdDocument(
        name=str(name).strip(),
        description=str(description).strip(),
        body=body.strip(),
        license=frontmatter.get("license"),
        compatibility=frontmatter.get("compatibility"),
        allowed_tools=frontmatter.get("allowed-tools"),
        homepage=frontmatter.get("homepage"),
        user_invocable=_parse_bool(frontmatter.get("user-invocable", "yes")),
        disable_model_invocation=_parse_bool(
            frontmatter.get("disable-model-invocation", "no")
        ),
        openclaw_metadata=openclaw_meta,
        metadata=raw_metadata,
        raw_frontmatter=frontmatter,
        source_path=source_path,
        source_format=source_format,
    )


def _split_frontmatter(content: str) -> tuple[str, str]:
    """Split SKILL.md content into frontmatter string and body string.

    Returns (frontmatter_str, body_str). Raises ValueError if no frontmatter.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
    if not match:
        raise ValueError(
            "SKILL.md must start with YAML frontmatter (--- delimiters)"
        )
    return match.group(1), match.group(2)


def _parse_frontmatter(fm_str: str) -> dict[str, Any]:
    """Parse YAML frontmatter, handling OpenClaw's single-line JSON metadata.

    OpenClaw constraint: metadata can be a single-line JSON object.
    We parse the whole frontmatter as YAML first. If `metadata` is a string
    (happens when YAML parses inline JSON as a string), we JSON-decode it.
    """
    import yaml

    try:
        result = yaml.safe_load(fm_str)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML frontmatter: {e}") from e

    if not isinstance(result, dict):
        raise ValueError("YAML frontmatter must be a mapping")

    # Handle metadata that YAML parsed as string (single-line JSON case)
    if isinstance(result.get("metadata"), str):
        try:
            result["metadata"] = json.loads(result["metadata"])
        except (json.JSONDecodeError, TypeError):
            pass  # Leave as string if it's not valid JSON

    return result


def _detect_format(frontmatter: dict[str, Any]) -> str:
    """Detect the SKILL.md format variant.

    Returns: "canonical" | "openclaw" | "clawhub"
    """
    metadata = frontmatter.get("metadata", {})

    # OpenClaw: has metadata.openclaw or metadata.clawdbot, or has homepage/user-invocable
    if isinstance(metadata, dict) and (
        "openclaw" in metadata or "clawdbot" in metadata
    ):
        return "openclaw"

    if "homepage" in frontmatter or "user-invocable" in frontmatter:
        return "openclaw"

    # ClawHub: has non-standard fields like model, category, version, keywords
    clawhub_indicators = {"model", "category", "version", "keywords"}
    if clawhub_indicators & set(frontmatter.keys()):
        return "clawhub"

    return "canonical"


def _parse_bool(value: Any) -> bool:
    """Parse a boolean-ish value from YAML frontmatter."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("yes", "true", "1")
    return bool(value)


def _extract_sections(body: str) -> list[dict[str, str]]:
    """Split markdown body into sections by ## headings.

    Returns list of {"heading": str, "content": str} dicts.
    The first entry may have heading="" if there's content before any ## heading.
    """
    parts = re.split(r"^## (.+)$", body, flags=re.MULTILINE)

    sections = []
    # parts[0] is everything before the first ## heading (preamble)
    preamble = parts[0].strip()
    if preamble:
        sections.append({"heading": "", "content": preamble})

    # After that, pairs of (heading, content)
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append({"heading": heading, "content": content})

    return sections


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def render_claude_code_skill_md(
    name: str,
    description: str,
    body: str,
    *,
    license: str | None = None,
    compatibility: str | None = None,
    allowed_tools: str | None = None,
) -> str:
    """Render a canonical Agent Skills format SKILL.md.

    Frontmatter: name + description + optional license/compatibility/allowed-tools.
    Body: markdown instructions.
    """
    lines = [
        "---",
        f"name: {name}",
        f"description: {_yaml_quote(description)}",
    ]
    if license:
        lines.append(f"license: {license}")
    if compatibility:
        lines.append(f"compatibility: {compatibility}")
    if allowed_tools:
        lines.append(f"allowed-tools: {allowed_tools}")
    lines.append("---")

    return "\n".join(lines) + "\n\n" + body + "\n"


def render_openclaw_skill_md(
    name: str,
    description: str,
    body: str,
    *,
    homepage: str | None = None,
    openclaw_metadata: dict[str, Any] | None = None,
    user_invocable: bool = True,
    disable_model_invocation: bool = False,
    license: str | None = None,
) -> str:
    """Render an OpenClaw format SKILL.md.

    Frontmatter: name + description + homepage + metadata (single-line JSON).
    Body: markdown instructions.

    metadata is rendered as single-line JSON per OpenClaw's parser constraint.
    """
    lines = [
        "---",
        f"name: {name}",
        f"description: {_yaml_quote(description)}",
    ]
    if homepage:
        lines.append(f"homepage: {homepage}")
    if license:
        lines.append(f"license: {license}")
    if not user_invocable:
        lines.append('user-invocable: "no"')
    if disable_model_invocation:
        lines.append('disable-model-invocation: "yes"')
    if openclaw_metadata:
        meta_json = json.dumps(
            {"openclaw": openclaw_metadata}, separators=(",", ":"), ensure_ascii=False
        )
        lines.append(f"metadata: {meta_json}")
    lines.append("---")

    return "\n".join(lines) + "\n\n" + body + "\n"


def _yaml_quote(value: str) -> str:
    """Quote a string for YAML frontmatter if it would be misinterpreted.

    Normalizes to a canonical quoting style:
    - Unquoted when safe (most values)
    - Double-quoted when the value contains `: ` (YAML mapping indicator),
      starts with a YAML indicator char, or contains newlines.

    This means some originally-unquoted descriptions gain quotes on round-trip.
    That's intentional normalization, not data loss.
    """
    if "\n" in value:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    # Values starting with YAML indicators need quoting
    if value and value[0] in ('{', '}', '[', ']', '"', "'", '#', '&', '*', '!', '|', '>', '%', '@', '`'):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    # Colon-space is YAML mapping syntax — must quote to avoid parse errors
    if ": " in value:
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value
