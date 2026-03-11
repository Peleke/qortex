"""OpenClawSkillTarget -- re-emit rules as OpenClaw SKILL.md files.

Same structure as ClaudeCodeSkillTarget but renders OpenClaw format with
metadata.openclaw, homepage, and platform-specific fields.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule
from qortex.projectors.skillmd import render_openclaw_skill_md


@dataclass
class OpenClawSkillTarget:
    """Serialize rules to OpenClaw SKILL.md files.

    If ``skill_name`` is set, all rules are emitted into a single SKILL.md.
    Otherwise, rules are grouped by domain and each domain gets its own file.

    Implements the ProjectionTarget[list[dict]] protocol.
    """

    skill_name: str | None = None
    include_enrichment: bool = True
    default_emoji: str = "\U0001f9e0"
    default_license: str = "MIT"

    def serialize(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> list[dict]:
        """Serialize rules to a list of OpenClaw SKILL.md file descriptors.

        Returns:
            List of ``{"path": "name/SKILL.md", "content": "..."}`` dicts.
        """
        if not rules:
            return []

        if self.skill_name is not None:
            content = self._render_one(self.skill_name, rules)
            return [{"path": f"{self.skill_name}/SKILL.md", "content": content}]

        # Group by domain
        grouped: dict[str, list[EnrichedRule | Rule]] = defaultdict(list)
        for rule in rules:
            grouped[rule.domain].append(rule)

        results: list[dict] = []
        for domain, domain_rules in grouped.items():
            name = self._domain_to_name(domain)
            content = self._render_one(name, domain_rules)
            results.append({"path": f"{name}/SKILL.md", "content": content})

        return results

    def _render_one(
        self,
        name: str,
        rules: list[EnrichedRule] | list[Rule],
    ) -> str:
        """Render a single OpenClaw SKILL.md from a set of rules."""
        body = self._rules_to_body(rules)
        description = self._rules_to_description(rules)
        openclaw_meta = self._extract_openclaw_metadata(rules)
        homepage = self._extract_homepage(rules)
        license_val = self._extract_license(rules)

        return render_openclaw_skill_md(
            name=name,
            description=description,
            body=body,
            homepage=homepage,
            openclaw_metadata=openclaw_meta,
            license=license_val,
        )

    def _rules_to_body(self, rules: list[EnrichedRule] | list[Rule]) -> str:
        """Render rules as markdown body text."""
        parts: list[str] = []
        for rule in rules:
            parts.append(f"## {rule.id}\n\n{rule.text}")
            if (
                self.include_enrichment
                and isinstance(rule, EnrichedRule)
                and rule.enrichment
            ):
                e = rule.enrichment
                if e.context:
                    parts.append(f"\n**Context:** {e.context}")
                if e.antipattern:
                    parts.append(f"\n**Antipattern:** {e.antipattern}")
                if e.rationale:
                    parts.append(f"\n**Rationale:** {e.rationale}")

        return "\n\n".join(parts)

    def _rules_to_description(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> str:
        """Build a description from the first rule's text."""
        if not rules:
            return ""
        first_text = rules[0].text
        return first_text[:200] if len(first_text) > 200 else first_text

    def _extract_openclaw_metadata(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> dict[str, Any]:
        """Extract OpenClaw metadata from rule metadata, falling back to defaults."""
        for rule in rules:
            r = rule.rule if isinstance(rule, EnrichedRule) else rule
            meta = r.metadata
            if isinstance(meta.get("openclaw"), dict):
                return meta["openclaw"]

        # Fallback defaults
        return {
            "emoji": self.default_emoji,
        }

    def _extract_homepage(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> str | None:
        """Pull homepage from rule metadata if present."""
        for rule in rules:
            r = rule.rule if isinstance(rule, EnrichedRule) else rule
            homepage = r.metadata.get("homepage")
            if homepage:
                return str(homepage)
        return None

    def _extract_license(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> str:
        """Extract license from rule metadata, falling back to default."""
        for rule in rules:
            r = rule.rule if isinstance(rule, EnrichedRule) else rule
            license_val = r.metadata.get("license")
            if license_val:
                return str(license_val)
        return self.default_license

    @staticmethod
    def _domain_to_name(domain: str) -> str:
        """Convert a domain string to a skill name.

        Strips the ``skill:`` prefix if present.
        """
        if domain.startswith("skill:"):
            return domain[len("skill:"):]
        return domain
