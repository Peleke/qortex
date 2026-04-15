"""ClaudeCodeSkillTarget -- re-emit rules as Claude Code SKILL.md files.

Serializes rules back into the canonical Agent Skills format (agentskills.io)
so they can be consumed by Claude Code as slash-command skills.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule
from qortex.projectors.skillmd import render_claude_code_skill_md


@dataclass
class ClaudeCodeSkillTarget:
    """Serialize rules to Claude Code SKILL.md files.

    If ``skill_name`` is set, all rules are emitted into a single SKILL.md.
    Otherwise, rules are grouped by domain and each domain gets its own file.

    Implements the ProjectionTarget[list[dict]] protocol.
    """

    skill_name: str | None = None
    include_enrichment: bool = True

    def serialize(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> list[dict]:
        """Serialize rules to a list of SKILL.md file descriptors.

        Returns:
            List of ``{"path": "name/SKILL.md", "content": "..."}`` dicts.
        """
        if not rules:
            return []

        if self.skill_name is not None:
            # All rules into one file
            body = self._rules_to_body(rules)
            description = self._rules_to_description(rules)
            license_val = self._extract_license(rules)
            content = render_claude_code_skill_md(
                name=self.skill_name,
                description=description,
                body=body,
                license=license_val,
            )
            return [{"path": f"{self.skill_name}/SKILL.md", "content": content}]

        # Group by domain
        grouped: dict[str, list[EnrichedRule | Rule]] = defaultdict(list)
        for rule in rules:
            grouped[rule.domain].append(rule)

        results: list[dict] = []
        for domain, domain_rules in grouped.items():
            name = self._domain_to_name(domain)
            body = self._rules_to_body(domain_rules)
            description = self._rules_to_description(domain_rules)
            license_val = self._extract_license(domain_rules)
            content = render_claude_code_skill_md(
                name=name,
                description=description,
                body=body,
                license=license_val,
            )
            results.append({"path": f"{name}/SKILL.md", "content": content})

        return results

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

    @staticmethod
    def _extract_license(rules: list[EnrichedRule] | list[Rule]) -> str | None:
        """Extract license from rule metadata if present."""
        for rule in rules:
            r = rule.rule if isinstance(rule, EnrichedRule) else rule
            license_val = r.metadata.get("license")
            if license_val:
                return str(license_val)
        return None

    @staticmethod
    def _domain_to_name(domain: str) -> str:
        """Convert a domain string to a skill name.

        Strips the ``skill:`` prefix if present.
        """
        if domain.startswith("skill:"):
            return domain[len("skill:"):]
        return domain
