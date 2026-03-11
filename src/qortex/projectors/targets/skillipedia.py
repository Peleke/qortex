"""SkillipediaTarget -- serialize rules to Skillipedia MDX format.

Produces MDX files with YAML frontmatter suitable for the Skillipedia
knowledge base. Each entry includes provenance, enrichment, and type
inference based on rule metadata.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import yaml

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule
from qortex.projectors.targets._serialize import _build_provenance


@dataclass
class SkillipediaTarget:
    """Serialize rules to Skillipedia MDX format.

    Output is a list of ``{"path": "entries/slug.mdx", "content": "---\\nyaml\\n---\\n\\nbody"}``
    dicts. Supports per-rule and per-domain grouping.

    Implements the ProjectionTarget[list[dict]] protocol.
    """

    group_by: str = "per_rule"
    base_dir: str = "entries"
    include_enrichment: bool = True

    def serialize(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> list[dict]:
        """Serialize rules to Skillipedia MDX file descriptors.

        Returns:
            List of ``{"path": "entries/slug.mdx", "content": "..."}`` dicts.
        """
        if not rules:
            return []

        if self.group_by == "per_domain":
            return self._serialize_per_domain(rules)

        return self._serialize_per_rule(rules)

    def _serialize_per_rule(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> list[dict]:
        """One MDX file per rule."""
        results: list[dict] = []
        for rule in rules:
            slug = self._slugify(rule.id)
            frontmatter = self._build_frontmatter(rule)
            body = self._build_body(rule)
            content = self._render_mdx(frontmatter, body)
            results.append({
                "path": f"{self.base_dir}/{slug}.mdx",
                "content": content,
            })
        return results

    def _serialize_per_domain(
        self, rules: list[EnrichedRule] | list[Rule]
    ) -> list[dict]:
        """One MDX file per domain, combining all rules."""
        grouped: dict[str, list[EnrichedRule | Rule]] = defaultdict(list)
        for rule in rules:
            grouped[rule.domain].append(rule)

        results: list[dict] = []
        for domain, domain_rules in grouped.items():
            slug = self._slugify(domain)
            # Use first rule for frontmatter base, merge source_concepts
            first = domain_rules[0]
            all_source_concepts: list[str] = []
            for r in domain_rules:
                for sc in r.source_concepts:
                    if sc not in all_source_concepts:
                        all_source_concepts.append(sc)

            frontmatter = self._build_frontmatter(first)
            frontmatter["source_concepts"] = all_source_concepts
            frontmatter["id"] = slug

            # Combine bodies
            body_parts: list[str] = []
            for r in domain_rules:
                body_parts.append(self._build_body(r))
            body = "\n\n---\n\n".join(body_parts)

            content = self._render_mdx(frontmatter, body)
            results.append({
                "path": f"{self.base_dir}/{slug}.mdx",
                "content": content,
            })
        return results

    def _build_frontmatter(self, rule: EnrichedRule | Rule) -> dict[str, Any]:
        """Build YAML frontmatter dict for a single rule."""
        r = rule.rule if isinstance(rule, EnrichedRule) else rule

        fm: dict[str, Any] = {
            "id": rule.id,
            "type": self._infer_type(rule),
            "claim": rule.text[:200] if len(rule.text) > 200 else rule.text,
            "confidence": rule.confidence,
            "domain": rule.domain,
            "derivation": rule.derivation,
        }

        # Tags from enrichment
        if (
            self.include_enrichment
            and isinstance(rule, EnrichedRule)
            and rule.enrichment
            and rule.enrichment.tags
        ):
            fm["tags"] = rule.enrichment.tags

        fm["category"] = rule.category or rule.domain
        fm["source_concepts"] = list(rule.source_concepts)
        fm["provenance"] = _build_provenance(rule)

        # Metadata passthrough
        if r.metadata:
            fm["metadata"] = r.metadata

        fm["generated_at"] = datetime.now(UTC).isoformat()

        return fm

    def _build_body(self, rule: EnrichedRule | Rule) -> str:
        """Build the MDX body content for a rule."""
        parts: list[str] = []

        # Rule text as heading
        # Use first 80 chars of rule id as heading, full text as content
        parts.append(f"## {rule.id}\n\n{rule.text}")

        # Enrichment sections
        if (
            self.include_enrichment
            and isinstance(rule, EnrichedRule)
            and rule.enrichment
        ):
            e = rule.enrichment
            if e.context:
                parts.append(f"### Context\n\n{e.context}")
            if e.antipattern:
                parts.append(f"### Antipattern\n\n{e.antipattern}")
            if e.rationale:
                parts.append(f"### Rationale\n\n{e.rationale}")

        return "\n\n".join(parts)

    @staticmethod
    def _render_mdx(frontmatter: dict[str, Any], body: str) -> str:
        """Render frontmatter + body as MDX content."""
        fm_str = yaml.dump(
            frontmatter,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ).rstrip()
        return f"---\n{fm_str}\n---\n\n{body}\n"

    @staticmethod
    def _infer_type(rule: EnrichedRule | Rule) -> str:
        """Infer the Skillipedia entry type from rule metadata.

        Returns:
            "skill" if rule has skill_format in metadata,
            "pattern" if derivation is "derived",
            "learning" otherwise.
        """
        r = rule.rule if isinstance(rule, EnrichedRule) else rule
        if r.metadata.get("skill_format"):
            return "skill"
        if rule.derivation == "derived":
            return "pattern"
        return "learning"

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a filesystem-safe slug, max 80 chars.

        Lowercases, replaces non-alphanumeric sequences with hyphens,
        and strips leading/trailing hyphens.
        """
        slug = text.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return slug[:80]
