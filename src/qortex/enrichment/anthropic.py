"""Anthropic (Claude) enrichment backend. Batch enrichment via API."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from qortex_observe import emit
from qortex_observe.carbon import calculate_carbon
from qortex_observe.events import CarbonTracked

from qortex.core.models import Rule
from qortex.projectors.models import RuleEnrichment

logger = logging.getLogger(__name__)

ENRICHMENT_SYSTEM_PROMPT = """\
You are a senior engineering advisor who translates knowledge graph rules into
actionable guidance for code review and software development.

You operate in the domain of "{domain}".

Your enrichments must be:
1. SPECIFIC: Reference concrete scenarios, not abstract platitudes
2. ACTIONABLE: A reviewer should know exactly what to look for
3. CONCISE: Each field 1-3 sentences maximum
4. DOMAIN-AWARE: Use terminology from the specific domain

For each rule, provide a JSON object with:
- context: Start with "When..." or "In code that..."
- antipattern: Start with "e.g.," or describe a specific code smell
- rationale: Focus on consequences of violation
- tags: 2-5 lowercase snake_case tags

Respond with a JSON array of enrichments, one per rule, in the same order.
"""

RE_ENRICHMENT_PROMPT = """\
A rule has been previously enriched. New context has been discovered.
Update the enrichment to incorporate the new information WITHOUT discarding
existing insights.

Rule: {rule_text}

Existing enrichment:
- context: {existing_context}
- antipattern: {existing_antipattern}
- rationale: {existing_rationale}
- tags: {existing_tags}

New context: {new_context}

Respond with a single JSON object with updated: context, antipattern, rationale, tags.
"""


class AnthropicEnrichmentBackend:
    """Enriches rules via Claude API. Batch mode: 5-10 rules per call."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 10,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic package required: pip install 'qortex[anthropic]'") from e

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or self.DEFAULT_MODEL
        self._batch_size = batch_size

    def enrich_batch(
        self,
        rules: list[Rule],
        domain: str,
    ) -> list[RuleEnrichment]:
        """Enrich rules in batches via Claude."""
        all_enrichments: list[RuleEnrichment] = []

        for i in range(0, len(rules), self._batch_size):
            batch = rules[i : i + self._batch_size]
            enrichments = self._enrich_one_batch(batch, domain)
            all_enrichments.extend(enrichments)

        return all_enrichments

    def re_enrich(
        self,
        rule: Rule,
        existing: RuleEnrichment,
        new_context: str,
    ) -> RuleEnrichment:
        """Re-enrich a single rule with new context."""
        prompt = RE_ENRICHMENT_PROMPT.format(
            rule_text=rule.text,
            existing_context=existing.context,
            existing_antipattern=existing.antipattern,
            existing_rationale=existing.rationale,
            existing_tags=", ".join(existing.tags),
            new_context=new_context,
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        self._emit_carbon(response)

        data = self._parse_json(response.content[0].text)
        return RuleEnrichment(
            context=data.get("context", existing.context),
            antipattern=data.get("antipattern", existing.antipattern),
            rationale=data.get("rationale", existing.rationale),
            tags=data.get("tags", existing.tags),
            enrichment_version=existing.enrichment_version + 1,
            enriched_at=datetime.now(UTC),
            enrichment_source="anthropic",
            source_contexts=[*existing.source_contexts, new_context],
        )

    def _enrich_one_batch(
        self,
        rules: list[Rule],
        domain: str,
    ) -> list[RuleEnrichment]:
        """Send one batch to Claude and parse response."""
        rules_text = "\n".join(f"{i + 1}. [{r.id}] {r.text}" for i, r in enumerate(rules))

        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=ENRICHMENT_SYSTEM_PROMPT.format(domain=domain),
            messages=[
                {
                    "role": "user",
                    "content": f"Enrich these {len(rules)} rules:\n\n{rules_text}",
                }
            ],
        )
        self._emit_carbon(response)

        try:
            data = self._parse_json(response.content[0].text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            return [self._fallback_enrichment(r) for r in rules]

        enrichments: list[RuleEnrichment] = []

        if isinstance(data, list):
            for item in data:
                enrichments.append(self._parse_enrichment(item))
        else:
            enrichments.append(self._parse_enrichment(data))

        # Pad if API returned fewer than expected
        while len(enrichments) < len(rules):
            enrichments.append(self._fallback_enrichment(rules[len(enrichments)]))

        return enrichments[: len(rules)]

    def _parse_enrichment(self, data: dict[str, Any]) -> RuleEnrichment:
        return RuleEnrichment(
            context=data.get("context", ""),
            antipattern=data.get("antipattern", ""),
            rationale=data.get("rationale", ""),
            tags=data.get("tags", []),
            enrichment_version=1,
            enriched_at=datetime.now(UTC),
            enrichment_source="anthropic",
        )

    def _fallback_enrichment(self, rule: Rule) -> RuleEnrichment:
        """Mechanical fallback when API fails for a rule."""
        return RuleEnrichment(
            context=f"When working in the {rule.domain} domain",
            antipattern="Violating this rule",
            rationale=rule.text,
            tags=[rule.domain],
            enrichment_version=1,
            enriched_at=datetime.now(UTC),
            enrichment_source="template",
        )

    def _emit_carbon(self, response: Any) -> None:
        """Emit CarbonTracked event from an Anthropic API response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        calc = calculate_carbon(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            provider="anthropic",
            model=self._model,
        )
        emit(
            CarbonTracked(
                provider="anthropic",
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                total_co2_grams=calc.total_co2_grams,
                water_ml=calc.water_ml,
                confidence=calc.factor.confidence,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

    def _parse_json(self, text: str) -> Any:
        """Extract JSON from Claude's response (may be wrapped in markdown)."""
        text = text.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
