"""FlatRuleSource -- derives rules from the KG via explicit rules + edge templates.

Collects explicit rules stored in the backend, then walks edges to derive
additional rules using the EdgeRuleTemplate registry. Deduplicates by
(source_id, target_id) pair for edge-derived rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, RelationType, Rule
from qortex.core.templates import select_template
from qortex.projectors.models import ProjectionFilter


@dataclass
class FlatRuleSource:
    """Derives rules from the knowledge graph.

    Two strategies:
    1. Collect explicit rules stored in the backend (from ingestion).
    2. Derive rules from edges using EdgeRuleTemplate registry.

    Implements the ProjectionSource protocol.
    """

    backend: InMemoryBackend
    include_derived: bool = True
    _seen_edge_pairs: set[tuple[str, str]] = field(default_factory=set, repr=False)

    def derive(
        self,
        domains: list[str] | None = None,
        filters: ProjectionFilter | None = None,
    ) -> list[Rule]:
        """Derive rules from the graph, optionally filtered."""
        self._seen_edge_pairs = set()
        filt = filters or ProjectionFilter()

        rules: list[Rule] = []

        # 1. Collect explicit rules
        if filt.derivation in ("explicit", "all"):
            rules.extend(self._collect_explicit_rules(domains, filt))

        # 2. Derive from edges
        if self.include_derived and filt.derivation in ("derived", "all"):
            rules.extend(self._derive_rules_from_edges(domains, filt))

        return rules

    def _collect_explicit_rules(
        self,
        domains: list[str] | None,
        filt: ProjectionFilter,
    ) -> list[Rule]:
        """Collect explicit rules from the backend."""
        target_domains = domains or self._all_domain_names()
        rules: list[Rule] = []

        for domain_name in target_domains:
            for explicit_rule in self.backend.get_rules(domain_name):
                rule = Rule(
                    id=explicit_rule.id,
                    text=explicit_rule.text,
                    domain=explicit_rule.domain,
                    derivation="explicit",
                    source_concepts=explicit_rule.concept_ids,
                    confidence=explicit_rule.confidence,
                    category=explicit_rule.category,
                )
                if self._passes_filter(rule, filt):
                    rules.append(rule)

        return rules

    def _derive_rules_from_edges(
        self,
        domains: list[str] | None,
        filt: ProjectionFilter,
    ) -> list[Rule]:
        """Derive rules from KG edges using template registry."""
        target_domains = domains or self._all_domain_names()
        rules: list[Rule] = []

        for domain_name in target_domains:
            nodes = list(self.backend.find_nodes(domain=domain_name, limit=10_000))
            for node in nodes:
                edges = list(self.backend.get_edges(node.id, direction="out"))
                for edge in edges:
                    rule = self._derive_one(edge, domain_name, filt)
                    if rule is not None:
                        rules.append(rule)

        return rules

    def _derive_one(
        self,
        edge: ConceptEdge,
        domain: str,
        filt: ProjectionFilter,
    ) -> Rule | None:
        """Derive a single rule from an edge, or None if filtered/duplicate."""
        # Dedup by (source_id, target_id)
        pair = (edge.source_id, edge.target_id)
        if pair in self._seen_edge_pairs:
            return None
        self._seen_edge_pairs.add(pair)

        # Filter by relation type
        if filt.relation_types and edge.relation_type not in filt.relation_types:
            return None

        # Resolve nodes for template expansion
        source_node = self.backend.get_node(edge.source_id)
        target_node = self.backend.get_node(edge.target_id)
        if source_node is None or target_node is None:
            return None

        # Select template
        template = select_template(edge.relation_type, category_hint=filt.categories[0] if filt.categories else None)

        # Expand template
        text = template.template.format(
            source=source_node.name,
            target=target_node.name,
        )

        rule = Rule(
            id=f"derived:{edge.source_id}->{edge.target_id}:{template.variant}",
            text=text,
            domain=domain,
            derivation="derived",
            source_concepts=[edge.source_id, edge.target_id],
            confidence=edge.confidence,
            category=template.category,
        )

        if self._passes_filter(rule, filt):
            return rule
        return None

    def _passes_filter(self, rule: Rule, filt: ProjectionFilter) -> bool:
        """Check if a rule passes the projection filter."""
        if filt.min_confidence > 0 and rule.confidence < filt.min_confidence:
            return False

        if filt.domains and rule.domain not in filt.domains:
            return False

        if filt.categories and rule.category not in filt.categories:
            return False

        return True

    def _all_domain_names(self) -> list[str]:
        """Get all domain names from the backend."""
        return [d.name for d in self.backend.list_domains()]
