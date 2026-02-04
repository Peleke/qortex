"""Flat rule projector - outputs buildlog-compatible YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import yaml

from qortex.core.backend import GraphBackend
from qortex.core.models import EDGE_RULE_TEMPLATES, RelationType, Rule


@dataclass
class FlatRuleProjector:
    """Project KG into flat rules for buildlog consumption.

    Two modes:
    - Phase C: Collect explicit rules from graph
    - Phase B: Derive rules from edges using templates
    """

    backend: GraphBackend
    include_derived: bool = True  # Include Phase B derived rules

    def project(
        self,
        domains: list[str] | None = None,
        category: str | None = None,
    ) -> list[Rule]:
        """Project rules from specified domains.

        Args:
            domains: Domains to include (None = all)
            category: Filter by category (None = all)

        Returns:
            List of rules ready for buildlog
        """
        rules: list[Rule] = []

        # Phase C: Explicit rules
        rules.extend(self._collect_explicit_rules(domains, category))

        # Phase B: Derived rules (if enabled)
        if self.include_derived:
            rules.extend(self._derive_rules_from_edges(domains))

        return rules

    def _collect_explicit_rules(
        self,
        domains: list[str] | None,
        category: str | None,
    ) -> Iterator[Rule]:
        """Collect explicit rules stored in the graph."""
        # TODO M4: Query graph for stored rules
        # For now, return empty
        return iter([])

    def _derive_rules_from_edges(
        self,
        domains: list[str] | None,
    ) -> Iterator[Rule]:
        """Derive rules from edge relationships (Phase B).

        Uses EDGE_RULE_TEMPLATES to generate rules from relationships.
        """
        # TODO M4: Implement edge-based derivation
        #
        # for domain in (domains or self.backend.list_domains()):
        #     for node in self.backend.find_nodes(domain=domain.name):
        #         for edge in self.backend.get_edges(node.id):
        #             if edge.relation_type in EDGE_RULE_TEMPLATES:
        #                 template = EDGE_RULE_TEMPLATES[edge.relation_type]
        #                 source_node = self.backend.get_node(edge.source_id)
        #                 target_node = self.backend.get_node(edge.target_id)
        #                 if source_node and target_node:
        #                     yield Rule(
        #                         id=f"derived:{edge.source_id}:{edge.target_id}",
        #                         text=template.format(
        #                             source=source_node.name,
        #                             target=target_node.name,
        #                         ),
        #                         domain=domain.name,
        #                         derivation="derived",
        #                         source_concepts=[edge.source_id, edge.target_id],
        #                         confidence=edge.confidence,
        #                     )
        return iter([])

    def to_yaml(self, rules: list[Rule]) -> str:
        """Export rules as buildlog-compatible YAML."""
        data = {
            "rules": [
                {
                    "id": r.id,
                    "text": r.text,
                    "domain": r.domain,
                    "category": r.category,
                    "confidence": r.confidence,
                    "derivation": r.derivation,
                }
                for r in rules
            ]
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
