"""FlatYAMLTarget -- serialize rules to flat YAML string."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule
from qortex.projectors.targets._serialize import rule_to_dict


@dataclass
class FlatYAMLTarget:
    """Serialize rules to flat YAML format.

    Output is a YAML string with a list of rule dicts.
    Implements the ProjectionTarget[str] protocol.
    """

    include_enrichment: bool = True

    def serialize(self, rules: list[EnrichedRule] | list[Rule]) -> str:
        """Serialize rules to YAML string."""
        entries = [rule_to_dict(r, self.include_enrichment) for r in rules]
        return yaml.dump(
            {"rules": entries},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
