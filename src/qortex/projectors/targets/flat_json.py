"""FlatJSONTarget -- serialize rules to JSON string."""

from __future__ import annotations

import json
from dataclasses import dataclass

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule
from qortex.projectors.targets._serialize import rule_to_dict


@dataclass
class FlatJSONTarget:
    """Serialize rules to JSON format.

    Output is a JSON string with a list of rule dicts.
    Implements the ProjectionTarget[str] protocol.
    """

    include_enrichment: bool = True
    indent: int = 2

    def serialize(self, rules: list[EnrichedRule] | list[Rule]) -> str:
        """Serialize rules to JSON string."""
        entries = [rule_to_dict(r, self.include_enrichment) for r in rules]
        return json.dumps({"rules": entries}, indent=self.indent, ensure_ascii=False)
