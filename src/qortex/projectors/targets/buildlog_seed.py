"""BuildlogSeedTarget -- thin wrapper over serialize_ruleset for buildlog.

Produces a dict compatible with buildlog's SeedFile.from_dict().
The heavy lifting is in _serialize.serialize_ruleset() (universal schema).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule
from qortex.projectors.targets._serialize import serialize_ruleset


@dataclass
class BuildlogSeedTarget:
    """Serialize rules to buildlog seed format.

    Output is a dict matching the universal rule set schema, which buildlog's
    SeedFile.from_dict() consumes directly. Persona is a flat string
    (buildlog uses it as the filename), version is an int.

    Implements the ProjectionTarget[dict] protocol.
    """

    persona_name: str = "qortex"
    version: int = 1
    source_version: str = "0.1.0"
    graph_version: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def serialize(self, rules: list[EnrichedRule] | list[Rule]) -> dict:
        """Serialize rules to buildlog seed dict."""
        return serialize_ruleset(
            rules,
            persona=self.persona_name,
            version=self.version,
            source="qortex",
            source_version=self.source_version,
            graph_version=self.graph_version,
            extra_metadata=self.extra_metadata,
        )
