"""Legacy flat rule projector -- thin delegate to the new pipeline.

DEPRECATED: Use Projection(FlatRuleSource(backend), enricher, target) instead.
This module exists only to avoid breaking imports during transition.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from qortex.core.backend import GraphBackend
from qortex.projectors.enrichers.passthrough import PassthroughEnricher
from qortex.projectors.models import ProjectionFilter
from qortex.projectors.projection import Projection
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.targets.flat_yaml import FlatYAMLTarget


@dataclass
class FlatRuleProjector:
    """DEPRECATED: Thin wrapper delegating to Projection pipeline.

    Preserves the old API:
        projector = FlatRuleProjector(backend)
        yaml_str = projector.project(domains=["error_handling"])

    New code should use:
        projection = Projection(
            source=FlatRuleSource(backend),
            enricher=enricher,
            target=target,
        )
        result = projection.project(domains=["error_handling"])
    """

    backend: GraphBackend
    include_derived: bool = True

    def __post_init__(self) -> None:
        warnings.warn(
            "FlatRuleProjector is deprecated. "
            "Use Projection(FlatRuleSource(backend), enricher, target) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._projection = Projection(
            source=FlatRuleSource(
                backend=self.backend,
                include_derived=self.include_derived,
            ),
            enricher=PassthroughEnricher(),
            target=FlatYAMLTarget(),
        )

    def project(
        self,
        domains: list[str] | None = None,
        filters: ProjectionFilter | None = None,
    ) -> str:
        """Project rules to flat YAML. Delegates to Projection pipeline."""
        return self._projection.project(domains=domains, filters=filters)
