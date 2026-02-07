"""CausalBackend protocol and NetworkX implementation.

The protocol mirrors GraphBackend's runtime-checkable pattern.
NetworkXCausalBackend is the Phase 1 concrete implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .dag import CausalDAG
from .dsep import DSeparationEngine
from .types import (
    CausalCapability,
    CausalQuery,
    CausalResult,
    IndependenceAssertion,
    QueryType,
)


@runtime_checkable
class CausalBackend(Protocol):
    """Backend-agnostic causal inference interface."""

    def capabilities(self) -> frozenset[CausalCapability]: ...

    def has_capability(self, cap: CausalCapability) -> bool: ...

    def query(self, cq: CausalQuery) -> CausalResult: ...

    def is_d_separated(
        self,
        x: frozenset[str],
        y: frozenset[str],
        z: frozenset[str],
    ) -> IndependenceAssertion: ...


# =============================================================================
# Phase 1 concrete implementation
# =============================================================================


@dataclass
class NetworkXCausalBackend:
    """Causal backend using networkx d-separation.

    Claims only ``{D_SEPARATION}`` capability.
    """

    dag: CausalDAG
    _engine: DSeparationEngine | None = None

    @property
    def engine(self) -> DSeparationEngine:
        if self._engine is None:
            self._engine = DSeparationEngine(dag=self.dag)
        return self._engine

    def capabilities(self) -> frozenset[CausalCapability]:
        return frozenset({CausalCapability.D_SEPARATION})

    def has_capability(self, cap: CausalCapability) -> bool:
        return cap in self.capabilities()

    def query(self, cq: CausalQuery) -> CausalResult:
        if cq.query_type != QueryType.OBSERVATIONAL:
            raise NotImplementedError(f"{cq.query_type.value} queries require Phase 2+ backends")
        return self.engine.query(cq)

    def is_d_separated(
        self,
        x: frozenset[str],
        y: frozenset[str],
        z: frozenset[str],
    ) -> IndependenceAssertion:
        return self.engine.is_d_separated(x, y, z)
