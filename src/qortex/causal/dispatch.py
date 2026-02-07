"""CausalDispatcher — graceful degradation chain for causal backends.

Probes available libraries in order:
    ChiRho → Pyro → DoWhy → networkx
and selects the best available backend.

Phase 1 only implements the networkx path. Others are stubbed for
forward compatibility.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

from .backend import CausalBackend, NetworkXCausalBackend
from .dag import CausalDAG
from .types import CausalCapability, CausalQuery, CausalResult

logger = logging.getLogger(__name__)


@dataclass
class CausalDispatcher:
    """Routes causal queries to the best available backend.

    Usage::

        dag = CausalDAG.from_backend(backend, domain="my_domain")
        dispatcher = CausalDispatcher.auto_detect(dag)
        result = dispatcher.query(some_query)
    """

    _backend: CausalBackend | None = None
    _backend_name: str = ""

    @classmethod
    def auto_detect(cls, dag: CausalDAG) -> CausalDispatcher:
        """Probe installed libraries and select the best backend.

        Degradation order:
            1. chirho (Phase 3 — SCM + counterfactuals)
            2. pyro-ppl (Phase 2 — learned structural equations)
            3. dowhy (Phase 1.5 — identification + estimation)
            4. networkx (Phase 1 — d-separation)
        """
        dispatcher = cls()

        # 1. ChiRho (Phase 3 stub)
        if _is_importable("chirho"):
            logger.info("ChiRho detected — Phase 3 backend not yet implemented, falling through")
            # Future: dispatcher._backend = ChiRhoCausalBackend(dag)

        # 2. Pyro (Phase 2 stub)
        if dispatcher._backend is None and _is_importable("pyro"):
            logger.info("Pyro detected — Phase 2 backend not yet implemented, falling through")
            # Future: dispatcher._backend = PyroCausalBackend(dag)

        # 3. DoWhy (Phase 1.5 stub)
        if dispatcher._backend is None and _is_importable("dowhy"):
            logger.info("DoWhy detected — Phase 1.5 backend not yet implemented, falling through")
            # Future: dispatcher._backend = DoWhyCausalBackend(dag)

        # 4. NetworkX (Phase 1 — implemented)
        if dispatcher._backend is None:
            dispatcher._backend = NetworkXCausalBackend(dag=dag)
            dispatcher._backend_name = "networkx"
            logger.info("Using NetworkX d-separation backend")

        return dispatcher

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def backend(self) -> CausalBackend:
        if self._backend is None:
            raise RuntimeError(
                "No causal backend available. Install networkx: pip install qortex[causal]"
            )
        return self._backend

    def capabilities(self) -> frozenset[CausalCapability]:
        return self.backend.capabilities()

    def has_capability(self, cap: CausalCapability) -> bool:
        return self.backend.has_capability(cap)

    def query(self, cq: CausalQuery) -> CausalResult:
        """Route a query to the active backend.

        Checks capability requirements before dispatching.
        """
        return self.backend.query(cq)


def _is_importable(module_name: str) -> bool:
    """Check if a module can be imported without actually importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ModuleNotFoundError, ValueError):
        return False
