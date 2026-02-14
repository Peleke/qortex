"""QortexEventLinker: isolated event namespace for qortex observability.

All qortex subscribers register here. Separate from any other
pyventus usage in the process.
"""

from __future__ import annotations

from pyventus.events import EventLinker


class QortexEventLinker(EventLinker):
    """Isolated event namespace for qortex observability."""

    pass
