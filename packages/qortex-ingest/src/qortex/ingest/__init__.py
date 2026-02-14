"""qortex.ingest: Pluggable ingestors for qortex.

Separate workspace package under packages/qortex-ingest/.
Produces IngestionManifest objects that the KG consumes.
"""

from qortex.core.models import IngestionManifest

__all__ = ["IngestionManifest"]
