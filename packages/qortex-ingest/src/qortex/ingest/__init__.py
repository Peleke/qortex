"""qortex.ingest: Pluggable ingestors for qortex.

This package is SEPARABLE from qortex core.
It produces IngestionManifest objects that the KG consumes.

Could become its own package later:
- qortex: KG + hippocampus + projectors
- qortex-ingest: PDF, MD, text ingestors
"""

from qortex.core.models import IngestionManifest

__all__ = ["IngestionManifest"]
