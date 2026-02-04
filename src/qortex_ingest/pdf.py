"""PDF ingestor - extract text from PDFs.

TODO: Investigate PyMuPDF, pdfplumber for chapter/section detection.
"""

from __future__ import annotations

from .base import Chunk, Ingestor, Source


class PDFIngestor(Ingestor):
    """Ingest PDF sources.

    Extracts text and chunks by page/section.

    Dependencies (not yet added):
    - PyMuPDF (fitz) for text extraction
    - pdfplumber for table extraction
    """

    def chunk(self, source: Source) -> list[Chunk]:
        """Extract and chunk PDF content."""
        if not source.path:
            raise ValueError("PDF source must have a path")

        # TODO M2: Implement PDF extraction
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "PDF ingestion not yet implemented. "
            "Workaround: copy-paste text content and use TextIngestor."
        )

        # Future implementation sketch:
        #
        # import fitz  # PyMuPDF
        #
        # doc = fitz.open(source.path)
        # chunks = []
        #
        # for page_num, page in enumerate(doc):
        #     text = page.get_text()
        #     chunks.append(Chunk(
        #         id=f"page_{page_num}",
        #         content=text,
        #         location=f"Page {page_num + 1}",
        #         page=page_num + 1,
        #     ))
        #
        # return chunks
