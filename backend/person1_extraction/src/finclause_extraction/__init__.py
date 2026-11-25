"""FinClause Person 1 module: PDF/OCR text extraction, clause segmentation and
financial entity extraction for personal loan agreements.

Typical use::

    from finclause_extraction import analyze_pdf

    result = analyze_pdf("loan.pdf", use_ocr=True)
    payload = result.model_dump(mode="json")   # JSON-ready dict for an API response

This package extracts and organises information for further analysis and human
review. It does not judge whether any charge is fair, legal or fraudulent.
"""

from .config import EXTRACTION_VERSION, OFFSET_REFERENCE, ExtractionConfig
from .pdf.extractor import (
    EmptyDocumentError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFExtractionError,
    PDFNotFoundError,
    UnsupportedFormatError,
)
from .pdf.ocr import OCRUnavailableError
from .pipeline import ExtractionPipeline, analyze_pdf, analyze_pdf_bytes, analyze_text, get_pipeline
from .schemas import (
    BoundingBox,
    Clause,
    ClauseType,
    DocumentInfo,
    Entity,
    EntityLabel,
    ExtractionResult,
    ExtractionStatus,
    Page,
    TextBlock,
)

__version__ = EXTRACTION_VERSION

__all__ = [
    "EXTRACTION_VERSION",
    "OFFSET_REFERENCE",
    "ExtractionConfig",
    "ExtractionPipeline",
    "analyze_pdf",
    "analyze_pdf_bytes",
    "analyze_text",
    "get_pipeline",
    "BoundingBox",
    "Clause",
    "ClauseType",
    "DocumentInfo",
    "Entity",
    "EntityLabel",
    "ExtractionResult",
    "ExtractionStatus",
    "Page",
    "TextBlock",
    "PDFExtractionError",
    "PDFNotFoundError",
    "InvalidPDFError",
    "EncryptedPDFError",
    "UnsupportedFormatError",
    "EmptyDocumentError",
    "OCRUnavailableError",
]
