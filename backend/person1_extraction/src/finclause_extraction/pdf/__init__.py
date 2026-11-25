"""PDF loading, native text extraction and OCR."""

from .extractor import (
    EmptyDocumentError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFExtractionError,
    PDFNotFoundError,
    UnsupportedFormatError,
    assess_page,
    extract_native_blocks,
    open_pdf,
)
from .ocr import OCRProcessingError, OCRUnavailableError, check_tesseract, ocr_page

__all__ = [
    "EmptyDocumentError",
    "EncryptedPDFError",
    "InvalidPDFError",
    "PDFExtractionError",
    "PDFNotFoundError",
    "UnsupportedFormatError",
    "assess_page",
    "extract_native_blocks",
    "open_pdf",
    "OCRProcessingError",
    "OCRUnavailableError",
    "check_tesseract",
    "ocr_page",
]
