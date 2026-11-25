"""Native PDF text extraction with PyMuPDF.

Responsibilities:
* open and validate a PDF (missing, corrupted, encrypted, non-PDF, zero pages);
* extract blocks -> lines -> words with bounding boxes, in reading order;
* assess each page: empty, insufficient text (needs OCR), garbled text layer.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pymupdf

from ..config import ExtractionConfig
from ..schemas import BoundingBox, DocumentInfo, ExtractionMethod, Page, TextBlock, TextLine, Word

logger = logging.getLogger(__name__)

PdfSource = Union[str, Path, bytes]

_BOLD_FLAG = 16
_WHITESPACE_RE = re.compile(r"\s")


class PDFExtractionError(Exception):
    """Base class for document-level extraction failures."""

    code = "PDF_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PDFNotFoundError(PDFExtractionError):
    code = "FILE_NOT_FOUND"


class InvalidPDFError(PDFExtractionError):
    code = "INVALID_PDF"


class EncryptedPDFError(PDFExtractionError):
    code = "ENCRYPTED_PDF"


class UnsupportedFormatError(PDFExtractionError):
    code = "UNSUPPORTED_FORMAT"


class EmptyDocumentError(PDFExtractionError):
    code = "EMPTY_DOCUMENT"


@dataclass
class PageAssessment:
    """Result of inspecting a page's native text layer."""

    native_char_count: int
    garbage_ratio: float
    has_images: bool
    has_drawings: bool
    is_empty: bool
    needs_ocr: bool
    reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Opening / validation
# --------------------------------------------------------------------------- #
def read_pdf_bytes(source: PdfSource) -> tuple[bytes, str]:
    """Return ``(data, filename)`` for a path or raw bytes.

    Raises:
        PDFNotFoundError: path does not exist or is not a file.
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "document.pdf"
    path = Path(source)
    if not path.exists():
        raise PDFNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise PDFNotFoundError(f"Path is not a file: {path}")
    return path.read_bytes(), path.name


def open_pdf(data: bytes, filename: str = "document.pdf", password: Optional[str] = None) -> pymupdf.Document:
    """Open and validate PDF bytes.

    Raises:
        InvalidPDFError: empty, corrupted or unreadable data.
        UnsupportedFormatError: readable by MuPDF but not a PDF (e.g. an image or XPS).
        EncryptedPDFError: password protected and no/incorrect password supplied.
        EmptyDocumentError: PDF has zero pages.
    """
    if not data:
        raise InvalidPDFError(f"'{filename}' is empty (0 bytes).")
    if b"%PDF-" not in data[:1024]:
        # Still let MuPDF try (it may be a different supported format) so that we
        # can give a precise error message.
        logger.debug("No %%PDF- header found in first 1KB of %s", filename)
    try:
        doc = pymupdf.open(stream=data, filetype=None if b"%PDF-" not in data[:1024] else "pdf")
    except Exception as exc:  # pymupdf raises FileDataError / RuntimeError variants
        raise InvalidPDFError(f"'{filename}' is not a valid or readable PDF: {exc}") from exc

    if not doc.is_pdf:
        doc.close()
        raise UnsupportedFormatError(f"'{filename}' is not a PDF document (detected another format).")

    if doc.needs_pass:
        if not doc.authenticate(password or ""):
            doc.close()
            hint = "The supplied password is incorrect." if password else "Supply the password via `password=`."
            raise EncryptedPDFError(f"'{filename}' is password protected. {hint}")
        logger.info("Opened encrypted PDF %s", filename)

    if doc.page_count == 0:
        doc.close()
        raise EmptyDocumentError(f"'{filename}' contains no pages.")
    return doc


def build_document_info(doc: pymupdf.Document, data: bytes, filename: str, document_id: str) -> DocumentInfo:
    """Collect file-level metadata."""
    meta = {k: str(v) for k, v in (doc.metadata or {}).items() if v}
    return DocumentInfo(
        document_id=document_id,
        filename=filename,
        file_size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        page_count=doc.page_count,
        is_encrypted=bool(doc.is_encrypted or doc.metadata.get("encryption")),
        pdf_metadata=meta,
    )


def default_document_id(data: bytes) -> str:
    """Deterministic document ID derived from the file content."""
    return "doc_" + hashlib.sha256(data).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Page extraction
# --------------------------------------------------------------------------- #
def _bbox(seq) -> BoundingBox:
    return BoundingBox(x0=round(seq[0], 2), y0=round(seq[1], 2), x1=round(seq[2], 2), y1=round(seq[3], 2))


def _words_from_chars(chars: list[tuple[str, tuple]], include_words: bool) -> list[Word]:
    """Group characters (with boxes) into whitespace-separated words."""
    if not include_words:
        return []
    words: list[Word] = []
    start: Optional[int] = None
    boxes: list[tuple] = []
    for i, (c, box) in enumerate(chars + [(" ", None)]):
        if c.isspace():
            if start is not None:
                text = "".join(ch for ch, _ in chars[start:i])
                words.append(
                    Word(
                        text=text,
                        bbox=_bbox((min(b[0] for b in boxes), min(b[1] for b in boxes),
                                    max(b[2] for b in boxes), max(b[3] for b in boxes))),
                        char_start=start,
                        char_end=i,
                    )
                )
                start, boxes = None, []
        else:
            if start is None:
                start = i
            boxes.append(box)
    return words


def extract_native_blocks(
    page: pymupdf.Page, page_number: int, document_id: str, include_words: bool = True
) -> list[TextBlock]:
    """Extract text blocks/lines/words from a page's text layer in reading order.

    Reading order: blocks are sorted top-to-bottom then left-to-right
    (``sort=True``), which is correct for single-column agreements. Multi-column
    layouts may interleave columns (documented limitation).
    """
    raw = page.get_text("rawdict", sort=True)
    blocks: list[TextBlock] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 1 = image block
            continue
        block_id = f"p{page_number:03d}_b{len(blocks):03d}"
        lines: list[TextLine] = []
        for line in block.get("lines", []):
            chars: list[tuple[str, tuple]] = []
            sizes: list[float] = []
            bold_flags: list[bool] = []
            for span in line.get("spans", []):
                span_chars = span.get("chars", [])
                if not span_chars:
                    continue
                chars.extend((ch["c"], tuple(ch["bbox"])) for ch in span_chars)
                if "".join(ch["c"] for ch in span_chars).strip():
                    sizes.append(span.get("size", 0.0))
                    bold_flags.append(
                        bool(span.get("flags", 0) & _BOLD_FLAG) or "bold" in span.get("font", "").lower()
                    )
            text = "".join(c for c, _ in chars)
            if not text.strip():
                continue
            lines.append(
                TextLine(
                    line_id=f"{block_id}_l{len(lines):03d}",
                    text=text,
                    bbox=_bbox(line["bbox"]),
                    words=_words_from_chars(chars, include_words),
                    font_size=round(max(sizes), 2) if sizes else None,
                    is_bold=all(bold_flags) if bold_flags else None,
                )
            )
        if not lines:
            continue
        blocks.append(
            TextBlock(
                block_id=block_id,
                document_id=document_id,
                page_number=page_number,
                text="\n".join(l.text for l in lines),
                bbox=_bbox(block["bbox"]),
                extraction_method=ExtractionMethod.NATIVE,
                lines=lines,
            )
        )
    return blocks


def _garbage_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = sum(1 for c in chars if c == "�" or 0xE000 <= ord(c) <= 0xF8FF)
    return bad / len(chars)


def assess_page(page: pymupdf.Page, blocks: list[TextBlock], config: ExtractionConfig) -> PageAssessment:
    """Decide whether a page is empty, has good native text, or needs OCR."""
    text = "\n".join(b.text for b in blocks)
    char_count = len(_WHITESPACE_RE.sub("", text))
    garbage = _garbage_ratio(text)
    has_images = bool(page.get_images(full=False))
    has_drawings = False
    if char_count < config.min_native_chars and not has_images:
        try:
            has_drawings = bool(page.get_drawings())
        except Exception:  # pragma: no cover - defensive
            has_drawings = False

    is_empty = char_count == 0 and not has_images and not has_drawings
    needs_ocr = False
    reason = None
    if is_empty:
        reason = "page is empty"
    elif char_count < config.min_native_chars and (has_images or has_drawings):
        needs_ocr = True
        reason = f"only {char_count} native characters; page contains images/graphics (likely scanned)"
    elif garbage > config.max_garbage_ratio:
        needs_ocr = True
        reason = f"{garbage:.0%} of native characters are unmappable glyphs"
    return PageAssessment(char_count, garbage, has_images, has_drawings, is_empty, needs_ocr, reason)


def build_page(
    page: pymupdf.Page,
    page_number: int,
    blocks: list[TextBlock],
    assessment: PageAssessment,
    method: ExtractionMethod,
    ocr_applied: bool = False,
) -> Page:
    """Assemble a :class:`Page` model."""
    rect = page.rect
    return Page(
        page_number=page_number,
        width=round(rect.width, 2),
        height=round(rect.height, 2),
        rotation=page.rotation,
        extraction_method=method,
        native_char_count=assessment.native_char_count,
        has_images=assessment.has_images,
        is_empty=assessment.is_empty,
        needs_ocr=assessment.needs_ocr,
        ocr_applied=ocr_applied,
        text="\n\n".join(b.text for b in blocks),
        blocks=blocks,
    )
