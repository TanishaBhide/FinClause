"""Tests for native PDF loading, validation and page extraction."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from finclause_extraction.config import ExtractionConfig
from finclause_extraction.pdf.extractor import (
    EmptyDocumentError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFNotFoundError,
    UnsupportedFormatError,
    assess_page,
    extract_native_blocks,
    open_pdf,
    read_pdf_bytes,
)
from finclause_extraction.samples import build_pdf

CFG = ExtractionConfig()


def _open(path: Path) -> pymupdf.Document:
    data, name = read_pdf_bytes(path)
    return open_pdf(data, name)


def test_valid_digital_pdf_blocks_lines_words(agreement_pdf: Path) -> None:
    doc = _open(agreement_pdf)
    blocks = extract_native_blocks(doc[0], 1, "doc_test")
    assert blocks, "expected text blocks"
    all_text = "\n".join(b.text for b in blocks)
    assert "1.1 The Lender agrees to grant the Borrower a loan amount of Rs. 5,00,000" in all_text
    block = blocks[0]
    assert block.page_number == 1 and block.document_id == "doc_test"
    assert block.extraction_method == "native"
    assert block.block_id == "p001_b000"
    line = block.lines[0]
    assert line.words, "word-level boxes expected"
    for w in line.words:
        assert line.text[w.char_start:w.char_end] == w.text
        assert w.bbox.x1 > w.bbox.x0 and w.bbox.y1 > w.bbox.y0


def test_reading_order_is_top_to_bottom(agreement_pdf: Path) -> None:
    doc = _open(agreement_pdf)
    blocks = extract_native_blocks(doc[0], 1, "d")
    tops = [b.bbox.y0 for b in blocks]
    assert tops == sorted(tops)
    text = "\n".join(b.text for b in blocks)
    assert text.index("1.1 The Lender") < text.index("2.1 Interest") < text.index("3.1 A processing fee")


def test_bold_heading_detected(agreement_pdf: Path) -> None:
    doc = _open(agreement_pdf)
    lines = [l for b in extract_native_blocks(doc[0], 1, "d") for l in b.lines]
    heading = next(l for l in lines if l.text.startswith("2. INTEREST"))
    body = next(l for l in lines if l.text.startswith("2.1"))
    assert heading.is_bold is True
    assert body.is_bold is False


def test_multi_page_pdf_preserves_page_numbers(agreement_pdf: Path) -> None:
    doc = _open(agreement_pdf)
    assert doc.page_count == 2
    blocks_p2 = extract_native_blocks(doc[1], 2, "d")
    assert all(b.page_number == 2 for b in blocks_p2)
    assert any("Prepayment charges" in b.text for b in blocks_p2)


def test_empty_page_detected(pdf_with_empty_page: Path) -> None:
    doc = _open(pdf_with_empty_page)
    blank = doc[1]
    blocks = extract_native_blocks(blank, 2, "d")
    assessment = assess_page(blank, blocks, CFG)
    assert blocks == []
    assert assessment.is_empty and not assessment.needs_ocr
    assert not assess_page(doc[0], extract_native_blocks(doc[0], 1, "d"), CFG).is_empty


def test_scanned_page_flagged_for_ocr(scanned_pdf: Path) -> None:
    doc = _open(scanned_pdf)
    blocks = extract_native_blocks(doc[0], 1, "d")
    assessment = assess_page(doc[0], blocks, CFG)
    assert assessment.native_char_count == 0
    assert assessment.has_images
    assert assessment.needs_ocr and not assessment.is_empty


def test_good_native_text_not_flagged_for_ocr(agreement_pdf: Path) -> None:
    doc = _open(agreement_pdf)
    assessment = assess_page(doc[0], extract_native_blocks(doc[0], 1, "d"), CFG)
    assert not assessment.needs_ocr


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(PDFNotFoundError):
        read_pdf_bytes(tmp_path / "does_not_exist.pdf")


def test_directory_path_raises(tmp_path: Path) -> None:
    with pytest.raises(PDFNotFoundError):
        read_pdf_bytes(tmp_path)


def test_invalid_pdf_raises() -> None:
    with pytest.raises((InvalidPDFError, UnsupportedFormatError)):
        open_pdf(b"this is definitely not a pdf file", "bad.pdf")


def test_corrupted_pdf_raises() -> None:
    with pytest.raises(InvalidPDFError):
        open_pdf(b"%PDF-1.7\n%%%% garbage \x00\x01\x02 no objects here", "corrupt.pdf")


def test_empty_bytes_raise() -> None:
    with pytest.raises(InvalidPDFError):
        open_pdf(b"", "empty.pdf")


def test_non_pdf_format_rejected() -> None:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    png = pix.tobytes("png")
    with pytest.raises(UnsupportedFormatError):
        open_pdf(png, "image.png")


def test_encrypted_pdf(tmp_path: Path) -> None:
    src = pymupdf.open(stream=build_pdf(None, [["1. Interest at 12% per annum."]]), filetype="pdf")
    data = src.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    with pytest.raises(EncryptedPDFError):
        open_pdf(data, "locked.pdf")
    with pytest.raises(EncryptedPDFError):
        open_pdf(data, "locked.pdf", password="wrong")
    doc = open_pdf(data, "locked.pdf", password="secret")
    assert "12% per annum" in doc[0].get_text()


def test_zero_page_pdf_rejected() -> None:
    # MuPDF refuses to serialise a zero-page document, so craft a minimal one by hand.
    data = (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF"
    )
    with pytest.raises((EmptyDocumentError, InvalidPDFError)):
        open_pdf(data, "zero.pdf")
