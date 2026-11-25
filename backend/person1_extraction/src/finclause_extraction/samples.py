"""Synthetic loan-agreement PDFs for tests and demos.

The content is fictional and contains no personal data. PDFs are generated with
PyMuPDF's built-in Helvetica fonts, which only cover Latin-1, so the rupee sign
(U+20B9) is written as "Rs." here; "₹" handling is covered by text-level tests.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional, Sequence, Union

import pymupdf

#: A line is either plain text or ``(text, {"bold": True, "gap": 10})``.
Line = Union[str, tuple[str, dict]]

SYNTHETIC_AGREEMENT: list[list[Line]] = [
    [
        ("PERSONAL LOAN AGREEMENT", {"bold": True, "size": 14, "gap": 12}),
        "This Agreement is made on 15/01/2025 between Example Finance Ltd. (the Lender) and the "
        "Borrower named in the Schedule.",
        ("1. DEFINITIONS AND LOAN DETAILS", {"bold": True, "gap": 10}),
        "1.1 The Lender agrees to grant the Borrower a loan amount of Rs. 5,00,000 (Rupees Five Lakh only).",
        "1.2 The tenure of the loan shall be 36 months from the date of disbursement.",
        "1.3 The loan shall be repaid in 36 EMIs of Rs. 16,607 each.",
        ("2. INTEREST", {"bold": True, "gap": 10}),
        "2.1 Interest shall be charged at 12.5% per annum on the outstanding principal.",
        "2.2 The rate of interest is a floating interest rate linked to the external benchmark (repo rate) "
        "and may be revised by the Lender from time to time.",
        ("3. FEES AND CHARGES", {"bold": True, "gap": 10}),
        "3.1 A processing fee of 2% of the loan amount plus GST at 18% shall be deducted at disbursement.",
        "3.2 Late payment charges of Rs. 500 per instance shall apply, and penal interest of 2% per month "
        "shall be levied on the overdue amount.",
        "3.3 Foreclosure charges shall be levied as under:",
        "(a) 4% of the outstanding principal if the loan is foreclosed within 12 months;",
        "(b) 2% of the outstanding principal thereafter.",
    ],
    [
        "3.4 Prepayment charges shall be subject to applicable conditions as per the Schedule of Charges.",
        ("4. REPAYMENT", {"bold": True, "gap": 10}),
        "4.1 The EMI shall be payable on the fifth day of every month through NACH mandate. The first EMI "
        "is due on 05/03/2025.",
        "4.2 Any amount paid shall be appropriated first towards charges, then interest and then principal.",
    ],
]


def build_pdf(
    path: Optional[Union[str, Path]],
    pages: Sequence[Sequence[Line]],
    *,
    wrap: int = 95,
    font_size: float = 10.0,
    header: Optional[str] = None,
    footer_page_numbers: bool = False,
) -> bytes:
    """Build a text PDF from ``pages`` and return its bytes (also saved if ``path``).

    Each logical line is wrapped at ``wrap`` characters; paragraphs are separated
    by a vertical gap so PyMuPDF sees them as separate blocks.
    """
    doc = pymupdf.open()
    total = len(pages)
    for page_index, lines in enumerate(pages, start=1):
        page = doc.new_page(width=595, height=842)  # A4
        y = 72.0
        if header:
            page.insert_text((72, 36), header, fontname="helv", fontsize=8)
        for entry in lines:
            text, opts = (entry, {}) if isinstance(entry, str) else entry
            size = opts.get("size", font_size)
            font = "hebo" if opts.get("bold") else "helv"
            indent = 90 if text.startswith("(") else 72
            for chunk in textwrap.wrap(text, wrap) or [""]:
                page.insert_text((indent, y), chunk, fontname=font, fontsize=size)
                y += size * 1.35
            y += opts.get("gap", 8)
        if footer_page_numbers:
            page.insert_text((280, 815), f"Page {page_index} of {total}", fontname="helv", fontsize=8)
    data = doc.tobytes()
    doc.close()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def build_scanned_pdf(path: Optional[Union[str, Path]], pages: Sequence[Sequence[Line]], dpi: int = 150) -> bytes:
    """Build an image-only PDF (no text layer) that simulates a scanned document."""
    text_pdf = pymupdf.open(stream=build_pdf(None, pages), filetype="pdf")
    out = pymupdf.open()
    for src in text_pdf:
        pix = src.get_pixmap(dpi=dpi)
        page = out.new_page(width=src.rect.width, height=src.rect.height)
        page.insert_image(page.rect, pixmap=pix)
    data = out.tobytes()
    out.close()
    text_pdf.close()
    if path is not None:
        Path(path).write_bytes(data)
    return data


def sample_agreement_text() -> str:
    """The synthetic agreement as plain text (paragraphs separated by blank lines)."""
    paragraphs = []
    for page in SYNTHETIC_AGREEMENT:
        for entry in page:
            paragraphs.append(entry if isinstance(entry, str) else entry[0])
    return "\n\n".join(paragraphs)


def wrap_lines(text: str, width: int = 95) -> list[str]:
    return textwrap.wrap(text, width)
