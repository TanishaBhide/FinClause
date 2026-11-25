"""Synthetic (fictional) document PDFs for tests and demos.

The content is fictional and contains no personal data. PDFs are generated with
PyMuPDF's built-in Helvetica fonts, which only cover Latin-1, so the rupee sign
(U+20B9) is written as "Rs." in PDFs; "₹" handling is covered by text-level tests.

Entry formats accepted by :func:`build_pdf` (one list of entries per page):

* ``"text"`` - a paragraph, wrapped automatically. A ``\\n`` inside the text forces a
  line break (used to reproduce hyphenation such as ``"pay-\\nment"``).
* ``("text", {"bold": True, "size": 12, "gap": 10, "indent": 90})`` - styled paragraph.
* ``("", {"table": [("Key", "Value"), ...]})`` - a two-column table with ruled rows.
"""

from __future__ import annotations

import io
import random
import textwrap
from pathlib import Path
from typing import Optional, Sequence, Union

import pymupdf

#: A line is either plain text or ``(text, options)``.
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

_PAGE_W, _PAGE_H = 595, 842  # A4 in points
_LEFT = 72


def _wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for part in text.split("\n"):
        out.extend(textwrap.wrap(part, width) or [""])
    return out


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

    Paragraphs are separated by a vertical gap so PyMuPDF sees them as separate blocks.
    """
    doc = pymupdf.open()
    total = len(pages)
    for page_index, lines in enumerate(pages, start=1):
        page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
        y = 72.0
        if header:
            page.insert_text((_LEFT, 36), header, fontname="helv", fontsize=8)
        for entry in lines:
            text, opts = (entry, {}) if isinstance(entry, str) else entry
            if "table" in opts:
                y = _draw_table(page, opts["table"], y, font_size, opts.get("key_width", 190))
                y += opts.get("gap", 10)
                continue
            size = opts.get("size", font_size)
            font = "hebo" if opts.get("bold") else "helv"
            indent = opts.get("indent", 90 if text.startswith("(") else _LEFT)
            for chunk in _wrap(text, wrap if indent == _LEFT else wrap - 3):
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


def _draw_table(page: pymupdf.Page, rows: Sequence[tuple[str, str]], y: float, size: float, key_width: float) -> float:
    """Two-column ruled table. Returns the new y position."""
    x_key, x_val, x_end = _LEFT, _LEFT + key_width, _PAGE_W - _LEFT
    line_h = size * 1.35
    page.draw_line((x_key - 4, y - size - 3), (x_end, y - size - 3), color=(0.55, 0.55, 0.55), width=0.6)
    for key, value in rows:
        key_lines = textwrap.wrap(key, 32) or [""]
        val_lines = textwrap.wrap(value, 58) or [""]
        for i, chunk in enumerate(key_lines):
            page.insert_text((x_key, y + i * line_h), chunk, fontname="hebo", fontsize=size)
        for i, chunk in enumerate(val_lines):
            page.insert_text((x_val, y + i * line_h), chunk, fontname="helv", fontsize=size)
        y += max(len(key_lines), len(val_lines)) * line_h + 6
        page.draw_line((x_key - 4, y - size - 3), (x_end, y - size - 3), color=(0.55, 0.55, 0.55), width=0.6)
    return y + 4


def _photocopy(img, rng: random.Random):
    """Make a clean grayscale render look like a photocopied/scanned page."""
    from PIL import Image, ImageDraw, ImageFilter, ImageOps

    angle = rng.choice([-1, 1]) * rng.uniform(0.6, 1.1)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    noise = Image.effect_noise(img.size, 40).filter(ImageFilter.GaussianBlur(0.6))
    img = Image.blend(img, noise, 0.10)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(int(w * h / 9000)):  # toner speckles
        x, y, r = rng.randrange(w), rng.randrange(h), rng.choice([1, 1, 1, 2])
        draw.ellipse((x, y, x + r, y + r), fill=rng.randrange(60, 150))
    edge = rng.randrange(int(w * 0.01), int(w * 0.025))  # dark scanner edge on one side
    draw.rectangle((0, 0, edge, h), fill=110)
    img = img.filter(ImageFilter.GaussianBlur(0.45))
    img = ImageOps.autocontrast(img, cutoff=1)
    return ImageOps.colorize(img, black=(28, 26, 24), white=(242, 236, 220))


def build_scanned_pdf(
    path: Optional[Union[str, Path]],
    pages: Sequence[Sequence[Line]],
    dpi: int = 150,
    style: str = "clean",
    seed: int = 7,
    header: Optional[str] = None,
    footer_page_numbers: bool = False,
) -> bytes:
    """Build an image-only PDF (no text layer) that simulates a scanned document.

    Args:
        style: ``"clean"`` (sharp grayscale render) or ``"photocopy"`` (skewed, noisy,
            speckled, tinted, JPEG-compressed - like a real photocopy scan).
    """
    from PIL import Image

    text_pdf = pymupdf.open(stream=build_pdf(None, pages, header=header, footer_page_numbers=footer_page_numbers),
                            filetype="pdf")
    out = pymupdf.open()
    rng = random.Random(seed)
    for src in text_pdf:
        page = out.new_page(width=src.rect.width, height=src.rect.height)
        if style == "photocopy":
            pix = src.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            img = _photocopy(Image.frombytes("L", (pix.width, pix.height), pix.samples), rng)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=55)
            page.insert_image(page.rect, stream=buf.getvalue())
        else:
            page.insert_image(page.rect, pixmap=src.get_pixmap(dpi=dpi))
    data = out.tobytes(deflate=True)
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
