"""Shared pytest fixtures: synthetic PDFs generated on the fly (no confidential data)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from finclause_extraction.config import ExtractionConfig
from finclause_extraction.samples import SYNTHETIC_AGREEMENT, build_pdf, build_scanned_pdf


@pytest.fixture
def agreement_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_agreement.pdf"
    build_pdf(path, SYNTHETIC_AGREEMENT, footer_page_numbers=True)
    return path


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "scanned.pdf"
    build_scanned_pdf(path, [SYNTHETIC_AGREEMENT[0][:4]])
    return path


@pytest.fixture
def pdf_with_empty_page(tmp_path: Path) -> Path:
    path = tmp_path / "with_blank.pdf"
    doc = pymupdf.open(stream=build_pdf(None, [["1. The Borrower shall pay interest at 12% per annum on the loan."]]), filetype="pdf")
    doc.new_page()  # blank second page
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def no_ocr_config() -> ExtractionConfig:
    return ExtractionConfig(use_ocr=False)
