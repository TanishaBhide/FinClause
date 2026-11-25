"""End-to-end pipeline tests: processing, JSON, validation, errors, traceability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finclause_extraction import (
    ExtractionConfig,
    ExtractionResult,
    PDFNotFoundError,
    analyze_pdf,
    analyze_pdf_bytes,
    analyze_text,
)
from finclause_extraction.pdf import ocr as ocr_mod
from finclause_extraction.samples import SYNTHETIC_AGREEMENT, build_pdf


def assert_traceable(result: ExtractionResult) -> None:
    """Every entity must point at its exact clause text, page and source box."""
    clauses = {c.clause_id: c for c in result.clauses}
    pages = {p.page_number: p for p in result.pages}
    owner: dict[str, str] = {}
    for c in result.clauses:
        for eid in c.entity_ids:
            assert eid not in owner, f"{eid} listed in two clauses"
            owner[eid] = c.clause_id
    assert set(owner) == {e.entity_id for e in result.entities}
    for e in result.entities:
        clause = clauses[e.clause_id]
        assert owner[e.entity_id] == clause.clause_id
        assert clause.original_text[e.start_offset:e.end_offset] == e.surface_text
        assert clause.cleaned_text[e.cleaned_start_offset:e.cleaned_end_offset] == e.cleaned_surface_text
        assert e.page_number in clause.page_numbers
        assert e.bboxes, f"{e.entity_id} has no bounding box"
        page = pages[e.page_number]
        for b in e.bboxes:
            assert 0 <= b.x0 < b.x1 <= page.width + 1 and 0 <= b.y0 < b.y1 <= page.height + 1
    for c in result.clauses:
        for loc in c.source_locations:
            assert loc.page_number in c.page_numbers
            block = next(b for b in pages[loc.page_number].blocks if b.block_id == loc.block_id)
            line_texts = {l.line_id: l.text for l in block.lines}
            for lid in loc.line_ids:
                assert line_texts[lid] in c.original_text
        for s in c.financial_sentences:
            assert c.cleaned_text[s.cleaned_start_offset:s.cleaned_end_offset] == s.text
            assert set(s.entity_ids) <= set(c.entity_ids)


def test_end_to_end_digital_pdf(agreement_pdf: Path) -> None:
    result = analyze_pdf(agreement_pdf, use_ocr=False)
    assert result.extraction_status == "success"
    assert result.total_pages == 2 and result.filename == agreement_pdf.name
    assert result.document.sha256 and result.document_id.startswith("doc_")
    labels = {(e.label, e.normalized_value) for e in result.entities}
    for expected in [
        ("LOAN_AMOUNT", "500000 INR"), ("TENURE", "36 months"), ("EMI", "16607 INR"),
        ("INTEREST_RATE", "12.5% per annum"), ("PROCESSING_FEE", "2%"), ("TAX", "18%"),
        ("PENALTY", "500 INR"), ("PENALTY", "2% per month"), ("FORECLOSURE_CHARGE", "4%"),
        ("DATE", "2025-01-15"), ("DATE", "2025-03-05"),
    ]:
        assert expected in labels, expected
    present = {e.label for e in result.entities}
    assert {"VARIABLE_INTEREST_CONDITION", "REPAYMENT_CONDITION", "PREPAYMENT_CHARGE"} <= present
    assert result.metadata.native_pages == [1, 2] and not result.metadata.ocr_used
    assert_traceable(result)


def test_clause_tree_and_cross_reference(agreement_pdf: Path) -> None:
    result = analyze_pdf(agreement_pdf, use_ocr=False)
    by_path = {c.numbering_path: c for c in result.clauses if c.numbering_path}
    assert by_path["3.3(a)"].parent_clause_id == by_path["3.3"].clause_id
    assert by_path["3.4"].page_numbers == [2]
    # "(b) 2% of the outstanding principal thereafter." has no trigger of its own:
    # its meaning comes from the parent lead-in "3.3 Foreclosure charges shall be levied as under:"
    fc = next(e for e in result.entities if e.clause_id == by_path["3.3(b)"].clause_id)
    assert fc.label == "FORECLOSURE_CHARGE" and fc.evidence.trigger_source == "parent_clause"
    # no page footers inside clauses
    assert not any("Page 1 of 2" in c.original_text for c in result.clauses)
    assert {a.reason for a in result.metadata.excluded_artifact_lines} == {"page_number"}


def test_json_serialization_roundtrip(agreement_pdf: Path) -> None:
    result = analyze_pdf(agreement_pdf, use_ocr=False)
    payload = result.to_json()
    data = json.loads(payload)
    assert data["extraction_status"] == "success"
    assert isinstance(data["entities"][0]["bboxes"][0]["x0"], float)
    assert "entities" not in data["clauses"][0]  # entities stored once, referenced by ID
    again = ExtractionResult.model_validate_json(payload)
    assert again == result
    json.dumps(result.model_dump(mode="json"))  # FastAPI-style dump


def test_helpers_resolve_entities(agreement_pdf: Path) -> None:
    result = analyze_pdf(agreement_pdf, use_ocr=False)
    e = result.entities_by_label("EMI")[0]
    clause = result.get_clause(e.clause_id)
    assert e in result.entities_for_clause(clause.clause_id)
    assert result.get_entity(e.entity_id) == e


def test_document_id_is_deterministic(agreement_pdf: Path) -> None:
    a = analyze_pdf(agreement_pdf, use_ocr=False)
    b = analyze_pdf_bytes(agreement_pdf.read_bytes(), agreement_pdf.name, use_ocr=False)
    assert a.document_id == b.document_id
    assert [c.clause_id for c in a.clauses] == [c.clause_id for c in b.clauses]
    assert analyze_pdf(agreement_pdf, use_ocr=False, document_id="doc_custom").document_id == "doc_custom"


def test_empty_page_warning(pdf_with_empty_page: Path) -> None:
    result = analyze_pdf(pdf_with_empty_page, use_ocr=False)
    assert result.extraction_status == "success"
    assert result.metadata.empty_pages == [2]
    assert any(w.code == "EMPTY_PAGE" and w.page_number == 2 for w in result.warnings)
    assert result.pages[1].is_empty


def test_scanned_pdf_ocr_disabled(scanned_pdf: Path) -> None:
    result = analyze_pdf(scanned_pdf, use_ocr=False)
    assert result.extraction_status == "partial"
    assert result.metadata.unextracted_pages == [1]
    assert any(w.code == "OCR_DISABLED" for w in result.warnings)
    assert result.pages[0].needs_ocr and not result.pages[0].ocr_applied


def test_scanned_pdf_ocr_unavailable(scanned_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_mod, "default_install_locations", lambda: [])
    monkeypatch.delenv("FINCLAUSE_TESSERACT_CMD", raising=False)
    result = analyze_pdf(scanned_pdf, use_ocr=True)
    assert result.extraction_status == "partial"
    codes = [w.code for w in result.warnings]
    assert "OCR_UNAVAILABLE" in codes and "PAGE_NOT_EXTRACTED" in codes
    msg = next(w.message for w in result.warnings if w.code == "OCR_UNAVAILABLE")
    assert "FINCLAUSE_TESSERACT_CMD" in msg


def test_scanned_pdf_with_mocked_ocr(scanned_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pytesseract

    words = ["2.1", "Interest", "shall", "be", "charged", "at", "12%", "per", "annum."]
    n = len(words)
    fake = {
        "block_num": [1] * n, "par_num": [1] * n, "line_num": [1] * n, "word_num": list(range(1, n + 1)),
        "left": [300 + 150 * i for i in range(n)], "top": [300] * n, "width": [140] * n, "height": [40] * n,
        "conf": ["90"] * n, "text": words,
    }
    monkeypatch.setattr("finclause_extraction.pipeline.check_tesseract", lambda cfg: "5.3.0 (mocked)")
    monkeypatch.setattr(ocr_mod, "resolve_tesseract_cmd", lambda cfg: "tesseract")
    monkeypatch.setattr(pytesseract, "image_to_data", lambda image, lang, output_type: fake)
    result = analyze_pdf(scanned_pdf, use_ocr=True, config=ExtractionConfig(ocr_preprocess=False))
    assert result.extraction_status == "success"
    assert result.metadata.ocr_used and result.metadata.ocr_pages == [1]
    assert result.pages[0].extraction_method == "ocr" and result.pages[0].ocr_applied
    rate = result.entities_by_label("INTEREST_RATE")[0]
    assert rate.surface_text == "12% per annum"
    assert rate.bboxes[0].x0 == pytest.approx((300 + 150 * 6) * 72 / 300)
    assert_traceable(result)


def test_missing_file_returns_failed_result(tmp_path: Path) -> None:
    result = analyze_pdf(tmp_path / "missing.pdf")
    assert result.extraction_status == "failed"
    assert result.errors[0].code == "FILE_NOT_FOUND"
    assert result.clauses == [] and result.entities == []
    with pytest.raises(PDFNotFoundError):
        analyze_pdf(tmp_path / "missing.pdf", raise_on_error=True)


def test_invalid_pdf_returns_failed_result(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 this is broken")
    result = analyze_pdf(bad)
    assert result.extraction_status == "failed"
    assert result.errors[0].code in {"INVALID_PDF", "EMPTY_DOCUMENT"}
    json.loads(result.to_json())


def test_encrypted_pdf_needs_password(tmp_path: Path) -> None:
    import pymupdf

    src = pymupdf.open(stream=build_pdf(None, [["1. Interest at 12% per annum."]]), filetype="pdf")
    path = tmp_path / "locked.pdf"
    path.write_bytes(src.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="own"))
    assert analyze_pdf(path, use_ocr=False).errors[0].code == "ENCRYPTED_PDF"
    ok = analyze_pdf(path, use_ocr=False, password="pw")
    assert ok.extraction_status == "success" and ok.document.is_encrypted
    assert ok.entities_by_label("INTEREST_RATE")[0].normalized_value == "12% per annum"


def test_words_can_be_omitted_but_entities_keep_boxes(agreement_pdf: Path) -> None:
    result = analyze_pdf(agreement_pdf, use_ocr=False, config=ExtractionConfig(include_words=False))
    assert all(not l.words for p in result.pages for b in p.blocks for l in b.lines)
    assert all(e.bboxes for e in result.entities)


def test_analyze_text() -> None:
    result = analyze_text("1. FEES\n1.1 Processing fee of 2% of the loan amount.\n1.2 Penalty of Rs. 500.")
    assert result.extraction_status == "success"
    assert [e.label for e in result.entities] == ["PROCESSING_FEE", "PENALTY"]
    assert any(w.code == "SYNTHETIC_COORDINATES" for w in result.warnings)
    assert_traceable(result)


def test_sample_text_matches_pdf_extraction(agreement_pdf: Path) -> None:
    from finclause_extraction.samples import sample_agreement_text

    pdf_labels = sorted(e.label for e in analyze_pdf(agreement_pdf, use_ocr=False).entities)
    text_labels = sorted(e.label for e in analyze_text(sample_agreement_text()).entities)
    assert pdf_labels == text_labels


@pytest.mark.skipif(ocr_mod.resolve_tesseract_cmd(ExtractionConfig()) is None, reason="Tesseract OCR binary not installed")
def test_scanned_pdf_real_ocr_end_to_end(scanned_pdf: Path) -> None:  # pragma: no cover - environment dependent
    result = analyze_pdf(scanned_pdf, use_ocr=True)  # default path: with OCR preprocessing
    assert result.extraction_status == "success"
    assert result.metadata.ocr_used and result.metadata.ocr_pages == [1]
    assert result.metadata.tesseract_version
    assert result.pages[0].extraction_method == "ocr"
    found = {(e.label, e.normalized_value) for e in result.entities}
    assert ("LOAN_AMOUNT", "500000 INR") in found
    assert ("DATE", "2025-01-15") in found
    assert_traceable(result)
