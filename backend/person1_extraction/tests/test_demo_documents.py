"""End-to-end tests on the five curated demo documents (digital, and scanned with real OCR)."""

from __future__ import annotations

import pytest

from finclause_extraction import ExtractionConfig, analyze_pdf_bytes
from finclause_extraction.config import ExtractionConfig as _Cfg
from finclause_extraction.demo_documents import DEMO_DOCUMENTS, list_demo_documents
from finclause_extraction.pdf import ocr as ocr_mod

HAS_TESSERACT = ocr_mod.resolve_tesseract_cmd(_Cfg()) is not None

#: label -> normalized values that must be found in the digital version of each document
EXPECTED = {
    "northwind_bank_personal_loan_agreement": {
        "LOAN_AMOUNT": {"650000 INR"}, "EMI": {"17230 INR"}, "INTEREST_RATE": {"11.75% per annum"},
        "PROCESSING_FEE": {"1.5%", "3000 INR"}, "PENALTY": {"2% per month", "590 INR"},
        "FORECLOSURE_CHARGE": {"4%", "2%"}, "GRACE_PERIOD": {"5 days"}, "DATE": {"2025-02-12", "2025-04-07"},
    },
    "sahyadri_finance_key_fact_statement": {
        "LOAN_AMOUNT": {"300000 INR"}, "EMI": {"10442 INR"}, "INTEREST_RATE": {"15.5% per annum", "17.9%"},
        "PROCESSING_FEE": {"2.5%"}, "PREMIUM": {"4200 INR"}, "PENALTY": {"2% per month", "500 INR"},
        "FORECLOSURE_CHARGE": {"5%"},
    },
    "kaveri_cooperative_bank_sanction_letter": {
        "LOAN_AMOUNT": {"200000 INR"}, "EMI": {"9525 INR"}, "INTEREST_RATE": {"13.25% per annum"},
        "PROCESSING_FEE": {"1%"}, "PENALTY": {"24% per annum"},
    },
    "suraksha_health_insurance_policy_terms": {
        "SUM_INSURED": {"500000 INR"}, "PREMIUM": {"18450 INR"}, "CO_PAYMENT": {"20%"},
        "COVERAGE_LIMIT": {"1% per day", "2% per day"}, "WAITING_PERIOD": {"30 days", "24 months", "36 months"},
        "GRACE_PERIOD": {"30 days"},
    },
    "shopkart_no_cost_emi_terms": {
        "INTEREST_RATE": {"15% per annum", "13%", "16% per annum"}, "TAX": {"18%"}, "PROCESSING_FEE": {"199 INR"},
        "FORECLOSURE_CHARGE": {"3%"}, "PENALTY": {"100 INR", "1300 INR"}, "TENURE": {"3, 6 or 9 months", "12 to 24 months"},
    },
}
EXPECTED_CONDITIONS = {
    "northwind_bank_personal_loan_agreement": {"DEFAULT_CONDITION", "DISCRETIONARY_CHANGE_CONDITION", "CANCELLATION_CONDITION", "REPAYMENT_CONDITION"},
    "sahyadri_finance_key_fact_statement": {"VARIABLE_INTEREST_CONDITION", "DEFAULT_CONDITION", "CANCELLATION_CONDITION"},
    "kaveri_cooperative_bank_sanction_letter": {"DEFAULT_CONDITION", "SECURITY_CONDITION", "REPAYMENT_CONDITION"},
    "suraksha_health_insurance_policy_terms": {"EXCLUSION", "CANCELLATION_CONDITION"},
    "shopkart_no_cost_emi_terms": {"CANCELLATION_CONDITION", "DISCRETIONARY_CHANGE_CONDITION", "DEFAULT_CONDITION"},
}
DOC_TYPES = {
    "northwind_bank_personal_loan_agreement": "loan_agreement", "sahyadri_finance_key_fact_statement": "key_fact_statement",
    "kaveri_cooperative_bank_sanction_letter": "sanction_letter", "suraksha_health_insurance_policy_terms": "insurance_policy",
    "shopkart_no_cost_emi_terms": "emi_terms",
}


def values_by_label(result) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for e in result.entities:
        if e.normalized_value is not None:
            out.setdefault(e.label, set()).add(e.normalized_value)
    return out


@pytest.fixture(scope="module")
def digital_results():
    cfg = ExtractionConfig(include_words=False)
    return {doc_id: analyze_pdf_bytes(doc.build(scanned=False), doc.filename, config=cfg) for doc_id, doc in DEMO_DOCUMENTS.items()}


def test_catalogue() -> None:
    cat = list_demo_documents()
    assert len(cat) == 5 and {d["category"] for d in cat} == {"loan", "insurance", "emi"}
    assert all("fictional" in d["institution"] for d in cat)


@pytest.mark.parametrize("doc_id", list(DEMO_DOCUMENTS))
def test_digital_demo_extraction(digital_results, doc_id: str) -> None:
    r = digital_results[doc_id]
    assert r.extraction_status == "success" and not r.metadata.ocr_used
    assert r.metadata.document_type == DOC_TYPES[doc_id]
    found = values_by_label(r)
    for label, expected in EXPECTED[doc_id].items():
        assert expected <= found.get(label, set()), (label, found.get(label))
    assert EXPECTED_CONDITIONS[doc_id] <= {e.label for e in r.entities}
    assert len(r.key_points) >= 8
    for e in r.entities:  # traceability
        c = r.get_clause(e.clause_id)
        assert c.original_text[e.start_offset:e.end_offset] == e.surface_text and e.bboxes


def test_headers_and_cross_page_clause(digital_results) -> None:
    r = digital_results["northwind_bank_personal_loan_agreement"]
    hf = next(w for w in r.warnings if w.code == "HEADER_FOOTER_EXCLUDED")
    assert len(hf.locations) == 4 and all(l.bbox for l in hf.locations)
    c34 = next(c for c in r.clauses if c.numbering_path == "3.4")
    assert c34.page_numbers == [1, 2] and "SPECIMEN" not in c34.original_text
    c22 = next(c for c in r.clauses if c.numbering_path == "2.2")
    assert "pay-\nment" in c22.original_text and "payment." in c22.cleaned_text


def test_kfs_table_rows_are_separate_clauses(digital_results) -> None:
    r = digital_results["sahyadri_finance_key_fact_statement"]
    rows = [c.cleaned_text for c in r.clauses if "\t" in c.original_text]
    assert "Loan amount sanctioned Rs. 3,00,000" in rows and "Loan tenure 36 months" in rows
    assert len(rows) >= 12


def test_warnings_have_locations(digital_results) -> None:
    r = digital_results["northwind_bank_personal_loan_agreement"]
    amb = next(w for w in r.warnings if w.code == "AMBIGUOUS_DATE")
    assert {l.text for l in amb.locations} == {"12/02/2025", "07/04/2025"}
    assert all(l.entity_id and l.bbox for l in amb.locations)


def test_scanned_demo_is_image_only() -> None:
    import pymupdf

    doc = pymupdf.open(stream=DEMO_DOCUMENTS["kaveri_cooperative_bank_sanction_letter"].build(scanned=True), filetype="pdf")
    assert doc[0].get_text().strip() == "" and doc[0].get_images()


@pytest.mark.skipif(not HAS_TESSERACT, reason="Tesseract OCR binary not installed")
@pytest.mark.parametrize("doc_id", ["kaveri_cooperative_bank_sanction_letter", "northwind_bank_personal_loan_agreement"])
def test_scanned_demo_with_real_ocr_matches_digital(digital_results, doc_id: str) -> None:  # pragma: no cover
    doc = DEMO_DOCUMENTS[doc_id]
    r = analyze_pdf_bytes(doc.build(scanned=True), doc.filename, config=ExtractionConfig(include_words=False))
    assert r.extraction_status == "success" and r.metadata.ocr_used
    page = r.pages[0]
    assert page.ocr_applied and page.ocr_mean_confidence > 85 and abs(page.ocr_deskew_angle) >= 0.5
    assert any(s.startswith("deskew") for s in page.ocr_preprocessing)
    found = values_by_label(r)
    for label, expected in EXPECTED[doc_id].items():
        assert expected <= found.get(label, set()), (label, found.get(label))
