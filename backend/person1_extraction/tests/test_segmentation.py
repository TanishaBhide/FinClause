"""Tests for clause segmentation and sentence splitting."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from finclause_extraction.config import ExtractionConfig
from finclause_extraction.pdf.extractor import extract_native_blocks, open_pdf, read_pdf_bytes
from finclause_extraction.preprocessing.segmenter import (
    is_title_like,
    plausible_successor,
    segment_pages,
    split_sentences,
)
from finclause_extraction.samples import build_pdf
from finclause_extraction.text_input import pages_from_text


def _segment(text: str):
    return [s.clause for s in segment_pages(pages_from_text(text)).clauses]


def _by_number(clauses, number):
    return next(c for c in clauses if c.numbering_path == number)


def test_numbered_and_nested_clauses() -> None:
    text = (
        "1. DEFINITIONS\n"
        "1.1 Loan means the amount sanctioned.\n"
        "1.2 EMI means equated monthly instalment.\n"
        "1.2.1 EMI includes principal and interest.\n"
        "2. INTEREST\n"
        "2.1 Interest shall be charged at 12% per annum."
    )
    clauses = _segment(text)
    assert [c.numbering_path for c in clauses] == ["1", "1.1", "1.2", "1.2.1", "2", "2.1"]
    c1, c11, c12, c121, c2, c21 = clauses
    assert c1.clause_type == "clause" and c11.clause_type == "subclause"
    assert c11.parent_clause_id == c1.clause_id
    assert c121.parent_clause_id == c12.clause_id
    assert c21.parent_clause_id == c2.clause_id
    assert c1.parent_clause_id is None and c2.parent_clause_id is None
    assert c1.child_clause_ids == [c11.clause_id, c12.clause_id]
    assert c121.level == 2


def test_headings_attached_to_numbered_clauses() -> None:
    clauses = _segment("3. FEES AND CHARGES\n3.1 A processing fee of 2% applies.\n3.2 Late fee of Rs. 500.")
    c3 = _by_number(clauses, "3")
    assert c3.heading == "FEES AND CHARGES"
    assert _by_number(clauses, "3.1").heading == "FEES AND CHARGES"
    assert _by_number(clauses, "3.1").heading_clause_id == c3.clause_id


def test_section_heading_with_dash_and_clause_keyword() -> None:
    text = "Section 7 - Fees and Charges\n\nThe following charges apply to the loan.\n\nClause 5. The Borrower shall pay the EMI on time."
    clauses = _segment(text)
    sec = clauses[0]
    assert sec.clause_type == "section" and sec.heading == "Fees and Charges" and sec.clause_number == "Section 7"
    para = clauses[1]
    assert para.clause_type == "paragraph" and para.parent_clause_id == sec.clause_id
    assert para.heading == "Fees and Charges"
    cl5 = clauses[2]
    assert cl5.clause_number == "Clause 5." or cl5.clause_number.startswith("Clause 5")
    assert cl5.numbering_path == "5"


def test_clause_cross_reference_is_not_a_heading() -> None:
    clauses = _segment("4.1 Charges apply.\n\nClause 5 of this Agreement shall apply to all prepayments.")
    assert len(clauses) == 2
    assert clauses[1].clause_type == "paragraph"


def test_alpha_and_roman_subclauses() -> None:
    text = (
        "2. The Borrower shall:\n"
        "  (a) pay all charges;\n"
        "      (i) on the due date; and\n"
        "      (ii) without deduction;\n"
        "  (b) maintain insurance.\n"
        "2(c) furnish documents."
    )
    clauses = _segment(text)
    c2 = clauses[0]
    a, i, ii, b, c = clauses[1:]
    assert a.numbering_path == "2(a)" and a.parent_clause_id == c2.clause_id
    assert i.numbering_path == "2(a)(i)" and i.parent_clause_id == a.clause_id
    assert ii.parent_clause_id == a.clause_id
    assert b.numbering_path == "2(b)" and b.parent_clause_id == c2.clause_id
    assert c.clause_number == "2(c)" and c.parent_clause_id == c2.clause_id


def test_letter_i_after_h_is_alpha_not_roman() -> None:
    text = "1. Items:\n" + "\n".join(f"({ch}) item {ch};" for ch in "abcdefghij")
    clauses = _segment(text)
    paths = [c.numbering_path for c in clauses[1:]]
    assert paths == [f"1({ch})" for ch in "abcdefghij"]
    assert all(c.parent_clause_id == clauses[0].clause_id for c in clauses[1:])


def test_bullets() -> None:
    clauses = _segment("Charges include:\n• processing fee of 1%;\n• stamp duty as applicable.")
    bullets = [c for c in clauses if c.clause_type == "bullet"]
    assert len(bullets) == 2


def test_decimal_values_do_not_split_clauses() -> None:
    text = (
        "1.1 The Lender shall disburse a loan of Rs.\n"
        "2.5 lakh to the Borrower at an interest rate of\n"
        "10.75 p.a. and a processing fee of\n"
        "1.5 percent of the loan amount.\n"
        "1.2 The tenure is 24 months."
    )
    clauses = _segment(text)
    assert [c.numbering_path for c in clauses] == ["1.1", "1.2"]
    assert "Rs. 2.5 lakh" in clauses[0].cleaned_text
    assert "1.5 percent" in clauses[0].cleaned_text


def test_wrapped_number_that_is_not_plausible_is_continuation() -> None:
    text = "3.1 Foreclosure charges shall be levied as per\n7.4 of the Schedule of Charges.\n3.2 Next."
    clauses = _segment(text)
    assert [c.numbering_path for c in clauses] == ["3.1", "3.2"]


def test_multiple_paragraphs() -> None:
    text = "The Borrower agrees to the terms.\n\nThe Lender agrees to disburse.\n\nBoth parties sign below."
    clauses = _segment(text)
    assert len(clauses) == 3 and all(c.clause_type == "paragraph" for c in clauses)


def test_paragraph_continuing_unfinished_sentence_is_merged() -> None:
    text = "1.1 The Borrower shall pay interest on the\n\noutstanding amount every month."
    clauses = _segment(text)
    assert len(clauses) == 1
    assert clauses[0].cleaned_text == "1.1 The Borrower shall pay interest on the\noutstanding amount every month."


def test_cross_page_clause() -> None:
    text = "1.1 The Borrower shall pay a foreclosure charge of 4% of the\fprincipal outstanding at the time of foreclosure.\n1.2 Next clause."
    clauses = _segment(text)
    first = clauses[0]
    assert first.page_numbers == [1, 2]
    assert "of the\n\nprincipal outstanding" in first.original_text
    assert len(first.source_locations) == 2
    assert [s.page_number for s in first.source_locations] == [1, 2]
    assert clauses[1].numbering_path == "1.2"


def test_source_locations_map_to_original_text() -> None:
    text = "2.1 Interest shall be charged\nat 12% per annum.\n\nwhich is payable monthly."
    seg = segment_pages(pages_from_text(text)).clauses[0]
    clause = seg.clause
    for loc in clause.source_locations:
        assert loc.bbox is not None
        assert clause.original_text[loc.start_offset:loc.end_offset].startswith(("2.1", "which"))
    for span in seg.line_spans:
        assert clause.original_text[span.start:span.end] == span.line.text


def test_clause_ids_stable_and_unique() -> None:
    text = "1. A.\n1.1 B.\n2. C."
    a = _segment(text)
    b = _segment(text)
    assert [c.clause_id for c in a] == [c.clause_id for c in b] == ["clause_0001", "clause_0002", "clause_0003"]


def test_pdf_segmentation_with_headers_footers(tmp_path: Path) -> None:
    pages = [
        ["1. LOAN", "1.1 The loan amount is Rs. 1,00,000.", "1.2 The loan shall be repaid in 12 months and the"],
        ["borrower shall pay interest at 11% per annum.", "1.3 Late payment charge of Rs. 500 applies."],
    ]
    data = build_pdf(None, pages, header="Example Finance - Personal Loan Agreement", footer_page_numbers=True)
    doc = open_pdf(data, "x.pdf")
    from finclause_extraction.schemas import Page
    from finclause_extraction.pdf.extractor import assess_page, build_page

    page_models = []
    for i, page in enumerate(doc, start=1):
        blocks = extract_native_blocks(page, i, "d")
        page_models.append(build_page(page, i, blocks, assess_page(page, blocks, ExtractionConfig()), "native"))
    result = segment_pages(page_models)
    reasons = {a.reason for a in result.artifacts}
    assert reasons == {"page_number", "repeated_header_footer"}
    clauses = [s.clause for s in result.clauses]
    assert [c.numbering_path for c in clauses] == ["1", "1.1", "1.2", "1.3"]
    c12 = clauses[2]
    assert c12.page_numbers == [1, 2]
    assert "Page 1 of 2" not in c12.original_text and "Example Finance" not in c12.original_text


def test_sentence_splitting_respects_abbreviations_and_decimals() -> None:
    text = "Interest is 12.5% p.a. on Rs. 5,000. The fee is Rs. 500 viz. a fixed charge. Example Pvt. Ltd. agrees."
    spans = split_sentences(text)
    sentences = [text[s:e] for s, e in spans]
    assert sentences == [
        "Interest is 12.5% p.a. on Rs. 5,000.",
        "The fee is Rs. 500 viz. a fixed charge.",
        "Example Pvt. Ltd. agrees.",
    ]


def test_sentence_splitting_keeps_leading_clause_number() -> None:
    assert split_sentences("1. DEFINITIONS apply here. Second sentence.") == [(0, 26), (27, 43)]


def test_title_heuristics() -> None:
    assert is_title_like("FEES AND CHARGES")
    assert is_title_like("Interest Rate")
    assert not is_title_like("The tenure of the loan shall be 36 months.")
    assert not is_title_like("Interest Rate applicable to the", next_text="loan amount")
    assert plausible_successor((1, 2), (1, 3)) and plausible_successor((1, 2), (2,)) and plausible_successor((1,), (1, 1))
    assert not plausible_successor((3, 1), (7, 4))


def test_article_roman_heading_and_title_before_bullets() -> None:
    text = (
        "ARTICLE I - DEFINITIONS\n\n"
        "In this Agreement the Loan means the amount sanctioned.\n\n"
        "Key Facts\n\u2022 Loan amount: INR 3.5 lakh\n\u2022 APR: 15.1%"
    )
    clauses = _segment(text)
    art = clauses[0]
    assert art.clause_type == "section" and art.clause_number == "ARTICLE I" and art.heading == "DEFINITIONS"
    key = next(c for c in clauses if c.cleaned_text == "Key Facts")
    assert key.clause_type == "heading"
    bullets = [c for c in clauses if c.clause_type == "bullet"]
    assert len(bullets) == 2 and all(b.parent_clause_id == key.clause_id for b in bullets)
