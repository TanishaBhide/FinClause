"""Tests for text cleaning: whitespace, line breaks, hyphenation and value preservation."""

from __future__ import annotations

import re

import pytest

from finclause_extraction.preprocessing.cleaner import clean_text


def _roundtrip_ok(original: str) -> None:
    """Every cleaned character must map to a valid original index (monotonic)."""
    c = clean_text(original)
    assert len(c.offset_map) == len(c.text) + 1
    assert all(0 <= i <= len(original) for i in c.offset_map)
    assert c.offset_map[:-1] == sorted(c.offset_map[:-1])


def test_multiple_spaces_collapsed() -> None:
    c = clean_text("Interest   at    12%   per  annum")
    assert c.text == "Interest at 12% per annum"


def test_tabs_and_nbsp_normalized() -> None:
    assert clean_text("Rs. 5,000\tonly").text == "Rs. 5,000 only"


def test_single_line_breaks_become_spaces() -> None:
    assert clean_text("The Borrower shall\npay interest\nmonthly.").text == "The Borrower shall pay interest monthly."


def test_paragraph_breaks_preserved() -> None:
    c = clean_text("First paragraph.\n\n\nSecond paragraph.")
    assert c.text == "First paragraph.\nSecond paragraph."
    assert clean_text("A.\n\nB.", preserve_paragraphs=False).text == "A. B."


def test_hyphenated_line_break_joined() -> None:
    assert clean_text("the repay-\nment schedule").text == "the repayment schedule"


def test_compound_hyphen_kept_across_line_break() -> None:
    assert clean_text("any pre-\npayment charges").text == "any pre-payment charges"
    assert clean_text("non-\nrefundable fee").text == "non-refundable fee"


def test_hyphen_before_number_not_joined() -> None:
    # "-\n5" is not word hyphenation
    assert clean_text("charges of Rs.-\n500").text == "charges of Rs.- 500"


def test_soft_hyphen_and_zero_width_removed() -> None:
    assert clean_text("fore­\nclosure​ charges").text == "foreclosure charges"


def test_ligatures_and_quotes_normalized() -> None:
    assert clean_text("ﬁnal “Borrower”’s").text == "final \"Borrower\"'s"


def test_unicode_minus_and_dashes() -> None:
    assert clean_text("−500 and Section 7 – Fees").text == "-500 and Section 7 - Fees"


@pytest.mark.parametrize(
    "value",
    [
        "₹5,000", "₹ 5,000", "Rs. 10,000/-", "INR 1,25,000", "Rs. 2.5 lakh", "₹5 crore",
        "12.5% p.a.", "10.75% per annum", "1.5 percent per month", "-2.75%", "15/01/2025", "05-03-2025",
        "1.1.1", "2(a)", "(ii)", "Clause 5", "12 bps", "0.50%", "1,00,00,000",
    ],
)
def test_financial_values_preserved_verbatim(value: str) -> None:
    text = f"The amount   {value}\nshall apply."
    cleaned = clean_text(text).text
    assert value in cleaned


def test_digits_and_symbols_never_removed() -> None:
    text = "Fee: ₹ 1,250.50 + 18% GST; penal 2%/month; Rs.\n500/- (Rupees Five Hundred only)\n\n3.2 Foreclosure 4%."
    c = clean_text(text)
    keep = re.compile(r"[0-9%₹/.,+()-]")
    assert keep.findall(text.replace("\n", "")) == keep.findall(c.text)


def test_clause_numbering_preserved() -> None:
    text = "1.1 Interest\n1.1.1 Rate\n(a) monthly\n(ii) yearly\nSection 7 - Fees and Charges"
    c = clean_text(text)
    for marker in ["1.1 Interest", "1.1.1 Rate", "(a) monthly", "(ii) yearly", "Section 7 - Fees and Charges"]:
        assert marker in c.text


def test_original_text_untouched_and_offsets_map_back() -> None:
    original = "Processing  fee of\nRs. 5,000 and the repay-\nment of ﬁve EMIs"
    c = clean_text(original)
    assert c.original == original
    i = c.text.index("Rs. 5,000")
    s, e = c.to_original_span(i, i + len("Rs. 5,000"))
    assert original[s:e] == "Rs. 5,000"
    j = c.text.index("repayment")
    s, e = c.to_original_span(j, j + len("repayment"))
    assert original[s:e] == "repay-\nment"
    _roundtrip_ok(original)


def test_empty_and_whitespace_input() -> None:
    assert clean_text("").text == ""
    assert clean_text("  \n\n  ").text == ""
    _roundtrip_ok("   ")


def test_control_characters_removed() -> None:
    assert clean_text("Rs.\x00 500\x07 only").text == "Rs. 500 only"
