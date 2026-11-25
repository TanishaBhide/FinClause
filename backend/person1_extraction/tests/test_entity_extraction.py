"""Tests for financial entity extraction (regex + spaCy + context rules)."""

from __future__ import annotations

import pytest

from finclause_extraction.config import ExtractionConfig
from finclause_extraction.entities.ner import ClauseContext, FinancialEntityExtractor, find_value_candidates
from finclause_extraction.schemas import SENTENCE_LEVEL_LABELS


@pytest.fixture(scope="module")
def extractor() -> FinancialEntityExtractor:
    return FinancialEntityExtractor(ExtractionConfig())


def ents(extractor, text: str, **ctx):
    result = extractor.extract(ClauseContext(text=text, **ctx))
    return [(e.label.value, text[e.start:e.end], e) for e in result.entities]


def values(extractor, text: str, **ctx):
    """Only value entities (exclude sentence-level conditions)."""
    return [(l, s) for l, s, e in ents(extractor, text, **ctx)
            if l not in SENTENCE_LEVEL_LABELS]


@pytest.mark.parametrize(
    "surface,normalized",
    [
        ("Rs. 5,000", "5000 INR"), ("Rs 5,000", "5000 INR"), ("INR 5,000", "5000 INR"),
        ("₹5,000", "5000 INR"), ("₹ 5,000", "5000 INR"), ("INR 1,25,000", "125000 INR"),
        ("Rs. 2.5 lakh", "250000 INR"), ("₹5 crore", "50000000 INR"), ("10,000/-", "10000"),
        ("5000 rupees", "5000 INR"), ("Rs.5,000", "5000 INR"), ("Rs. 10,000/-", "10000 INR"),
    ],
)
def test_currency_amounts(extractor, surface: str, normalized: str) -> None:
    found = [c for c in find_value_candidates(f"An amount of {surface} is mentioned.") if c.kind == "money"]
    assert len(found) == 1
    text = f"An amount of {surface} is mentioned."
    assert text[found[0].start:found[0].end] == surface
    assert found[0].normalized.normalized == normalized


@pytest.mark.parametrize(
    "surface,normalized",
    [
        ("12%", "12%"), ("12.5% per annum", "12.5% per annum"), ("10.75% p.a.", "10.75% per annum"),
        ("1.5 percent per month", "1.5% per month"), ("2 per cent", "2%"),
    ],
)
def test_percentages(surface: str, normalized: str) -> None:
    text = f"A figure of {surface} applies"
    found = [c for c in find_value_candidates(text) if c.kind == "percent"]
    assert [text[c.start:c.end] for c in found] == [surface]
    assert found[0].normalized.normalized == normalized


def test_annual_interest_rate(extractor) -> None:
    out = values(extractor, "Interest shall be charged at 12% per annum.")
    assert out == [("INTEREST_RATE", "12% per annum")]


def test_monthly_rate_keeps_period(extractor) -> None:
    (label, surface, e), = [x for x in ents(extractor, "Interest at 1.5 percent per month on the loan.") if x[0] == "INTEREST_RATE"]
    assert e.normalized.unit == "percent_per_month"
    assert e.normalized.normalized == "1.5% per month"


def test_period_from_following_context(extractor) -> None:
    (_, surface, e), = [x for x in ents(extractor, "The rate of interest is 11% (fixed) per annum.") if x[0] == "INTEREST_RATE"]
    assert surface == "11%"
    assert e.normalized.normalized == "11% per annum"
    assert e.normalized.attributes["period_source"] == "following_context"
    assert e.attributes["rate_type"] == "fixed"


def test_processing_fee(extractor) -> None:
    assert values(extractor, "Processing fee of 2%.") == [("PROCESSING_FEE", "2%")]
    out = ents(extractor, "A processing fee of 2% of the loan amount plus GST at 18% shall be deducted.")
    labels = [(l, s) for l, s, _ in out]
    assert ("PROCESSING_FEE", "2%") in labels and ("TAX", "18%") in labels
    fee = next(e for l, s, e in out if l == "PROCESSING_FEE")
    assert fee.attributes["basis"] == "loan amount" and fee.attributes["taxes"] == "additional"


def test_percent_not_always_interest(extractor) -> None:
    assert values(extractor, "Processing fee of 2%.")[0][0] != "INTEREST_RATE"
    assert values(extractor, "GST at 18% applies.")[0][0] == "TAX"


def test_late_payment_penalties(extractor) -> None:
    assert values(extractor, "Penalty of Rs. 500.") == [("PENALTY", "Rs. 500")]
    assert values(extractor, "Late payment charge of Rs. 500.") == [("PENALTY", "Rs. 500")]
    out = values(extractor, "Penal interest of 2% per month shall be levied.")
    assert out == [("PENALTY", "2% per month")]


def test_trigger_after_value_with_connector(extractor) -> None:
    assert values(extractor, "Rs. 500 will be charged as late payment fee.") == [("PENALTY", "Rs. 500")]
    assert values(extractor, "The Borrower shall pay 2% of the loan amount as processing fee.") == [("PROCESSING_FEE", "2%")]


def test_foreclosure_and_prepayment(extractor) -> None:
    out = ents(extractor, "Foreclosure charges of 4% on the outstanding principal.")
    (label, surface, e), = out
    assert (label, surface) == ("FORECLOSURE_CHARGE", "4%")
    assert e.attributes["basis"] == "outstanding principal"
    assert values(extractor, "Part prepayment charges of 2% of the amount prepaid apply.") == [("PREPAYMENT_CHARGE", "2%")]


def test_charge_mention_without_value(extractor) -> None:
    out = ents(extractor, "Prepayment charges subject to applicable conditions.")
    (label, surface, e), = out
    assert (label, surface) == ("PREPAYMENT_CHARGE", "Prepayment charges")
    assert e.normalized is None and e.attributes["value_specified"] is False
    assert "subject to applicable conditions" in e.attributes["qualifiers"]
    assert e.extraction_method == "spacy_span_ruler"


def test_loan_amount_emi_and_tenure(extractor) -> None:
    out = values(extractor, "The Lender grants a loan of Rs. 5,00,000 repayable in 36 EMIs of Rs. 16,607 each.")
    assert out == [("LOAN_AMOUNT", "Rs. 5,00,000"), ("TENURE", "36 EMIs"), ("EMI", "Rs. 16,607")]
    assert values(extractor, "The tenure of the loan shall be 36 months.") == [("TENURE", "36 months")]
    assert values(extractor, "The loan amount is INR 2,00,000.") == [("LOAN_AMOUNT", "INR 2,00,000")]


def test_duration_that_is_not_tenure(extractor) -> None:
    out = values(extractor, "A grace period of thirty (30) days is allowed before penal interest applies.")
    assert ("GRACE_PERIOD", "thirty (30) days") in out
    out = values(extractor, "Foreclosure is allowed only after 12 months from disbursement.")
    assert ("DURATION", "12 months") in out


def test_dates(extractor) -> None:
    out = ents(extractor, "Agreement dated 15/01/2025; first EMI due on 5th March, 2025 or January 15, 2025.")
    dates = [(s, e.normalized.normalized) for l, s, e in out if l == "DATE"]
    assert dates == [("15/01/2025", "2025-01-15"), ("5th March, 2025", "2025-03-05"), ("January 15, 2025", "2025-01-15")]


def test_clause_numbers_are_not_dates(extractor) -> None:
    out = ents(extractor, "As per clause 1.2.12 and 2.5.1 of this Agreement.")
    assert not [l for l, _, _ in out if l == "DATE"]


def test_repayment_condition(extractor) -> None:
    out = ents(extractor, "EMI payable on the fifth day of every month.")
    (label, surface, e), = out
    assert label == "REPAYMENT_CONDITION"
    assert e.attributes["due_day_of_month"] == 5 and e.attributes["frequency"] == "monthly"


def test_variable_interest_condition(extractor) -> None:
    out = ents(extractor, "Floating interest rate linked to an external benchmark.")
    (label, surface, e), = out
    assert label == "VARIABLE_INTEREST_CONDITION"
    assert e.attributes["benchmark"] == "external benchmark" and e.attributes["rate_type"] == "floating"


def test_basis_points_in_rate_context(extractor) -> None:
    out = values(extractor, "Interest shall be the repo rate plus a spread of 250 basis points.")
    assert out == [("INTEREST_RATE", "250 basis points")]


def test_ambiguous_values_get_generic_labels(extractor) -> None:
    out = values(extractor, "The Borrower acknowledges 15% and Rs. 2,000 as stated above.")
    assert out == [("OTHER_PERCENTAGE", "15%"), ("OTHER_FINANCIAL_AMOUNT", "Rs. 2,000")]
    e = ents(extractor, "The Borrower acknowledges 15% as stated.")[0][2]
    assert e.extraction_method == "regex_pattern" and e.evidence.trigger_source == "pattern"


def test_multiple_entities_same_clause(extractor) -> None:
    text = ("Late payment charges of Rs. 500 per instance shall apply, and penal interest of 2% per month "
            "shall be levied on the overdue amount. Interest at 12.5% p.a. is payable.")
    out = values(extractor, text)
    assert out == [("PENALTY", "Rs. 500"), ("PENALTY", "2% per month"), ("INTEREST_RATE", "12.5% p.a.")]
    rs500 = next(e for l, s, e in ents(extractor, text) if s == "Rs. 500")
    assert rs500.attributes["frequency"] == "per instance"


def test_parent_clause_context(extractor) -> None:
    out = values(extractor, "(a) 4% of the outstanding principal if the loan is closed within 12 months;",
                 parent_text="3.3 Foreclosure charges shall be levied as under:")
    assert out[0] == ("FORECLOSURE_CHARGE", "4%")
    e = ents(extractor, "(a) 4% of the outstanding principal;", parent_text="3.3 Foreclosure charges shall be levied as under:")[0][2]
    assert e.evidence.trigger_source == "parent_clause"


def test_heading_context(extractor) -> None:
    out = ents(extractor, "2% of the amount prepaid.", heading="Prepayment Charges")
    (label, _, e), = out
    assert label == "PREPAYMENT_CHARGE" and e.evidence.trigger_source == "heading"


def test_bounds_and_other_charges(extractor) -> None:
    out = ents(extractor, "Documentation charges of Rs. 1,000 plus GST and stamp duty up to Rs. 500.")
    doc = next(e for l, s, e in out if s == "Rs. 1,000")
    stamp = next(e for l, s, e in out if s == "Rs. 500")
    assert doc.label.value == "OTHER_FINANCIAL_AMOUNT" and doc.attributes["related_to"] == "documentation charges"
    assert stamp.attributes["bound"] == "upper"


def test_evidence_records_trigger(extractor) -> None:
    e = ents(extractor, "Interest shall be charged at 12% per annum.")[0][2]
    assert e.evidence.trigger_text == "Interest" and e.evidence.trigger_label == "INTEREST"
    assert e.extraction_method == "regex_context"


def test_blank_spacy_fallback_still_extracts() -> None:
    blank = FinancialEntityExtractor(ExtractionConfig(use_spacy=False))
    assert blank.nlp_description.startswith("blank:en")
    out = [(e.label.value, e.start) for e in blank.extract_text("Processing fee of 2% and interest at 11% p.a.").entities]
    assert [l for l, _ in out] == ["PROCESSING_FEE", "INTEREST_RATE"]


def test_missing_spacy_model_falls_back_with_warning() -> None:
    x = FinancialEntityExtractor(ExtractionConfig(spacy_model="xx_model_that_does_not_exist"))
    assert x.load_warnings and x.load_warnings[0][0] == "SPACY_MODEL_UNAVAILABLE"
    assert x.extract_text("Penalty of Rs. 500.").entities[0].label.value == "PENALTY"


# ---- regressions found by stress-testing a differently formatted agreement ----
def test_basis_words_do_not_label_later_values(extractor) -> None:
    out = values(extractor, "Processing Fee: 1.5% of the Loan amount (non-refundable), subject to a minimum of Rs. 2,500;")
    assert out == [("PROCESSING_FEE", "1.5%"), ("PROCESSING_FEE", "Rs. 2,500")]
    minimum = next(e for l, s, e in ents(extractor, "Processing Fee: 1.5% of the Loan amount, subject to a minimum of Rs. 2,500;") if s == "Rs. 2,500")
    assert minimum.attributes["bound"] == "lower"


def test_leading_clause_number_is_not_a_value(extractor) -> None:
    out = values(extractor, "3.2 Instalments are payable on or before the 10th of each month.", clause_number="3.2")
    assert out == []


def test_single_mode_word_is_not_a_repayment_condition(extractor) -> None:
    labels = [l for l, _, _ in ents(extractor, "(ii) Cheque/NACH bounce charges: Rs. 750 per instance;")]
    assert labels == ["PENALTY"]


def test_symbol_and_at_sign_forms(extractor) -> None:
    assert values(extractor, "Penal interest @ 24% per annum on the overdue EMI.") == [("PENALTY", "24% per annum")]
    assert values(extractor, "Foreclosure charges: 5% + GST on principal outstanding.") == [("FORECLOSURE_CHARGE", "5%")]
    assert values(extractor, "Annual Percentage Rate (APR): 15.1%") == [("INTEREST_RATE", "15.1%")]


def test_ocr_confusions_of_emi(extractor) -> None:
    """Real Tesseract output read 'EMIs' as 'EMls' (lowercase L). The text is kept as-is,
    but the EMI/TENURE meaning must still be recognised."""
    for variant in ("EMls", "EM1s"):
        out = values(extractor, f"The loan shall be repaid in 36 {variant} of Rs. 16,607 each.")
        assert out == [("TENURE", f"36 {variant}"), ("EMI", "Rs. 16,607")], variant
    tenure = next(e for l, s, e in ents(extractor, "Repayable in 36 EMls.") if l == "TENURE")
    assert tenure.normalized.normalized == "36 instalments"
