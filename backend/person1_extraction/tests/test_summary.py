"""Tests for the plain-language layer: simplifier, key points, document type."""

from __future__ import annotations

import re

import pytest

from finclause_extraction import analyze_text
from finclause_extraction.summary import detect_document_type, simplify_text
from finclause_extraction.summary.key_points import indian_grouping, pretty_date
from decimal import Decimal


# ---------------- simplifier ----------------
@pytest.mark.parametrize(
    "legal,plain",
    [
        ("The Borrower shall pay the EMI.", "You must pay the EMI."),
        ("In the event of any default, the Lender may recall the loan.", "If there is any default, the lender may recall the loan."),
        ("Prior to disbursement, the Borrower shall submit documents.", "Before payout, you must submit documents."),
        ("If the Borrower fails to pay, it shall constitute an event of default.", "If you fail to pay, it counts as an event of default."),
        ("The first EMI shall fall due on 07/04/2025.", "The first EMI is due on 07/04/2025."),
        ("The Company shall not be liable to make any payment for cosmetic surgery.", "The company will not pay for cosmetic surgery."),
        ("The Lender reserves the right to revise the charges.", "The lender may revise the charges."),
    ],
)
def test_simplifier_rewrites_legalese(legal: str, plain: str) -> None:
    assert simplify_text(legal, glossary=False) == plain


def test_simplifier_never_changes_numbers() -> None:
    text = ("3.2 The Borrower shall pay penal interest at 2% per month on Rs. 1,25,000.50 from 05/03/2025, "
            "notwithstanding anything contained herein.")
    out = simplify_text(text)
    nums = re.compile(r"\d[\d,]*(?:\.\d+)?")
    assert nums.findall(out) == nums.findall(text[4:])  # only the leading clause number is dropped


def test_simplifier_subject_aware_verb_agreement() -> None:
    # "you" is not the subject here, so "is" must stay "is"
    assert "furnished by you is found" in simplify_text("If any information furnished by the Borrower is found incorrect.")
    assert simplify_text("The Borrower has agreed.").startswith("You have")


def test_simplifier_glossary_once_and_possessives() -> None:
    out = simplify_text("Foreclosure charges apply. Foreclosure is allowed. The Lender may report the default to the credit "
                        "bureau, affecting the credit score of the Borrower.")
    assert out.count("(closing the loan early)") == 1
    assert "your credit score" in out


# ---------------- key points ----------------
LOAN_TEXT = """1. FEES
1.1 A processing fee of 2% of the loan amount plus GST at 18% shall be deducted, subject to a minimum of Rs. 3,000.
1.2 Late payment charges ranging from Rs. 100 to Rs. 1,300 shall apply.
1.3 Penal interest of 2% per month shall be levied on the overdue amount.
1.4 Foreclosure charges shall be levied as under:
(a) 4% of the principal outstanding if the loan is foreclosed within 12 months;
(b) 2% of the principal outstanding thereafter.
1.5 Part-prepayment charges shall be as per the Schedule of Charges.
1.6 The Borrower may cancel within 3 days without any foreclosure charges.
2. DEFAULT
2.1 The Lender may report the default to credit information companies.
2.2 The Lender reserves the right to revise the fees at its sole discretion."""


@pytest.fixture(scope="module")
def loan_points():
    result = analyze_text(LOAN_TEXT)
    return result, {p.text: p for p in result.key_points}


def find(points: dict, pattern: str):
    hits = [p for t, p in points.items() if re.search(pattern, t)]
    assert hits, f"no key point matching {pattern!r} in {list(points)}"
    return hits[0]


def test_fee_point_with_gst_and_minimum(loan_points) -> None:
    _, pts = loan_points
    p = find(pts, r"^Processing fee \(one-time\): 2% of the loan amount")
    assert "GST 18%" in p.text and p.category == "costs"
    find(pts, r"Processing fee \(one-time\): minimum ₹3,000")


def test_range_is_merged(loan_points) -> None:
    _, pts = loan_points
    p = find(pts, r"Late payment charges: ₹100 to ₹1,300")
    assert len(p.entity_ids) == 2 and p.importance == "attention"


def test_monthly_rate_gets_marked_approximation(loan_points) -> None:
    _, pts = loan_points
    p = find(pts, r"^Penal interest: 2% a month")
    assert "about 24% a year" in p.text


def test_tiered_conditions_are_kept(loan_points) -> None:
    _, pts = loan_points
    find(pts, r"Foreclosure charge .*: 4% of the principal outstanding, if the loan is foreclosed .*within 12 months")
    p = find(pts, r"Foreclosure charge .*: 2% of the principal outstanding")
    assert p.category == "costs"


def test_charge_without_amount_and_negation(loan_points) -> None:
    result, pts = loan_points
    p = find(pts, r"^Part-prepayment charges apply, but no amount is stated")
    assert p.category == "check" and "schedule of charges" in p.text.lower()
    # "without any foreclosure charges" is not a charge, and foreclosure amounts are stated elsewhere
    assert not [t for t in pts if t.lower().startswith("foreclosure charges apply")]


def test_condition_points_are_simplified_and_traceable(loan_points) -> None:
    result, pts = loan_points
    p = find(pts, r"credit information companies")
    assert p.category == "default" and p.method == "simplified_sentence"
    find(pts, r"entirely at its own choice")
    clause_ids = {c.clause_id for c in result.clauses}
    entity_ids = {e.entity_id for e in result.entities}
    for point in result.key_points:
        assert set(point.clause_ids) <= clause_ids and set(point.entity_ids) <= entity_ids
        assert point.page_numbers and point.source_text
    assert [p.point_id for p in result.key_points] == [f"point_{i:03d}" for i in range(1, len(result.key_points) + 1)]


def test_clause_simplified_text_present(loan_points) -> None:
    result, _ = loan_points
    c = next(c for c in result.clauses if c.numbering_path == "2.1")
    assert c.simplified_text.startswith("The lender may report the default")


def test_no_judgemental_words(loan_points) -> None:
    result, _ = loan_points
    text = " ".join(p.text for p in result.key_points).lower()
    for word in ("fraud", "illegal", "unfair", "scam", "hidden"):
        assert word not in text


# ---------------- helpers & document type ----------------
def test_formatting_helpers() -> None:
    assert indian_grouping(Decimal("1250000")) == "12,50,000"
    assert indian_grouping(Decimal("999")) == "999"
    assert pretty_date("2025-04-07") == "7 Apr 2025"
    assert pretty_date("2025-01") == "Jan 2025"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("KEY FACT STATEMENT (KFS) annual percentage rate", "key_fact_statement"),
        ("We are pleased to inform you that your loan has been sanctioned. Sanction letter.", "sanction_letter"),
        ("The Sum Insured is Rs. 5,00,000. Premium. Waiting period. Exclusions.", "insurance_policy"),
        ("No Cost EMI is available at checkout on select credit cards.", "emi_terms"),
        ("This Loan Agreement between the Borrower and the Lender.", "loan_agreement"),
        ("A recipe for tomato soup.", "unknown"),
    ],
)
def test_document_type(text: str, expected: str) -> None:
    assert detect_document_type(text)[0] == expected
