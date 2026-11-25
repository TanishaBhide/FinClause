"""Tests for entity normalisation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from finclause_extraction.entities.normalizer import (
    normalize_bps,
    normalize_date,
    normalize_duration,
    normalize_money,
    normalize_percent,
    parse_number,
    parse_ordinal_day,
)


@pytest.mark.parametrize(
    "number,cur,expected",
    [
        ("5,000", "₹", "5000 INR"),
        ("10,000", "Rs.", "10000 INR"),
        ("1,25,000", "INR", "125000 INR"),
        ("5000", "rupees", "5000 INR"),
        ("5,00,000", "Rs", "500000 INR"),
    ],
)
def test_inr_formatting(number: str, cur: str, expected: str) -> None:
    n = normalize_money(number, currency_token=cur)
    assert n.normalized == expected
    assert n.currency == "INR" and n.unit == "INR"


def test_indian_and_western_grouping() -> None:
    assert parse_number("1,25,000") == (Decimal("125000"), {"digit_grouping": "indian"})
    assert parse_number("12,50,00,000")[0] == Decimal("125000000")
    assert parse_number("125,000") == (Decimal("125000"), {"digit_grouping": "western"})
    assert parse_number("1,250,000")[1]["digit_grouping"] == "western"
    assert parse_number("12,34,5")[1]["digit_grouping"] == "irregular"
    with pytest.raises(ValueError):
        parse_number("abc")


@pytest.mark.parametrize(
    "number,scale,expected",
    [
        ("2.5", "lakh", "250000"),
        ("1", "crore", "10000000"),
        ("5", "crores", "50000000"),
        ("3", "lacs", "300000"),
        ("1.25", "Cr.", "12500000"),
        ("2.3456789", "lakh", "234567.89"),
    ],
)
def test_lakh_crore_conversion(number: str, scale: str, expected: str) -> None:
    assert normalize_money(number, scale=scale).normalized == expected


def test_scaled_amount_with_currency() -> None:
    n = normalize_money("2.5", scale="lakh", currency_token="Rs.")
    assert n.normalized == "250000 INR" and n.numeric == Decimal("250000")
    assert n.attributes["scale_word"] == "lakh"


def test_missing_currency_not_assumed() -> None:
    n = normalize_money("2.5", scale="lakh")
    assert n.currency is None and n.unit is None
    assert n.attributes["currency_stated"] is False
    assert "currency_hint" in n.attributes
    slash = normalize_money("10,000", slash_dash=True)
    assert slash.normalized == "10000" and slash.currency is None and slash.attributes["slash_dash_suffix"]


def test_decimal_precision_preserved() -> None:
    assert normalize_money("5,000.50", currency_token="Rs").normalized == "5000.50 INR"
    assert normalize_money("0.10", currency_token="Rs").numeric == Decimal("0.10")
    assert normalize_percent("10.750").normalized == "10.750%"
    assert normalize_percent("10.75").numeric == Decimal("10.75")


@pytest.mark.parametrize(
    "number,period,expected,unit",
    [
        ("12.5", "p.a.", "12.5% per annum", "percent_per_annum"),
        ("12.5", "per annum", "12.5% per annum", "percent_per_annum"),
        ("10.75", "p.a", "10.75% per annum", "percent_per_annum"),
        ("1.5", "per month", "1.5% per month", "percent_per_month"),
        ("2", "p.m.", "2% per month", "percent_per_month"),
        ("0.05", "per day", "0.05% per day", "percent_per_day"),
        ("12", None, "12%", "percent"),
    ],
)
def test_percentage_normalization(number, period, expected, unit) -> None:
    n = normalize_percent(number, period)
    assert n.normalized == expected and n.unit == unit


def test_monthly_rate_not_annualized() -> None:
    n = normalize_percent("2", "per month")
    assert n.numeric == Decimal("2")
    assert "annum" not in n.normalized
    assert n.attributes["period"] == "per_month"


def test_negative_percentage() -> None:
    n = normalize_percent("0.25", sign="-")
    assert n.numeric == Decimal("-0.25") and n.normalized == "-0.25%"


def test_basis_points() -> None:
    n = normalize_bps("250")
    assert n.normalized == "250 bps (2.5%)" and n.unit == "basis_points"
    assert n.attributes["percent_equivalent"] == "2.5"


def test_dates() -> None:
    assert normalize_date("15", "01", "2025", numeric=True).normalized == "2025-01-15"
    ambiguous = normalize_date("05", "03", "2025", numeric=True)
    assert ambiguous.normalized == "2025-03-05" and ambiguous.attributes["date_order_assumed"] == "DMY"
    assert normalize_date("05", "03", "2025", "MDY", numeric=True).normalized == "2025-05-03"
    assert normalize_date("03", "25", "2025", numeric=True).normalized == "2025-03-25"  # only MDY valid
    assert normalize_date("5", "March", "2025").normalized == "2025-03-05"
    assert normalize_date(None, "Jan", "2025").normalized == "2025-01"
    assert normalize_date("31", "02", "2025", numeric=True) is None  # invalid date rejected
    assert normalize_date("1", "1", "25", numeric=True).attributes["two_digit_year"]


def test_durations() -> None:
    assert normalize_duration("36", None, "months").normalized == "36 months"
    y = normalize_duration("3", None, "years")
    assert y.normalized == "3 years" and y.attributes["derived_months"] == "36"
    w = normalize_duration(None, "thirty", "days")
    assert w.normalized == "30 days" and w.attributes["number_in_words"] == "thirty"
    assert normalize_duration("24", None, "EMIs").unit == "instalments"


def test_ordinal_day() -> None:
    assert parse_ordinal_day("5th") == 5
    assert parse_ordinal_day("fifth") == 5
    assert parse_ordinal_day("twenty-first") == 21
    assert parse_ordinal_day("last") == "last"
