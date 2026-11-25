"""Normalisation of financial surface forms.

Rules (see README "Normalisation"):
* Amounts use :class:`decimal.Decimal` so precision is never lost through floats.
* Indian (``1,25,000``) and Western (``125,000``) digit grouping are both accepted.
* ``lakh``/``lac`` = 100,000; ``crore``/``cr`` = 10,000,000.
* A currency is only reported when it is written (``₹``, ``Rs``, ``INR``, ``rupees``).
  A bare ``2.5 lakh`` or ``10,000/-`` gets ``currency=None`` and a hint attribute.
* Rates keep their period: a monthly rate is never converted to an annual one.
* Ambiguous numeric dates are read using the configured order (default DMY)
  and flagged with ``date_order_assumed``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

SCALES: dict[str, int] = {
    "thousand": 1_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "million": 1_000_000, "mn": 1_000_000,
    "crore": 10_000_000, "crores": 10_000_000, "cr": 10_000_000, "cr.": 10_000_000,
    "billion": 1_000_000_000,
}

_CURRENCY_TOKENS: dict[str, str] = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "re": "INR", "re.": "INR", "inr": "INR",
    "rupee": "INR", "rupees": "INR", "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
}

_PERIODS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"per\s+annum|p\.\s?a\.?|^pa$|per\s+year|a\s+year|annually|yearly", re.I), "per_annum", "per annum"),
    (re.compile(r"per\s+month|p\.\s?m\.?|a\s+month|monthly", re.I), "per_month", "per month"),
    (re.compile(r"per\s+day|daily|a\s+day", re.I), "per_day", "per day"),
    (re.compile(r"per\s+quarter|quarterly", re.I), "per_quarter", "per quarter"),
    (re.compile(r"half[\s-]year", re.I), "per_half_year", "per half-year"),
]

_INDIAN_GROUPING = re.compile(r"^\d{1,2}(?:,\d{2})*,\d{3}$|^\d{1,3}$")
_WESTERN_GROUPING = re.compile(r"^\d{1,3}(?:,\d{3})+$|^\d{1,3}$")

MONTHS: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORD_NUMBERS: dict[str, int] = {w: i for i, w in enumerate(_UNITS)}
for _t, _v in _TENS.items():
    WORD_NUMBERS[_t] = _v
    for _i in range(1, 10):
        WORD_NUMBERS[f"{_t}-{_UNITS[_i]}"] = _v + _i
        WORD_NUMBERS[f"{_t} {_UNITS[_i]}"] = _v + _i
WORD_NUMBERS["hundred"] = 100

_ORD_UNITS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
              "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
              "eighteenth", "nineteenth", "twentieth"]
ORDINAL_WORDS: dict[str, int] = {w: i for i, w in enumerate(_ORD_UNITS) if w}
for _i in range(1, 10):
    ORDINAL_WORDS[f"twenty-{_ORD_UNITS[_i]}"] = 20 + _i
    ORDINAL_WORDS[f"twenty {_ORD_UNITS[_i]}"] = 20 + _i
ORDINAL_WORDS.update({"thirtieth": 30, "thirty-first": 31, "thirty first": 31})

_DURATION_UNITS = {
    "month": "months", "months": "months", "year": "years", "years": "years", "yr": "years", "yrs": "years",
    "day": "days", "days": "days", "week": "weeks", "weeks": "weeks", "quarter": "quarters", "quarters": "quarters",
}


@dataclass
class Normalized:
    """Normalised representation of a value."""

    normalized: Optional[str]
    numeric: Optional[Decimal] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)


def decimal_to_str(value: Decimal) -> str:
    """Plain (non-scientific) string without trailing zeros."""
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def parse_number(text: str) -> tuple[Decimal, dict[str, Any]]:
    """Parse ``"1,25,000.50"`` style numbers exactly.

    Returns the value and attributes describing the digit grouping.

    Raises:
        ValueError: if ``text`` is not a number.
    """
    raw = text.strip()
    attrs: dict[str, Any] = {}
    integer = raw.split(".")[0]
    if "," in integer:
        if _INDIAN_GROUPING.match(integer) and not _WESTERN_GROUPING.match(integer):
            attrs["digit_grouping"] = "indian"
        elif _WESTERN_GROUPING.match(integer):
            # "2,500" is valid in both systems
            attrs["digit_grouping"] = "western" if not _INDIAN_GROUPING.match(integer) else "indian_or_western"
        else:
            attrs["digit_grouping"] = "irregular"
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError(f"not a number: {text!r}") from exc
    return value, attrs


def currency_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    return _CURRENCY_TOKENS.get(token.strip().lower())


def normalize_money(
    number: str, scale: Optional[str] = None, currency_token: Optional[str] = None, slash_dash: bool = False
) -> Normalized:
    """Normalise a monetary amount.

    >>> normalize_money("1,25,000", currency_token="INR").normalized
    '125000 INR'
    >>> normalize_money("2.5", scale="lakh").normalized
    '250000'
    """
    value, attrs = parse_number(number)
    factor = None
    if scale:
        factor = SCALES.get(scale.strip().lower().rstrip("."))
        if factor is None:
            raise ValueError(f"unknown scale word {scale!r}")
        value = value * factor
        attrs["scale_word"] = scale.strip()
        amount = decimal_to_str(value)
    else:
        amount = str(value)  # keeps written precision, e.g. 5000.50
    currency = currency_from_token(currency_token)
    if slash_dash:
        attrs["slash_dash_suffix"] = True
    if currency is None:
        attrs["currency_stated"] = False
        if slash_dash or factor in (100_000, 10_000_000):
            attrs["currency_hint"] = "Indian notation ('/-' or lakh/crore) usually denotes INR; currency not stated"
    return Normalized(
        normalized=f"{amount} {currency}" if currency else amount,
        numeric=value,
        unit=currency,
        currency=currency,
        attributes=attrs,
    )


def period_from_text(text: Optional[str]) -> Optional[tuple[str, str]]:
    """Map a period phrase (``p.a.``, ``per month``...) to ``(code, words)``."""
    if not text:
        return None
    t = text.strip()
    for pattern, code, words in _PERIODS:
        if pattern.search(t):
            return code, words
    return None


def normalize_percent(number: str, period_text: Optional[str] = None, sign: Optional[str] = None) -> Normalized:
    """Normalise a percentage, keeping its period.

    >>> normalize_percent("12.5", "p.a.").normalized
    '12.5% per annum'
    >>> normalize_percent("1.5", "per month").unit
    'percent_per_month'
    """
    value, attrs = parse_number(number)
    if sign == "-":
        value = -value
    text_value = ("-" if sign == "-" else "") + str(parse_number(number)[0])
    period = period_from_text(period_text)
    if period:
        code, words = period
        return Normalized(f"{text_value}% {words}", value, f"percent_{code}", None, {**attrs, "period": code})
    return Normalized(f"{text_value}%", value, "percent", None, attrs)


def normalize_bps(number: str) -> Normalized:
    """Basis points: keeps the bps value and adds the percentage equivalent."""
    value, attrs = parse_number(number)
    pct = value / Decimal(100)
    attrs["percent_equivalent"] = decimal_to_str(pct)
    return Normalized(f"{decimal_to_str(value)} bps ({decimal_to_str(pct)}%)", value, "basis_points", None, attrs)


def normalize_date(
    day: Optional[str], month: Optional[str], year: str, date_order: str = "DMY", numeric: bool = False
) -> Optional[Normalized]:
    """Return an ISO date (``YYYY-MM-DD``, or ``YYYY-MM`` without a day). None if invalid."""
    attrs: dict[str, Any] = {}
    y = int(year)
    if len(year) == 2:
        y += 2000 if y < 70 else 1900
        attrs["two_digit_year"] = True
    if month is None:
        return None
    if numeric:
        a, b = int(day or 0), int(month)
        if a > 12 and b <= 12:
            d, m = a, b
        elif b > 12 and a <= 12:
            d, m = b, a
            attrs["date_order_detected"] = "MDY"
        elif a == b:
            d, m = a, b
        else:
            d, m = (b, a) if date_order.upper() == "MDY" else (a, b)
            attrs["date_order_assumed"] = date_order.upper()
    else:
        m = MONTHS.get(month.lower().rstrip("."))
        if m is None:
            return None
        if day is None:
            return Normalized(f"{y:04d}-{m:02d}", None, "date", None, {"precision": "month"})
        d = int(day)
    try:
        iso = date(y, m, d).isoformat()
    except ValueError:
        return None
    return Normalized(iso, None, "date", None, attrs)


def normalize_duration(number: Optional[str], word: Optional[str], unit: str) -> Normalized:
    """Normalise ``36 months``, ``thirty (30) days``, ``3 years``, ``24 EMIs``."""
    attrs: dict[str, Any] = {}
    if number:
        value, _ = parse_number(number)
    else:
        value = Decimal(WORD_NUMBERS[word.lower()])  # type: ignore[union-attr]
        attrs["number_in_words"] = word
    u = unit.lower()
    if re.fullmatch(r"em[il1]s?", u) or "instal" in u or u.startswith("equated"):
        canon = "instalments"
    else:
        canon = _DURATION_UNITS.get(u, u)
    if canon == "years":
        attrs["derived_months"] = decimal_to_str(value * 12)
    return Normalized(f"{decimal_to_str(value)} {canon}", value, canon, None, attrs)


def normalize_duration_options(values: list[str], unit: str, kind: str) -> Normalized:
    """``12 to 24 months`` (kind="range") or ``3, 6 and 9 months`` (kind="options")."""
    base = normalize_duration(values[-1], None, unit)
    nums = [str(int(v)) for v in values]
    if kind == "range":
        text, attrs = f"{nums[0]} to {nums[-1]} {base.unit}", {"duration_range": nums}
    else:
        text, attrs = f"{', '.join(nums[:-1])} or {nums[-1]} {base.unit}", {"duration_options": nums}
    return Normalized(text, base.numeric, base.unit, None, attrs)


def parse_ordinal_day(text: str) -> Optional[int | str]:
    """``"5th"`` -> 5, ``"fifth"`` -> 5, ``"last"`` -> ``"last"``."""
    t = text.strip().lower()
    if t == "last":
        return "last"
    m = re.match(r"(\d{1,2})", t)
    if m:
        return int(m.group(1))
    return ORDINAL_WORDS.get(t)
