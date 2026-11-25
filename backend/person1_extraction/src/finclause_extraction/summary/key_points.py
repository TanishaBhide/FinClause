"""Plain-language key points built from extracted entities (templates, no ML).

Each :class:`KeyPoint` is generated from one or more entities (value points), one
sentence (condition points) or a definition sentence, and keeps the IDs of its source
clause and entities, so every point can be checked against the original text.

Wording is neutral: points describe what the document says and flag items worth
checking ("attention"); they never call a term unfair, illegal or fraudulent.

Rules (in order):
1. Value entities -> "<noun>: <value><details>[, <condition>]". The noun comes from the
   label or the trigger phrase ("Bounce charges"); details are basis ("of the loan amount"),
   bound ("up to", "minimum"), frequency, GST and refundability. A conditional tail from
   the same sentence ("if the loan is closed within 12 months") is appended for charges
   and rates. Monthly rates also show an approximate yearly figure (clearly marked).
2. Two values with the same label joined by "to"/"and" ("Rs. 100 to Rs. 1,300") -> one range.
3. GST stated in the same sentence as a fee is attached to that fee.
4. Sentence-level conditions -> the sentence rewritten in plain English, one point per
   sentence. Repayment/interest/cancellation sentences whose content is already covered
   by value points are not repeated.
5. A charge mentioned without an amount -> a "check" point, unless an amount for the same
   kind of charge is given elsewhere in the document, or the text negates it
   ("without any foreclosure charges").
6. Dates -> "Key dates" with a role taken from the sentence (first EMI due, policy period...).
7. Definitions ("X means ...") of financial terms -> "The basics".
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

from ..preprocessing.segmenter import split_sentences
from ..schemas import SENTENCE_LEVEL_LABELS, Clause, Entity, KeyPoint
from .simplifier import simplify_text

CATEGORY_ORDER = ["basics", "costs", "interest", "repayment", "penalties", "default", "security", "cover",
                  "exclusions", "cancellation", "changes", "dates", "check"]
CATEGORY_TITLES = {
    "basics": "The basics", "costs": "Fees and charges", "interest": "Interest", "repayment": "Repayment",
    "penalties": "If you pay late", "default": "If you default", "security": "Security and guarantees",
    "cover": "What is covered", "exclusions": "What is not covered", "cancellation": "Cancellation and refunds",
    "changes": "Terms that can change", "dates": "Key dates", "check": "Worth checking before you sign",
}

#: label -> (category, default noun, importance)
_VALUE_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "LOAN_AMOUNT": ("basics", "Loan amount", "info"),
    "EMI": ("basics", "Monthly instalment (EMI)", "info"),
    "TENURE": ("basics", "Loan tenure", "info"),
    "PREMIUM": ("basics", "Premium", "info"),
    "SUM_INSURED": ("cover", "Maximum cover (sum insured)", "info"),
    "INTEREST_RATE": ("interest", "Interest rate", "info"),
    "PROCESSING_FEE": ("costs", "Processing fee (one-time)", "info"),
    "FORECLOSURE_CHARGE": ("costs", "Foreclosure charge (closing the loan early)", "attention"),
    "PREPAYMENT_CHARGE": ("costs", "Part-prepayment charge (paying back part of the loan early)", "attention"),
    "TAX": ("costs", "GST / tax", "info"),
    "OTHER_FINANCIAL_AMOUNT": ("costs", "Other amount", "info"),
    "OTHER_PERCENTAGE": ("costs", "Other rate", "info"),
    "PENALTY": ("penalties", "Late payment / penalty charge", "attention"),
    "GRACE_PERIOD": ("repayment", "Grace period", "info"),
    "CO_PAYMENT": ("cover", "Co-payment", "attention"),
    "COVERAGE_LIMIT": ("cover", "Limit", "attention"),
    "WAITING_PERIOD": ("cover", "Waiting period", "attention"),
}
_BASICS_LABELS = {"LOAN_AMOUNT", "EMI", "TENURE", "PREMIUM", "SUM_INSURED"}
_CHARGE_LABELS = {"PROCESSING_FEE", "FORECLOSURE_CHARGE", "PREPAYMENT_CHARGE", "PENALTY", "OTHER_FINANCIAL_AMOUNT",
                  "OTHER_PERCENTAGE", "COVERAGE_LIMIT"}
#: sentence-level label -> (category, importance); first match in this order wins for a sentence.
_CONDITION_CATEGORY = [
    ("DEFAULT_CONDITION", "default", "attention"),
    ("SECURITY_CONDITION", "security", "attention"),
    ("EXCLUSION", "exclusions", "attention"),
    ("CANCELLATION_CONDITION", "cancellation", "info"),
    ("DISCRETIONARY_CHANGE_CONDITION", "changes", "attention"),
    ("VARIABLE_INTEREST_CONDITION", "interest", "attention"),
    ("REPAYMENT_CONDITION", "repayment", "info"),
]
_CONDITION_TAIL = re.compile(
    r"\b(if|within|after|before|beyond|thereafter|until|unless|provided|for|where|above|below|aged|in case|"
    r"depending on|between)\b", re.I)
_LEAD_CONDITION = re.compile(r"^\s*(?:\(?[a-z0-9.]{1,6}\)?\s+)?((?:for|if|in case of|where)\b[^,]{3,90}),", re.I)
_PERIOD_WORDS = [("per annum", "a year"), ("per month", "a month"), ("per day", "a day"), ("per quarter", "a quarter"),
                 ("per half-year", "every six months")]
_RANGE_JOIN = re.compile(r"^\s*(?:to|and|-|–|or)\s*$", re.I)
_DEFINITION_RE = re.compile(r"\b(means|shall mean|refers to|is defined as)\b", re.I)
_FINANCIAL_WORD_RE = re.compile(r"\b(emi|interest|loan|premium|charges?|fees?|insured|discount|rate|claim|cover)\b", re.I)
_MODE_NAMES = {"nach": "NACH auto-debit", "ecs": "ECS", "auto-debit": "auto-debit", "autodebit": "auto-debit",
               "standing instruction": "standing instruction", "post-dated": "post-dated cheques",
               "postdated": "post-dated cheques", "pdc": "post-dated cheques"}


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def indian_grouping(value: Decimal) -> str:
    """1234567.5 -> '12,34,567.5'"""
    sign = "-" if value < 0 else ""
    text = format(abs(value), "f")
    integer, _, frac = text.partition(".")
    if len(integer) > 3:
        head, tail = integer[:-3], integer[-3:]
        head = re.sub(r"(\d)(?=(\d{2})+$)", r"\1,", head)
        integer = f"{head},{tail}"
    return sign + integer + (f".{frac}" if frac else "")


def pretty_date(iso: str) -> str:
    """'2025-04-07' -> '7 Apr 2025' ('2025-04' -> 'Apr 2025')."""
    try:
        parts = [int(p) for p in iso.split("-")]
        if len(parts) == 3:
            d = date(*parts)
            return f"{d.day} {d.strftime('%b %Y')}"
        return date(parts[0], parts[1], 1).strftime("%b %Y")
    except (ValueError, TypeError):
        return iso


def split_value(e: Entity) -> tuple[str, str]:
    """Return (core value, period words): ('₹5,00,000', ''), ('1%', 'a day')."""
    if e.normalized_value is None:
        return e.cleaned_surface_text, ""
    if e.currency == "INR" or (e.unit is None and e.attributes.get("currency_stated") is False):
        try:
            amount = Decimal(e.normalized_value.split()[0])
            return ("₹" if e.currency == "INR" else "") + indian_grouping(amount), ""
        except (InvalidOperation, IndexError):
            return e.normalized_value, ""
    text = e.normalized_value
    for long, short in _PERIOD_WORDS:
        if text.endswith(" " + long):
            return text[: -len(long) - 1], short
    return text, ""


def display_value(e: Entity) -> str:
    core, period = split_value(e)
    return f"{core} {period}".strip()


def _approx_annual(e: Entity) -> str:
    if e.unit == "percent_per_month" and e.numeric_value is not None:
        annual = Decimal(str(e.numeric_value)) * 12
        return f" (about {format(annual.normalize(), 'f')}% a year if charged for 12 months)"
    return ""


def _details(e: Entity) -> str:
    a = e.attributes
    parts = ""
    if a.get("basis"):
        prep = a.get("basis_preposition", "of")
        parts += f" {prep} the {a['basis']}"
    if a.get("frequency"):
        parts += f" {a['frequency']}"
    if a.get("taxes") == "additional":
        parts += " + GST"
    elif a.get("taxes") == "included":
        parts += " (including GST)"
    if a.get("refundable") is False:
        parts += " (non-refundable)"
    return parts


def _bound_prefix(e: Entity) -> str:
    return {"upper": "up to ", "lower": "minimum "}.get(e.attributes.get("bound", ""), "")


def _sentence_for(clause: Clause, e: Entity) -> tuple[int, int]:
    for s, t in split_sentences(clause.cleaned_text):
        if s <= e.cleaned_start_offset < t:
            return s, t
    return 0, len(clause.cleaned_text)


def _strip_parentheticals(text: str) -> str:
    prev = None
    while prev != text:
        prev, text = text, re.sub(r"^\s*\([^()]*\)", "", text)
    return text


def _condition_fragments(clause: Clause, e: Entity, end: int) -> tuple[str, str]:
    """(lead, tail) conditions around a value, e.g. tail 'if the loan is closed within 12 months'."""
    s, t = _sentence_for(clause, e)
    sent = clause.cleaned_text[s:t]
    after = _strip_parentheticals(clause.cleaned_text[end:t])
    basis = e.attributes.get("basis")
    if basis:
        i = after.lower().find(basis.lower())
        if 0 <= i <= 15:
            after = after[i + len(basis):]
    after = re.sub(r"^\s*(?:per\s+\w+|a\s+(?:day|month|year))\b", "", after, flags=re.I)  # period already shown
    tail = ""
    m = _CONDITION_TAIL.search(after)
    if m and m.start() <= 40:
        tail = after[m.start():]
        # stop at the next clause or at the next *value* ("... within 3 months and 50% if ...")
        tail = re.split(r";|\.\s|\.$|,\s+(?:and|but|which|so)\s|\s+and\s+(?=\d[\d,.]*\s*(?:%|per\s?cent)|rs\b|₹)",
                        tail, flags=re.I)[0]
        tail = simplify_text(tail.strip().rstrip(",.;:"), glossary=False, strip_marker=False)
        tail = tail[0].lower() + tail[1:] if tail else tail
    lead = ""
    lm = _LEAD_CONDITION.match(sent)
    if lm and lm.end() <= e.cleaned_start_offset - s + 1:
        lead = simplify_text(lm.group(1), glossary=False, strip_marker=False)
    return lead, tail[:160]


def _noun_for(e: Entity, default: str) -> str:
    trig = (e.evidence.trigger_text if e.evidence else None) or ""
    related = e.attributes.get("related_to") or ""
    t = trig.lower()
    if e.label == "INTEREST_RATE" and t in ("apr", "annual percentage rate"):
        return "Annual Percentage Rate (APR - the total yearly cost including fees)"
    if e.label in ("PENALTY", "OTHER_FINANCIAL_AMOUNT", "OTHER_PERCENTAGE", "COVERAGE_LIMIT") and (trig or related):
        noun = re.sub(r"\s+", " ", trig or related).strip()
        noun = noun[0].upper() + noun[1:]
        return noun + " limit" if e.label == "COVERAGE_LIMIT" else noun
    if e.label == "TENURE":
        if "policy" in t:
            return "Policy period"
        if "emi" in t or "tenures" in t:
            return "EMI tenure"
        if e.unit == "instalments":
            return "Number of instalments"
    return default


def _category_for(e: Entity, default: str) -> str:
    related = (e.attributes.get("related_to") or "").lower()
    if "bonus" in related:
        return "cover"
    if any(w in related for w in ("refund", "cancel", "free look")):
        return "cancellation"
    if "order" in related:
        return "basics"
    return default


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self, clauses: list[Clause], entities: list[Entity]):
        self.clauses = {c.clause_id: c for c in clauses}
        self.entities = entities
        self.points: list[KeyPoint] = []
        self.seen_text: dict[str, KeyPoint] = {}
        #: labels that have at least one stated value somewhere in the document
        self.valued_labels = {e.label for e in entities if e.normalized_value is not None}
        self.covered_sentences: set[tuple[str, int]] = set()  # (clause_id, sentence start) with value points

    def add(self, category: str, text: str, importance: str, clause: Clause, entity_ids: list[str],
            source: str, method: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\.{2,}$", ".", text)
        if not text.endswith((".", "?", "!")):
            text += "."
        key = text.lower()
        if key in self.seen_text:  # same statement elsewhere: just add the reference
            p = self.seen_text[key]
            if clause.clause_id not in p.clause_ids:
                p.clause_ids.append(clause.clause_id)
            p.entity_ids.extend(i for i in entity_ids if i not in p.entity_ids)
            p.page_numbers = sorted(set(p.page_numbers) | set(clause.page_numbers))
            return
        point = KeyPoint(point_id="", category=category, text=text, importance=importance,
                         clause_ids=[clause.clause_id], entity_ids=list(entity_ids),
                         page_numbers=list(clause.page_numbers), source_text=source, method=method)
        self.seen_text[key] = point
        self.points.append(point)

    def build(self) -> list[KeyPoint]:
        by_clause: dict[str, list[Entity]] = {}
        for e in self.entities:
            by_clause.setdefault(e.clause_id, []).append(e)
        for clause_id, clause in self.clauses.items():
            ents = by_clause.get(clause_id, [])
            self._definition_points(clause)
            if ents:
                self._value_points(clause, ents)
                self._date_points(clause, ents)
                self._condition_points(clause, ents)
        order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
        self.points.sort(key=lambda p: (order.get(p.category, 99), p.page_numbers[:1], p.clause_ids[0]))
        for i, p in enumerate(self.points, start=1):
            p.point_id = f"point_{i:03d}"
        return self.points

    # ---- values ----
    def _value_points(self, clause: Clause, ents: list[Entity]) -> None:
        values = [e for e in ents if e.label not in SENTENCE_LEVEL_LABELS and e.label != "DATE"]
        taxes = [e for e in values if e.label == "TAX"]
        used: set[str] = set()
        for i, e in enumerate(values):
            if e.entity_id in used or e.label == "TAX":
                continue
            s, t = _sentence_for(clause, e)
            source = clause.cleaned_text[s:t]
            if e.attributes.get("value_specified") is False:
                self._charge_mention(clause, e, source)
                continue
            if any(p.source_text == source and clause.clause_id in p.clause_ids
                   and e.cleaned_surface_text in p.text for p in self.points):
                continue  # already stated in the condition of an earlier point from this sentence
            if e.label == "DURATION":
                if (e.attributes.get("related_to") or "") == "VALIDITY":
                    self.add("dates", f"Valid for: {display_value(e)}", "info", clause, [e.entity_id], source, "template")
                    self.covered_sentences.add((clause.clause_id, s))
                continue
            template = _VALUE_TEMPLATES.get(e.label)
            if template is None:
                continue
            category, default_noun, importance = template
            if e.label in ("OTHER_FINANCIAL_AMOUNT", "OTHER_PERCENTAGE") and not e.attributes.get("related_to"):
                continue  # meaning unknown: reported as an UNCLASSIFIED_VALUE warning instead
            category = _category_for(e, category)
            ids = [e.entity_id]
            core, period = split_value(e)
            end_entity = e
            # rule 2: ranges such as "Rs. 100 to Rs. 1,300", "13% and 16% p.a."
            nxt = values[i + 1] if i + 1 < len(values) else None
            if nxt is not None and nxt.label == e.label and nxt.cleaned_start_offset < t \
                    and _RANGE_JOIN.match(clause.cleaned_text[e.cleaned_end_offset:nxt.cleaned_start_offset]):
                ncore, nperiod = split_value(nxt)
                core = f"{core} to {ncore}"
                period = period or nperiod
                ids.append(nxt.entity_id)
                used.add(nxt.entity_id)
                end_entity = nxt
            value = _bound_prefix(e) + core + (f" {period}" if period else "") + _details(e)
            if e.label in _CHARGE_LABELS:  # rule 3: attach GST stated in the same sentence
                for tax in taxes:
                    if s <= tax.cleaned_start_offset < t and tax.entity_id not in used \
                            and tax.cleaned_start_offset > e.cleaned_start_offset:
                        value += f" (GST {display_value(tax)})"
                        used.add(tax.entity_id)
                        ids.append(tax.entity_id)
                        break
            extra = ""
            if e.label == "INTEREST_RATE":
                rt = e.attributes.get("rate_type")
                extra = {"fixed": " (fixed)", "floating": " (floating - it can change)",
                         "variable": " (variable - it can change)"}.get(rt or "", "")
                bench = next((x for x in ents if x.label == "VARIABLE_INTEREST_CONDITION"
                              and x.cleaned_start_offset <= e.cleaned_start_offset < x.cleaned_end_offset), None)
                if bench is not None and bench.attributes.get("benchmark"):
                    extra += f", linked to the {bench.attributes['benchmark']}"
                    ids.append(bench.entity_id)
            lead, tail = ("", "") if e.label in _BASICS_LABELS else _condition_fragments(clause, e, end_entity.cleaned_end_offset)
            noun = _noun_for(e, default_noun)
            if e.label == "CO_PAYMENT":
                text = f"Co-payment: you pay {value} of each claim yourself"
            elif e.label == "WAITING_PERIOD":
                text = f"Waiting period ({value}): " + simplify_text(source, glossary=False)
                tail = ""
            else:
                text = f"{noun}: {value}{extra}{_approx_annual(e)}"
            if tail:
                text += f", {tail}"
            if lead:
                text = f"{lead[0].upper() + lead[1:]}: {text[0].lower() + text[1:]}"
            self.add(category, text, importance, clause, ids, source, "template")
            self.covered_sentences.add((clause.clause_id, s))
        for tax in taxes:
            if tax.entity_id not in used:
                s, t = _sentence_for(clause, tax)
                self.add("costs", f"GST / tax: {display_value(tax)}{_details(tax)}", "info", clause,
                         [tax.entity_id], clause.cleaned_text[s:t], "template")
                self.covered_sentences.add((clause.clause_id, s))

    def _charge_mention(self, clause: Clause, e: Entity, source: str) -> None:
        if e.label in self.valued_labels:
            return  # an amount for this kind of charge is stated elsewhere in the document
        noun = re.sub(r"\s+", " ", e.cleaned_surface_text)
        noun = noun[0].upper() + noun[1:]
        verb = "apply" if noun.lower().endswith(("s", "charges")) else "applies"
        quals = e.attributes.get("qualifiers") or []
        ref = " (" + "; ".join(quals) + ")" if quals else ""
        follow = " The details are in the sub-clauses below." if e.attributes.get("details_follow") else \
            " Ask for the exact amount in writing."
        self.add("check", f"{noun} {verb}, but no amount is stated in this clause{ref}.{follow}",
                 "attention", clause, [e.entity_id], source, "template")

    # ---- dates ----
    def _date_points(self, clause: Clause, ents: list[Entity]) -> None:
        dates = [e for e in ents if e.label == "DATE" and e.normalized_value]
        skip: set[str] = set()
        for i, e in enumerate(dates):
            if e.entity_id in skip:
                continue
            s, t = _sentence_for(clause, e)
            sent = clause.cleaned_text[s:t].lower()
            nxt = dates[i + 1] if i + 1 < len(dates) else None
            if nxt is not None and re.fullmatch(r"\s*(?:to|till|until|-|–)\s*",
                                                clause.cleaned_text[e.cleaned_end_offset:nxt.cleaned_start_offset], re.I):
                what = "Policy period" if "policy" in sent else "Period"
                self.add("dates", f"{what}: {pretty_date(e.normalized_value)} to {pretty_date(nxt.normalized_value)}",
                         "info", clause, [e.entity_id, nxt.entity_id], clause.cleaned_text[s:t], "template")
                skip.add(nxt.entity_id)
                continue
            what = next((w for k, w in [("first emi", "First EMI due"), ("first instalment", "First instalment due"),
                                        ("valid", "Offer valid until"), ("expir", "Expiry"), ("commenc", "Start date"),
                                        ("start", "Start date"), ("made", "Agreement date"), ("updated", "Last updated"),
                                        ("date of kfs", "Date of this statement"), ("dated", "Document date"),
                                        ("date:", "Document date"), ("due", "Due date")] if k in sent), None)
            if what:
                self.add("dates", f"{what}: {pretty_date(e.normalized_value)}", "info", clause,
                         [e.entity_id], clause.cleaned_text[s:t], "template")
            self.covered_sentences.add((clause.clause_id, s))

    # ---- conditions ----
    def _condition_points(self, clause: Clause, ents: list[Entity]) -> None:
        sentences: dict[tuple[int, int], list[Entity]] = {}
        for e in ents:
            if e.label in SENTENCE_LEVEL_LABELS:
                sentences.setdefault((e.cleaned_start_offset, e.cleaned_end_offset), []).append(e)
        for (s, t), group in sorted(sentences.items()):
            labels = {e.label for e in group}
            category, importance = next(((c, i) for l, c, i in _CONDITION_CATEGORY if l in labels), ("check", "info"))
            source = clause.cleaned_text[s:t]
            repay = next((e for e in group if e.label == "REPAYMENT_CONDITION"), None)
            has_values = (clause.clause_id, s) in self.covered_sentences
            if category == "repayment" and repay is not None and repay.attributes.get("due_day_of_month"):
                day = repay.attributes["due_day_of_month"]
                day_txt = "the last day" if day == "last" else f"day {day}"
                modes = list(dict.fromkeys(_MODE_NAMES.get(m.lower(), m) for m in repay.attributes.get("payment_modes") or []))
                if "NACH auto-debit" in modes and "auto-debit" in modes:
                    modes.remove("auto-debit")
                text = f"Your EMI is due on {day_txt} of every month" + (f", paid by {' / '.join(modes)}" if modes else "")
                method = "template"
            else:
                if has_values and category in ("repayment", "interest", "cancellation") and not (
                        repay is not None and (repay.attributes.get("payment_modes") or repay.attributes.get("appropriation_clause"))):
                    continue  # rule 4: already covered by value points from the same sentence
                text, method = simplify_text(source), "simplified_sentence"
            ids = [e.entity_id for e in group]
            ids += [e.entity_id for e in ents if e.label not in SENTENCE_LEVEL_LABELS and s <= e.cleaned_start_offset < t]
            self.add(category, text, importance, clause, ids, source, method)

    # ---- definitions ----
    def _definition_points(self, clause: Clause) -> None:
        for s, t in split_sentences(clause.cleaned_text):
            sent = clause.cleaned_text[s:t]
            m = _DEFINITION_RE.search(sent)
            if m and m.start() <= 60 and _FINANCIAL_WORD_RE.search(sent):
                self.add("basics", "What it means: " + simplify_text(sent, glossary=False), "info", clause,
                         [], sent, "simplified_sentence")


def build_key_points(clauses: list[Clause], entities: list[Entity]) -> list[KeyPoint]:
    """Build plain-language key points from the extraction result (see module docstring)."""
    return _Builder(clauses, entities).build()
