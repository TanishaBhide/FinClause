"""Financial entity extraction: regex values + spaCy trigger matching + context rules.

How a label is decided
======================
1. **Value detection (regex, ``patterns.py``)** finds money, percentages, basis
   points, dates and durations. Overlaps are resolved longest-span-first
   (``Rs. 2.5 lakh`` beats ``2.5 lakh``), then by pattern priority
   (date > prefixed money > suffixed money > scaled > ``/-`` > bps/percent > duration).
2. **Trigger detection (spaCy ``span_ruler``)** finds phrases such as
   "processing fee", "penal interest", "tenure". Nested triggers are discarded in
   favour of the longest one ("penal interest" hides "interest").
3. **Context rules** choose the trigger that explains each value, in order:

   R1  a trigger *after* the value joined by a connector
       ("Rs. 500 will be charged **as** late payment fee") - ``sentence_following``;
   R2  the nearest applicable trigger *before* the value in the same sentence,
       within ``context_window_chars`` - ``sentence_preceding``; a generic
       trigger ("charges", "fee") yields to a specific trigger right after the value;
   R3  a specific trigger shortly after the value in the same sentence;
   R4  the last applicable trigger in the parent clause's lead-in sentence
       ("Foreclosure charges shall be levied as under:" -> "(a) 4% ...") - ``parent_clause``;
   R5  a trigger in the clause heading - ``heading``;
   R6  otherwise a generic label (OTHER_FINANCIAL_AMOUNT / OTHER_PERCENTAGE / DURATION).

   A trigger is *applicable* only if it makes sense for the value kind: e.g.
   "interest" turns a percentage into INTEREST_RATE but a rupee amount next to
   "interest" stays OTHER_FINANCIAL_AMOUNT; "loan amount" never labels a percentage.
   Text such as "of the loan amount" directly after a value is recorded as its
   ``basis`` and never used as a trigger.
4. **Conditions**: sentence-level VARIABLE_INTEREST_CONDITION and
   REPAYMENT_CONDITION entities, and value-less charge mentions
   ("Prepayment charges subject to applicable conditions").
5. **spaCy NER fallback**: DATE/MONEY/PERCENT entities from the pretrained model
   are added only where no regex entity overlaps (``extraction_method=spacy_ner``).
   Entities from a *custom trained* model whose labels match :class:`EntityLabel`
   are added the same way (``spacy_custom_model``). No model is trained here.

Offsets returned by this module index into the *cleaned* clause text; the
pipeline maps them back to the original text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..config import ExtractionConfig
from ..preprocessing.segmenter import is_title_like, split_sentences
from ..schemas import SENTENCE_LEVEL_LABELS, EntityLabel, Evidence
from . import patterns as P
from .normalizer import (
    Normalized,
    normalize_bps,
    normalize_date,
    normalize_duration,
    normalize_duration_options,
    normalize_money,
    normalize_percent,
    parse_ordinal_day,
)

logger = logging.getLogger(__name__)

SPANS_KEY = "finclause_triggers"
L = EntityLabel

_KIND_MAP: dict[str, dict[str, EntityLabel]] = {
    "money": {
        "INTEREST": L.OTHER_FINANCIAL_AMOUNT, "PROCESSING_FEE": L.PROCESSING_FEE, "PENALTY": L.PENALTY,
        "FORECLOSURE": L.FORECLOSURE_CHARGE, "PREPAYMENT": L.PREPAYMENT_CHARGE, "LOAN_AMOUNT": L.LOAN_AMOUNT,
        "EMI": L.EMI, "TAX": L.TAX, "OTHER_CHARGE": L.OTHER_FINANCIAL_AMOUNT,
        "PREMIUM": L.PREMIUM, "SUM_INSURED": L.SUM_INSURED, "CO_PAYMENT": L.CO_PAYMENT,
        "LIMIT": L.COVERAGE_LIMIT, "CANCEL": L.OTHER_FINANCIAL_AMOUNT,
    },
    "percent": {
        "INTEREST": L.INTEREST_RATE, "PROCESSING_FEE": L.PROCESSING_FEE, "PENALTY": L.PENALTY,
        "FORECLOSURE": L.FORECLOSURE_CHARGE, "PREPAYMENT": L.PREPAYMENT_CHARGE, "TAX": L.TAX,
        "OTHER_CHARGE": L.OTHER_PERCENTAGE, "CO_PAYMENT": L.CO_PAYMENT, "LIMIT": L.COVERAGE_LIMIT,
        "PREMIUM": L.OTHER_PERCENTAGE, "CANCEL": L.OTHER_PERCENTAGE,
    },
}
#: Duration labels chosen by the explaining trigger (anything else -> DURATION).
_DURATION_MAP = {"TENURE": L.TENURE, "WAITING": L.WAITING_PERIOD, "GRACE": L.GRACE_PERIOD}
_KIND_MAP["bps"] = _KIND_MAP["percent"]
_GENERIC = {"money": L.OTHER_FINANCIAL_AMOUNT, "percent": L.OTHER_PERCENTAGE, "bps": L.OTHER_PERCENTAGE}
_MENTION_LABELS = {
    "PROCESSING_FEE": L.PROCESSING_FEE, "PENALTY": L.PENALTY,
    "FORECLOSURE": L.FORECLOSURE_CHARGE, "PREPAYMENT": L.PREPAYMENT_CHARGE,
}
_PRIORITY = {"date": 0, "money_prefix": 1, "money_suffix": 2, "money_scale": 3, "money_slash": 4,
             "bps": 5, "percent": 5, "duration": 6}


@dataclass
class Trigger:
    start: int
    end: int
    label: str
    text: str


@dataclass
class ValueCandidate:
    kind: str  # money | percent | bps | date | duration
    pattern: str
    start: int
    end: int
    normalized: Optional[Normalized]
    method: str = "regex"


@dataclass
class RawEntity:
    """Entity in cleaned-text coordinates (before offset mapping and ID assignment)."""

    label: EntityLabel
    start: int
    end: int
    extraction_method: str
    normalized: Optional[Normalized] = None
    evidence: Optional[Evidence] = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClauseContext:
    text: str  #: cleaned clause text
    heading: Optional[str] = None
    parent_text: Optional[str] = None
    is_heading_clause: bool = False
    clause_number: Optional[str] = None  #: leading marker as written, excluded from value detection


@dataclass
class ClauseExtraction:
    entities: list[RawEntity]
    sentences: list[tuple[int, int]]
    triggers: list[Trigger]


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


# --------------------------------------------------------------------------- #
# spaCy setup
# --------------------------------------------------------------------------- #
def load_nlp(config: ExtractionConfig):
    """Load the spaCy pipeline and add the trigger ``span_ruler``.

    Returns ``(nlp, description, warnings)``. Falls back to ``spacy.blank("en")``
    (tokenizer only, no download needed) if the model is unavailable.
    """
    import spacy

    warnings: list[tuple[str, str]] = []
    nlp = None
    if config.use_spacy and config.spacy_model:
        try:
            nlp = spacy.load(config.spacy_model, exclude=["parser", "lemmatizer"])
        except OSError as exc:
            msg = (f"spaCy model '{config.spacy_model}' could not be loaded ({exc.__class__.__name__}). "
                   "Falling back to a blank English tokenizer: rule-based extraction still works but the "
                   "pretrained NER fallback is disabled. Install with: python -m spacy download en_core_web_sm")
            logger.warning(msg)
            warnings.append(("SPACY_MODEL_UNAVAILABLE", msg))
    if nlp is None:
        nlp = spacy.blank("en")
        description = f"blank:en (spaCy {spacy.__version__})"
    else:
        description = f"{nlp.meta.get('lang', 'en')}_{nlp.meta.get('name', '?')} {nlp.meta.get('version', '?')}"
    ruler = nlp.add_pipe(
        "span_ruler", name="finclause_triggers",
        config={"spans_key": SPANS_KEY, "phrase_matcher_attr": "LOWER", "overwrite": False},
    )
    ruler.add_patterns([{"label": lab, "pattern": phrase} for lab, phrases in P.TRIGGERS.items() for phrase in phrases])
    return nlp, description, warnings


def _filter_nested(triggers: Iterable[Trigger]) -> list[Trigger]:
    kept: list[Trigger] = []
    for t in sorted(triggers, key=lambda t: (-(t.end - t.start), t.start)):
        if any(k.start <= t.start and t.end <= k.end for k in kept):
            continue
        kept.append(t)
    return sorted(kept, key=lambda t: t.start)


# --------------------------------------------------------------------------- #
# Value candidates
# --------------------------------------------------------------------------- #
def find_value_candidates(text: str, date_order: str = "DMY") -> list[ValueCandidate]:
    """Find non-overlapping money/percent/bps/date/duration values in ``text``."""
    found: list[ValueCandidate] = []

    def add(kind: str, pattern: str, m: re.Match, fn) -> None:
        try:
            norm = fn(m)
        except (ValueError, KeyError) as exc:
            logger.debug("normalisation failed for %r: %s", m.group(0), exc)
            return
        if norm is None:
            return
        found.append(ValueCandidate(kind, pattern, m.start(), m.end(), norm))

    for m in P.MONEY_PREFIX_RE.finditer(text):
        add("money", "money_prefix", m, lambda m: normalize_money(m["num"], m["scale"], m["cur"], bool(m["slash"])))
    for m in P.MONEY_SUFFIX_RE.finditer(text):
        add("money", "money_suffix", m, lambda m: normalize_money(m["num"], m["scale"], m["cur"]))
    for m in P.MONEY_SCALE_RE.finditer(text):
        add("money", "money_scale", m, lambda m: normalize_money(m["num"], m["scale"]))
    for m in P.MONEY_SLASH_RE.finditer(text):
        add("money", "money_slash", m, lambda m: normalize_money(m["num"], slash_dash=True))
    for m in P.PERCENT_RE.finditer(text):
        add("percent", "percent", m, lambda m: normalize_percent(m["num"], m["period"], m["sign"]))
    for m in P.BPS_RE.finditer(text):
        add("bps", "bps", m, lambda m: normalize_bps(m["num"]))
    for m in P.DATE_NUMERIC_RE.finditer(text):
        if m["d"]:
            add("date", "date", m, lambda m: normalize_date(m["d"], m["m"], m["y"], date_order, numeric=True))
        else:
            add("date", "date", m, lambda m: normalize_date(m["d2"], m["m2"], m["y2"], date_order, numeric=True))
    for m in P.DATE_ISO_RE.finditer(text):
        add("date", "date", m, lambda m: normalize_date(m["d"], m["m"], m["y"], "DMY", numeric=True))
    for regex in (P.DATE_DMY_TEXT_RE, P.DATE_MDY_TEXT_RE):
        for m in regex.finditer(text):
            add("date", "date", m, lambda m: normalize_date(m["d"], m["mon"], m["y"]))
    for m in P.DATE_MONTH_YEAR_RE.finditer(text):
        add("date", "date", m, lambda m: normalize_date(None, m["mon"], m["y"]))
    for m in P.DURATION_RANGE_RE.finditer(text):
        add("duration", "duration", m, lambda m: normalize_duration_options([m["lo"], m["hi"]], m["unit"], "range"))
    for m in P.DURATION_LIST_RE.finditer(text):
        add("duration", "duration", m, lambda m: normalize_duration_options(
            [x.strip() for x in m["items"].split(",")] + [m["last"]], m["unit"], "options"))
    for m in P.DURATION_RE.finditer(text):
        add("duration", "duration", m, lambda m: normalize_duration(m["num"] or m["paren"], None if (m["num"] or m["paren"]) else m["word"], m["unit"]))

    accepted: list[ValueCandidate] = []
    for c in sorted(found, key=lambda c: (-(c.end - c.start), _PRIORITY[c.pattern], c.start)):
        if not any(_overlaps(c.start, c.end, a.start, a.end) for a in accepted):
            accepted.append(c)
    return sorted(accepted, key=lambda c: c.start)


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class FinancialEntityExtractor:
    """Extract financial entities from cleaned clause texts.

    The spaCy pipeline is loaded once per instance; reuse the instance.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.nlp, self.nlp_description, self.load_warnings = load_nlp(self.config)
        self._ruler = self.nlp.get_pipe("finclause_triggers")
        # Labels a custom-trained model may emit. "DATE" is excluded because the
        # pretrained English NER uses the same name; it goes through the
        # validated spaCy fallback path instead.
        self._custom_labels = {l.value for l in EntityLabel} - {"DATE"}
        self._trigger_cache: dict[str, list[Trigger]] = {}

    # ---- triggers ----
    def _triggers_from_doc(self, doc) -> list[Trigger]:
        spans = doc.spans[SPANS_KEY] if SPANS_KEY in doc.spans else []
        return _filter_nested(Trigger(s.start_char, s.end_char, s.label_, s.text) for s in spans)

    def triggers_for(self, text: str) -> list[Trigger]:
        """Trigger phrases in an arbitrary text (cached; uses only the tokenizer + ruler)."""
        if text not in self._trigger_cache:
            if len(self._trigger_cache) > 2048:
                self._trigger_cache.clear()
            self._trigger_cache[text] = self._triggers_from_doc(self._ruler(self.nlp.make_doc(text)))
        return self._trigger_cache[text]

    # ---- public API ----
    def extract(self, context: ClauseContext) -> ClauseExtraction:
        return self.extract_batch([context])[0]

    def extract_text(self, text: str) -> ClauseExtraction:
        """Convenience: extract from a single cleaned text with no clause context."""
        return self.extract(ClauseContext(text=text))

    def extract_batch(self, contexts: list[ClauseContext]) -> list[ClauseExtraction]:
        docs = self.nlp.pipe([c.text for c in contexts], batch_size=64)
        return [self._extract_one(ctx, doc) for ctx, doc in zip(contexts, docs)]

    # ---- core ----
    def _extract_one(self, ctx: ClauseContext, doc) -> ClauseExtraction:
        text = ctx.text
        triggers = self._triggers_from_doc(doc)
        sentences = split_sentences(text)
        entities: list[RawEntity] = []

        candidates = find_value_candidates(text, self.config.date_order)
        candidates += self._spacy_value_candidates(doc, candidates)
        # The leading clause number ("3.2 Instalments ...") is never a value.
        marker_end = len(ctx.clause_number) if ctx.clause_number and text.startswith(ctx.clause_number) else 0
        candidates = sorted((c for c in candidates if c.start >= marker_end), key=lambda c: c.start)

        def sentence_of(pos: int) -> tuple[int, int]:
            return next(((s, e) for s, e in sentences if s <= pos < e), (0, len(text)))

        # Words inside a value's basis ("2% of the *loan amount*") must not label other values.
        basis_spans: list[tuple[int, int]] = []
        for cand in candidates:
            if cand.kind in ("money", "percent", "bps"):
                m = P.BASIS_RE.match(text[cand.end:sentence_of(cand.start)[1]])
                if m:
                    basis_spans.append((cand.end + m.start("basis"), cand.end + m.end("basis")))
        value_triggers = [t for t in triggers if not any(b0 <= t.start and t.end <= b1 for b0, b1 in basis_spans)]

        for cand in candidates:
            entities.append(self._label_value(cand, text, sentence_of(cand.start), value_triggers, ctx))

        entities += self._condition_entities(text, sentences)
        entities += self._charge_mentions(text, sentences, triggers, entities, ctx)
        entities += self._custom_model_entities(doc, entities)
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))
        return ClauseExtraction(entities=entities, sentences=sentences, triggers=triggers)

    def _spacy_value_candidates(self, doc, existing: list[ValueCandidate]) -> list[ValueCandidate]:
        out: list[ValueCandidate] = []
        for ent in getattr(doc, "ents", ()):
            if ent.label_ not in ("DATE", "MONEY", "PERCENT") or not re.search(r"\d", ent.text):
                continue
            if any(_overlaps(ent.start_char, ent.end_char, c.start, c.end) for c in existing + out):
                continue
            norm: Optional[Normalized] = None
            if ent.label_ == "DATE":
                if not (re.search(r"\b(?:19|20)\d{2}\b", ent.text) or re.search(
                        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", ent.text, re.I)):
                    continue  # "36 months", "the fifth day" etc. are not calendar dates
                if re.search(r"\b(month|months|year|years|days?|weeks?)\b", ent.text, re.I) and not re.search(r"\b(?:19|20)\d{2}\b", ent.text):
                    continue
                norm = Normalized(None, None, "date", None, {"unparsed": True})
                kind = "date"
            else:
                num = re.search(P.NUM, ent.text)
                if not num:
                    continue
                try:
                    if ent.label_ == "MONEY":
                        cur = re.search(r"₹|\$|€|\b(?:rs|re|inr|usd|eur|rupees?)\b", ent.text, re.I)
                        scale = re.search(P.SCALE, ent.text[num.end():], re.I)
                        norm = normalize_money(num.group(0), scale.group(0) if scale else None, cur.group(0) if cur else None)
                        kind = "money"
                    else:
                        norm = normalize_percent(num.group(0))
                        kind = "percent"
                except ValueError:
                    continue
            out.append(ValueCandidate(kind, "spacy_" + ent.label_.lower(), ent.start_char, ent.end_char, norm, method="spacy_ner"))
        return out

    # ---- labelling ----
    def _choose_trigger(
        self, cand: ValueCandidate, text: str, sent: tuple[int, int], triggers: list[Trigger], applicable
    ) -> tuple[Optional[Trigger], Optional[str], dict[str, Any]]:
        attrs: dict[str, Any] = {}
        s0, s1 = sent
        basis = P.BASIS_RE.match(text[cand.end:s1])
        follow_from = cand.end + (basis.end() if basis else 0)
        if basis:
            attrs["basis"] = re.sub(r"\s+", " ", basis.group("basis"))
            attrs["basis_preposition"] = basis.group("prep").lower()
        following = [t for t in triggers if t.start >= follow_from and t.end <= s1 and applicable(t)]
        first_following = following[0] if following else None
        if first_following and P.FOLLOWING_CONNECTOR_RE.match(text[follow_from:first_following.start]):
            return first_following, "sentence_following", attrs
        window = self.config.context_window_chars
        # A tax word only labels a value it directly precedes ("GST at 18%"), not e.g.
        # "plus GST, subject to a minimum of Rs. 3,000".
        preceding = [t for t in triggers if t.end <= cand.start and t.start >= s0
                     and cand.start - t.end <= window and applicable(t)
                     and (t.label != "TAX" or cand.start - t.end <= 15)]
        if preceding:
            nearest = preceding[-1]
            if nearest.text.lower() in P.GENERIC_TRIGGERS and first_following \
                    and first_following.text.lower() not in P.GENERIC_TRIGGERS \
                    and first_following.start - follow_from <= 40:
                return first_following, "sentence_following", attrs
            return nearest, "sentence_preceding", attrs
        if first_following and first_following.text.lower() not in P.GENERIC_TRIGGERS \
                and first_following.start - follow_from <= 60:
            return first_following, "sentence_following", attrs
        return None, None, attrs

    def _context_fallback(self, ctx: ClauseContext, applicable) -> tuple[Optional[Trigger], Optional[str]]:
        if ctx.parent_text:
            sents = split_sentences(ctx.parent_text)
            if sents:
                s, e = sents[-1]
                cands = [t for t in self.triggers_for(ctx.parent_text) if s <= t.start < e and applicable(t)]
                if cands:
                    return cands[-1], "parent_clause"
        if ctx.heading:
            cands = [t for t in self.triggers_for(ctx.heading) if applicable(t)]
            specific = [t for t in cands if t.text.lower() not in P.GENERIC_TRIGGERS]
            if specific or cands:
                return (specific or cands)[0], "heading"
        return None, None

    def _label_value(
        self, cand: ValueCandidate, text: str, sent: tuple[int, int], triggers: list[Trigger], ctx: ClauseContext
    ) -> RawEntity:
        norm = cand.normalized
        base_method = "spacy_ner" if cand.method == "spacy_ner" else "regex_pattern"
        if cand.kind == "date":
            return RawEntity(L.DATE, cand.start, cand.end, base_method, norm,
                             Evidence(rule="date_pattern", trigger_source="pattern" if cand.method == "regex" else "model"))

        if cand.kind == "duration":
            applicable = lambda t: True  # noqa: E731
        else:
            mapping = _KIND_MAP[cand.kind]
            applicable = lambda t: t.label in mapping  # noqa: E731

        trigger, source, attrs = self._choose_trigger(cand, text, sent, triggers, applicable)
        if trigger is None:
            trigger, source = self._context_fallback(ctx, applicable)

        # extra context attributes
        before = text[max(0, cand.start - 30):cand.start]
        after = text[cand.end:sent[1]]
        if P.UPPER_BOUND_RE.search(before):
            attrs["bound"] = "upper"
        elif P.LOWER_BOUND_RE.search(before):
            attrs["bound"] = "lower"
        if cand.kind in ("money", "percent", "bps"):
            after_basis = after[len(attrs.get("basis", "")) + 8:] if "basis" in attrs else after
            if P.PLUS_TAX_RE.match(after) or P.PLUS_TAX_RE.match(after_basis) or P.EXCL_TAX_RE.search(text[sent[0]:sent[1]]):
                attrs["taxes"] = "additional"
            elif P.INCL_TAX_RE.search(text[sent[0]:sent[1]]):
                attrs["taxes"] = "included"
            if re.search(r"\bnon[-\s]?refundable\b", text[sent[0]:sent[1]], re.I):
                attrs["refundable"] = False
        if cand.kind == "money":
            freq = P.FREQUENCY_RE.match(after)
            if freq:
                attrs["frequency"] = "per " + freq.group("freq").lower()
        if cand.kind == "percent" and norm is not None and "period" not in norm.attributes:
            fp = P.FOLLOWING_PERIOD_RE.match(after)
            if not fp and "basis" in attrs:  # "1% of the Sum Insured per day"
                bm = P.BASIS_RE.match(after)
                fp = P.FOLLOWING_PERIOD_RE.match(after[bm.end():]) if bm else None
            if fp:
                m = P.PERCENT_RE.match(text, cand.start)
                if m:
                    norm = normalize_percent(m["num"], fp.group("period"), m["sign"])
                    norm.attributes["period_source"] = "following_context"

        # label
        if cand.kind == "duration":
            unit = norm.unit if norm else None
            if trigger is not None and trigger.label in _DURATION_MAP:
                label = _DURATION_MAP[trigger.label]
            elif unit == "instalments" and (trigger is None or trigger.label in ("EMI", "TENURE")):
                label, source = L.TENURE, source or "pattern"
            else:
                label = L.DURATION
                if trigger is not None:
                    attrs["related_to"] = trigger.label
        elif trigger is not None:
            label = _KIND_MAP[cand.kind][trigger.label]
            if label in (L.OTHER_FINANCIAL_AMOUNT, L.OTHER_PERCENTAGE):
                attrs["related_to"] = trigger.text.lower()
        else:
            label = _GENERIC[cand.kind]

        if label == L.INTEREST_RATE:
            rt = P.RATE_TYPE_RE.search(text[sent[0]:sent[1]])
            if rt:
                attrs["rate_type"] = rt.group(1).lower()

        if trigger is not None:
            method = "spacy_ner" if cand.method == "spacy_ner" else "regex_context"
            evidence = Evidence(rule=f"value:{cand.kind}+trigger:{source}", trigger_text=trigger.text,
                                trigger_label=trigger.label, trigger_source=source)
        else:
            method = base_method
            evidence = Evidence(rule=f"value:{cand.kind}+no_context", trigger_source="pattern" if cand.method == "regex" else "model")
        return RawEntity(label, cand.start, cand.end, method, norm, evidence, attrs)

    # ---- conditions ----
    def _condition_entities(self, text: str, sentences: list[tuple[int, int]]) -> list[RawEntity]:
        out: list[RawEntity] = []
        for s, e in sentences:
            sent = text[s:e]
            cues = [re.sub(r"\s+", " ", m.group("cue").lower()) for m in P.VARIABLE_CUES_RE.finditer(sent)]
            strong = [c for c in cues if c in P.VARIABLE_STRONG or c.startswith(("revis", "may ", "subject"))]
            if strong and P.INTEREST_WORD_RE.search(sent):
                unique = list(dict.fromkeys(cues))
                attrs: dict[str, Any] = {"cues": unique}
                bm = P.BENCHMARK_RE.search(sent)
                if bm:
                    attrs["benchmark"] = re.sub(r"\s+", " ", bm.group(1))
                rt = P.RATE_TYPE_RE.search(sent)
                if rt:
                    attrs["rate_type"] = rt.group(1).lower()
                out.append(RawEntity(
                    L.VARIABLE_INTEREST_CONDITION, s, e, "regex_pattern",
                    Normalized("; ".join(unique), None, None, None, {}),
                    Evidence(rule="sentence:variable_rate_cues", trigger_text=", ".join(unique),
                             trigger_source="sentence_preceding"),
                    attrs,
                ))
            core = P.REPAY_CORE_RE.search(sent)
            core_cues = {re.sub(r"\s+", " ", m.group(1).lower()) for m in P.REPAY_CORE_RE.finditer(sent)}
            sched = [re.sub(r"\s+", " ", m.group(1).lower()) for m in P.REPAY_SCHEDULE_RE.finditer(sent)]
            # need two distinct cues: a lone "NACH" (e.g. "NACH bounce charges") is not a repayment term
            if core and sched and len(core_cues | set(sched)) >= 2:
                attrs = {"cues": list(dict.fromkeys(sched))}
                parts: list[str] = []
                due = P.DUE_DAY_RE.search(sent)
                if due:
                    day = parse_ordinal_day(due.group("day"))
                    attrs["due_day_of_month"] = day
                    attrs["frequency"] = "monthly"
                    parts.append(f"due on day {day} of every month")
                elif re.search(r"\b(monthly|every month|each month)\b", sent, re.I):
                    attrs["frequency"] = "monthly"
                    parts.append("monthly")
                modes = [m for m in ("nach", "ecs", "auto-debit", "autodebit", "standing instruction", "post-dated", "postdated", "pdc")
                         if re.search(r"\b" + re.escape(m), sent, re.I)]
                if modes:
                    attrs["payment_modes"] = modes
                    parts.append("mode: " + ", ".join(modes))
                if re.search(r"appropriat", sent, re.I):
                    attrs["appropriation_clause"] = True
                    parts.append("appropriation order")
                out.append(RawEntity(
                    L.REPAYMENT_CONDITION, s, e, "regex_pattern",
                    Normalized("; ".join(parts) or "; ".join(attrs["cues"]), None, None, None, {}),
                    Evidence(rule="sentence:repayment_cues", trigger_text=core.group(0), trigger_source="sentence_preceding"),
                    attrs,
                ))
            for label, regex, rule in (
                (L.DEFAULT_CONDITION, P.DEFAULT_CUES_RE, "sentence:default_cues"),
                (L.EXCLUSION, P.EXCLUSION_CUES_RE, "sentence:exclusion_cues"),
                (L.CANCELLATION_CONDITION, P.CANCELLATION_CUES_RE, "sentence:cancellation_cues"),
                (L.DISCRETIONARY_CHANGE_CONDITION, P.DISCRETION_CUES_RE, "sentence:discretion_cues"),
                (L.SECURITY_CONDITION, P.SECURITY_CUES_RE, "sentence:security_cues"),
            ):
                found = list(dict.fromkeys(re.sub(r"\s+", " ", next(g for g in m.groups() if g).lower())
                                            for m in regex.finditer(sent)))
                if not found or is_title_like(sent):
                    continue
                attrs = {"cues": found}
                if label == L.CANCELLATION_CONDITION:
                    if re.search(r"free[-\s]look", sent, re.I):
                        attrs["free_look"] = True
                    if re.search(r"refund", sent, re.I):
                        attrs["refund_mentioned"] = True
                if label == L.DEFAULT_CONDITION and re.search(r"credit\s+(?:information|bureau)|cibil", sent, re.I):
                    attrs["credit_bureau_reporting"] = True
                out.append(RawEntity(
                    label, s, e, "regex_pattern", Normalized("; ".join(found), None, None, None, {}),
                    Evidence(rule=rule, trigger_text=", ".join(found), trigger_source="sentence_preceding"),
                    attrs,
                ))
        return out

    def _charge_mentions(
        self, text: str, sentences: list[tuple[int, int]], triggers: list[Trigger],
        entities: list[RawEntity], ctx: ClauseContext,
    ) -> list[RawEntity]:
        if ctx.is_heading_clause:
            return []
        out: list[RawEntity] = []
        for s, e in sentences:
            sent = text[s:e]
            if is_title_like(sent) and not re.search(r"[.:;]$", sent):
                continue
            seen: set[EntityLabel] = set()
            for t in triggers:
                if not (s <= t.start < e) or t.label not in _MENTION_LABELS or not P.CHARGE_NOUN_RE.search(t.text):
                    continue
                # "without any foreclosure charges", "no prepayment penalty": the charge does NOT apply
                if re.search(r"\b(?:without(?:\s+any)?|no|nil|free\s+of|waived?|exempt\s+from)\s+(?:\w+\s+)?$",
                             text[max(0, t.start - 25):t.start], re.I):
                    continue
                label = _MENTION_LABELS[t.label]
                if label in seen or any(x.label == label and s <= x.start < e for x in entities):
                    continue
                seen.add(label)
                attrs: dict[str, Any] = {"value_specified": False}
                quals = [re.sub(r"\s+", " ", m.group(1).lower()) for m in P.QUALIFIER_RE.finditer(sent)]
                if quals:
                    attrs["qualifiers"] = list(dict.fromkeys(quals))
                if sent.rstrip().endswith(":"):
                    attrs["details_follow"] = True
                out.append(RawEntity(label, t.start, t.end, "spacy_span_ruler", None,
                                     Evidence(rule="charge_mention_without_value", trigger_text=t.text,
                                              trigger_label=t.label, trigger_source="sentence_preceding"),
                                     attrs))
        return out

    def _custom_model_entities(self, doc, entities: list[RawEntity]) -> list[RawEntity]:
        out: list[RawEntity] = []
        for ent in getattr(doc, "ents", ()):
            if ent.label_ not in self._custom_labels:
                continue
            if any(_overlaps(ent.start_char, ent.end_char, x.start, x.end) for x in entities + out
                   if x.label.value not in SENTENCE_LEVEL_LABELS):
                continue
            out.append(RawEntity(EntityLabel(ent.label_), ent.start_char, ent.end_char, "spacy_custom_model", None,
                                 Evidence(rule="custom_model_prediction", trigger_source="model")))
        return out
