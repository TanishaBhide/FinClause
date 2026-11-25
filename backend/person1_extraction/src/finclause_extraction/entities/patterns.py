"""Regular expressions and trigger lexicons for financial entity extraction.

Two kinds of patterns live here:

1. **Value patterns** (regex): money, percentages, basis points, dates and
   durations. These find *what* the number is.
2. **Trigger phrases** (fed to a spaCy ``span_ruler``): words such as
   "processing fee", "penal interest" or "tenure" that tell us *what the number
   means*. Each phrase maps to a trigger label used by the context classifier.
"""

from __future__ import annotations

import re

from .normalizer import MONTHS, ORDINAL_WORDS, WORD_NUMBERS

# --------------------------------------------------------------------------- #
# Value patterns
# --------------------------------------------------------------------------- #
NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?"
SCALE = r"(?:lakhs?|lacs?|crores?|cr\.?|thousand|million|billion|mn)(?![a-z])"
PERIOD = (
    r"(?:per\s+annum|p\.\s?a\.?|pa(?![a-z])|per\s+year|a\s+year|annually|yearly|per\s+month|p\.\s?m\.?"
    r"|a\s+month|monthly|per\s+day|daily|a\s+day|per\s+quarter|quarterly|per\s+half[\s-]year|half[\s-]yearly)"
)
_CUR_PREFIX = r"₹|\bRs(?![A-Za-z])\.?|\bRe(?![A-Za-z])\.?|\bINR(?![A-Za-z])|\bRupees?(?![A-Za-z])"
_CUR_SUFFIX = r"rupees?(?![a-z])|INR(?![A-Za-z])|Rs(?![A-Za-z])\.?"
_LB = r"(?<![\w.,])"

MONEY_PREFIX_RE = re.compile(
    rf"(?P<cur>{_CUR_PREFIX})\s{{0,2}}(?P<num>{NUM})(?:\s{{0,2}}(?P<scale>{SCALE}))?(?P<slash>\s?/-)?", re.I
)
MONEY_SUFFIX_RE = re.compile(
    rf"{_LB}(?P<num>{NUM})\s{{0,2}}(?:(?P<scale>{SCALE})\s{{0,2}})?(?P<cur>{_CUR_SUFFIX})", re.I
)
MONEY_SCALE_RE = re.compile(rf"{_LB}(?P<num>{NUM})\s{{0,2}}(?P<scale>lakhs?|lacs?|crores?)(?![a-z])", re.I)
MONEY_SLASH_RE = re.compile(rf"{_LB}(?P<num>{NUM})\s?(?P<slash>/-)")
PERCENT_RE = re.compile(
    rf"(?<![\w.,])(?P<sign>[-+])?(?P<num>{NUM})\s{{0,2}}(?P<pct>%|percent(?![a-z])|per\s?cent(?![a-z])|pct(?![a-z]))"
    rf"(?:\s{{0,2}}(?P<period>{PERIOD}))?",
    re.I,
)
BPS_RE = re.compile(rf"{_LB}(?P<num>{NUM})\s{{0,2}}(?P<bps>basis\s+points?|bps|bp)(?![a-z])", re.I)

_MONTH_ALT = "|".join(
    sorted({v for k in MONTHS for v in (k.capitalize(), k.upper())}, key=len, reverse=True)
)
DATE_NUMERIC_RE = re.compile(
    r"(?<![\w./-])(?P<d>\d{1,2})(?P<sep>[/-])(?P<m>\d{1,2})(?P=sep)(?P<y>\d{4}|\d{2})(?![\w/-])"
    r"|(?<![\w./-])(?P<d2>\d{1,2})\.(?P<m2>\d{1,2})\.(?P<y2>\d{4})(?![\w.])"
)
DATE_ISO_RE = re.compile(r"(?<![\w/-])(?P<y>(?:19|20)\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?![\w/-])")
DATE_DMY_TEXT_RE = re.compile(
    rf"(?<!\w)(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?P<mon>{_MONTH_ALT})\.?,?\s+(?P<y>\d{{4}})(?!\w)"
)
DATE_MDY_TEXT_RE = re.compile(
    rf"(?<!\w)(?P<mon>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})(?!\w)"
)
DATE_MONTH_YEAR_RE = re.compile(rf"(?<!\w)(?P<mon>{_MONTH_ALT})\.?,?\s+(?P<y>(?:19|20)\d{{2}})(?!\w)")

_NUMWORD_ALT = "|".join(sorted((re.escape(w) for w in WORD_NUMBERS), key=len, reverse=True))
# "EM[Il1]" also accepts the frequent OCR confusions "EMl"/"EM1" for "EMI" (I vs l vs 1).
DURATION_UNIT = r"months?|years?|yrs?|days?|weeks?|quarters?|(?:equated\s+monthly\s+)?instal(?:l)?ments?|EM[Il1]s?"
DURATION_RE = re.compile(
    rf"(?<![\w.,])(?:(?P<word>{_NUMWORD_ALT})\s+(?:\(\s?(?P<paren>\d+)\s?\)\s+)?"
    rf"|(?P<num>\d+(?:\.\d+)?)\s{{0,2}}(?:\(\s?[a-z][a-z\- ]*\s?\)\s+)?)(?P<unit>{DURATION_UNIT})(?![a-z])",
    re.I,
)

_ORD_ALT = "|".join(sorted((re.escape(w) for w in ORDINAL_WORDS), key=len, reverse=True))
DUE_DAY_RE = re.compile(
    rf"(?P<day>\d{{1,2}}(?:st|nd|rd|th)?|{_ORD_ALT}|last)\s+(?:day\s+)?of\s+(?:every|each|the)\s+"
    rf"(?:calendar\s+|english\s+calendar\s+)?month",
    re.I,
)

# --------------------------------------------------------------------------- #
# Context helpers (applied to the text right before/after a value)
# --------------------------------------------------------------------------- #
BASIS_RE = re.compile(
    r"^\s*(?:of|on|over)\s+(?:the\s+|such\s+|any\s+)?(?P<basis>(?:(?:total|entire|outstanding|sanctioned|disbursed|"
    r"overdue|unpaid|principal|loan|prepaid|pre-paid|foreclosed|defaulted|monthly|future|due)\s+){0,3}"
    r"(?:loan\s+amount|loan|principal(?:\s+outstanding|\s+amount)?|outstanding(?:\s+principal|\s+amount|\s+balance)?|"
    r"amount\s+(?:pre-?paid|outstanding|overdue|due)|overdue\s+amount|emi(?:\s+amount)?|instal(?:l)?ment(?:\s+amount)?|"
    r"amount|balance|sum))\b",
    re.I,
)
FOLLOWING_CONNECTOR_RE = re.compile(
    r"^[\s,]*(?:/-\s*)?(?:\([^)]{0,60}\)\s*)?(?:(?:shall|will|would|may)\s+be\s+|is\s+|are\s+)?"
    r"(?:charged\s+|levied\s+|payable\s+|collected\s+|paid\s+|deducted\s+|recovered\s+)?"
    r"(?:as|towards|for|being|by\s+way\s+of)\s+(?:a\s+|an\s+|the\s+)?$",
    re.I,
)
UPPER_BOUND_RE = re.compile(r"(?:up\s?to|not\s+exceeding|maximum(?:\s+of)?|max\.?|at\s+most|capped\s+at)\s*$", re.I)
LOWER_BOUND_RE = re.compile(r"(?:minimum(?:\s+of)?|min\.?|at\s+least|not\s+less\s+than)\s*$", re.I)
PLUS_TAX_RE = re.compile(r"^\s*(?:\+|plus|and)\s*(?:applicable\s+|the\s+applicable\s+)?(?:gst|taxes|tax|service\s+tax)\b", re.I)
INCL_TAX_RE = re.compile(r"\binclusive\s+of\s+(?:all\s+)?(?:applicable\s+)?(?:gst|taxes|tax)\b", re.I)
EXCL_TAX_RE = re.compile(r"\bexclusive\s+of\s+(?:all\s+)?(?:applicable\s+)?(?:gst|taxes|tax)\b", re.I)
FREQUENCY_RE = re.compile(
    r"^\s*(?:/-\s*)?(?:per|for\s+each|for\s+every|each|every)\s+(?P<freq>instance|cheque|bounce|month|day|occasion|"
    r"default|statement|transaction|year|annum|request|dishonour|presentation)\b",
    re.I,
)
FOLLOWING_PERIOD_RE = re.compile(rf"^\s*(?:\([^)]{{0,25}}\)\s*)?(?P<period>{PERIOD})", re.I)

# --------------------------------------------------------------------------- #
# Condition cues
# --------------------------------------------------------------------------- #
VARIABLE_CUES_RE = re.compile(
    r"\b(?P<cue>floating|variable|external\s+benchmark|benchmark(?:\s+rate)?|repo\s+rate|mclr|eblr|rllr|"
    r"base\s+rate|plr|t-?bill|reset|revis(?:e|ed|ion|able)|subject\s+to\s+change|may\s+(?:be\s+)?(?:change[ds]?|var(?:y|ied)|revised)|"
    r"linked\s+to|spread)\b",
    re.I,
)
VARIABLE_STRONG = {"floating", "variable", "external benchmark", "benchmark", "benchmark rate", "repo rate", "mclr",
                   "eblr", "rllr", "base rate", "plr", "reset"}
INTEREST_WORD_RE = re.compile(r"\b(?:interest|rate|roi)\b", re.I)
BENCHMARK_RE = re.compile(r"\b(repo\s+rate|mclr|eblr|rllr|base\s+rate|plr|t-?bill(?:\s+rate)?|external\s+benchmark(?:\s+lending\s+rate)?)\b", re.I)

REPAY_CORE_RE = re.compile(
    r"\b(repay\w*|payable|due|em[il1]s?|instal(?:l)?ments?|nach|ecs|auto-?debit|standing\s+instructions?|"
    r"post-?dated\s+cheques?|pdcs?|appropriat\w+)\b",
    re.I,
)
REPAY_SCHEDULE_RE = re.compile(
    r"\b(day\s+of\s+(?:every|each|the)\s+(?:calendar\s+)?month|every\s+month|each\s+month|monthly|on\s+or\s+before|"
    r"due\s+date|due\s+on|repayment\s+schedule|amorti[sz]ation\s+schedule|nach|ecs|auto-?debit|standing\s+instructions?|"
    r"post-?dated|within\s+\d+\s+days|in\s+equal|appropriat\w+|first\s+towards|in\s+arrears|in\s+advance)\b",
    re.I,
)
CHARGE_NOUN_RE = re.compile(r"charge|fee|penalt|penal|interest", re.I)
QUALIFIER_RE = re.compile(
    r"\b(as\s+per\s+(?:the\s+)?schedule\s+of\s+charges|schedule\s+of\s+charges|subject\s+to\s+(?:applicable\s+)?conditions|"
    r"as\s+applicable|applicable\s+conditions|as\s+may\s+be\s+(?:decided|determined|specified|notified)|"
    r"at\s+the\s+(?:sole\s+)?discretion|prevailing|from\s+time\s+to\s+time|as\s+decided|as\s+notified|"
    r"as\s+per\s+(?:the\s+)?(?:lender'?s?|bank'?s?|company'?s?)\s+policy)\b",
    re.I,
)
RATE_TYPE_RE = re.compile(r"\b(fixed|floating|variable)\b", re.I)

# --------------------------------------------------------------------------- #
# Trigger lexicon (spaCy span_ruler phrase patterns, matched on LOWER)
# --------------------------------------------------------------------------- #
#: Internal trigger labels. They are mapped to entity labels by the classifier.
TRIGGERS: dict[str, list[str]] = {
    "INTEREST": [
        "interest", "interest rate", "rate of interest", "roi", "annual percentage rate", "apr",
        "annualised rate", "annualized rate", "effective interest rate", "spread", "margin over",
        "fixed rate", "floating rate", "interest charge",
    ],
    "PENALTY": [
        "penal interest", "penal charge", "penal charges", "penalty", "penalties", "late payment",
        "late payment charge", "late payment charges", "late payment fee", "late payment fees", "late fee",
        "late fees", "delayed payment", "delayed payment charges", "delayed payment interest", "delay in payment",
        "overdue interest", "overdue charge", "overdue charges", "default interest", "additional interest",
        "bounce charge", "bounce charges", "cheque bounce", "cheque bounce charges", "cheque return charges",
        "dishonour charge", "dishonour charges", "dishonor charges", "nach bounce", "emi bounce", "mandate rejection",
        "penal", "liquidated damages",
    ],
    "PROCESSING_FEE": [
        "processing fee", "processing fees", "processing charge", "processing charges", "login fee",
        "login fees", "origination fee", "upfront fee", "loan processing fee",
    ],
    "FORECLOSURE": [
        "foreclosure", "fore-closure", "fore closure", "foreclose", "foreclosed", "foreclosing",
        "pre-closure", "preclosure", "pre closure", "foreclosure charge", "foreclosure charges",
        "foreclosure fee", "pre-closure charges", "preclosure charges",
    ],
    "PREPAYMENT": [
        "prepayment", "pre-payment", "pre payment", "part prepayment", "part-prepayment", "part payment",
        "part-payment", "prepay", "prepaid", "prepayment charge", "prepayment charges", "prepayment penalty",
        "prepayment fee", "part prepayment charges",
    ],
    "LOAN_AMOUNT": [
        "loan amount", "amount of loan", "amount of the loan", "principal amount", "sanctioned amount",
        "sanction amount", "loan of", "facility of", "disbursed amount", "disbursement amount", "credit limit",
        "loan sanctioned", "loan facility", "amount sanctioned", "amount financed",
    ],
    "EMI": [
        "emi", "emis",
        "eml", "emls", "em1", "em1s",  # OCR confusions of "EMI"/"EMIs" (capital I read as l or 1)
        "equated monthly instalment", "equated monthly instalments", "equated monthly installment",
        "equated monthly installments", "monthly instalment", "monthly instalments", "monthly installment",
        "monthly installments", "instalment", "instalments", "installment", "installments", "instalment amount",
        "installment amount",
    ],
    "TENURE": [
        "tenure", "tenor", "term of the loan", "loan term", "loan tenure", "repayment period", "period of the loan",
        "repayable in", "repayable over", "repaid in", "repaid over", "repayment term", "loan period",
        "repayment tenure",
    ],
    "TAX": ["gst", "goods and services tax", "service tax", "tax", "taxes", "cess", "tds"],
    "OTHER_CHARGE": [
        "documentation charge", "documentation charges", "documentation fee", "stamp duty", "legal charges",
        "legal fee", "insurance premium", "valuation charges", "swap charges", "statement charges",
        "duplicate statement", "noc charges", "cancellation charges", "annual maintenance charge",
        "service charge", "service charges", "convenience fee", "collection charges", "repossession charges",
        "charge", "charges", "fee", "fees",
    ],
}

#: Generic trigger words whose meaning can be overridden by a specific trigger that follows.
GENERIC_TRIGGERS = {"charge", "charges", "fee", "fees"}
