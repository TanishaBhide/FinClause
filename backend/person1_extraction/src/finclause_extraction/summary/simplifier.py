"""Rule-based legal-to-plain-English rewriting.

This is a deterministic reading aid, not a translation model: it applies an ordered
list of documented substitutions (legalese -> everyday words, party names -> "you",
short glossary notes for jargon) and then fixes verb agreement for "you".
Numbers, amounts, percentages and dates are never changed: every rule only matches
words, and a final check verifies that all numeric tokens survive unchanged.
"""

from __future__ import annotations

import re
from typing import Callable, Union

Repl = Union[str, Callable[[re.Match], str]]

_I = re.IGNORECASE

#: Ordered (pattern, replacement) rules. Applied top to bottom.
_RULES: list[tuple[re.Pattern, Repl]] = [
    # --- remove boilerplate ---
    (re.compile(r"\bnotwithstanding anything (?:to the contrary )?contained (?:herein|in this (?:agreement|policy|document))"
                r"(?: or elsewhere)?(?: to the contrary)?,?\s*", _I), ""),
    (re.compile(r"\s*\((?:hereinafter|herein after)[^)]*\)", _I), ""),
    (re.compile(r",?\s*(?:hereinafter|herein after)\s+(?:referred\s+to\s+as|called)\s+[\"'][^\"']+[\"']", _I), ""),
    (re.compile(r"\b(?:and/or|and / or)\b", _I), "or"),
    (re.compile(r"\bit is hereby (?:agreed|clarified) (?:and declared )?that\s+", _I), ""),
    (re.compile(r"\bfor the avoidance of doubt,?\s*", _I), ""),
    # --- possessives and parties -> "you"/"your" ---
    (re.compile(r"\b(?:the\s+)?(?:Borrower|Co-Borrower|Applicant|Insured Person|Insured|Policy ?holder|Proposer|"
                r"Customer|Buyer|Cardholder|Card Member)s?['’]s?\b", _I), "your"),
    (re.compile(r"\bthe\s+(?:said\s+)?(?:Borrower|Co-Borrower|Applicant|Insured Person|Insured|Policy ?holder|Proposer|"
                r"Customer|Buyer|Cardholder|Card Member)s?\b", _I), "you"),
    (re.compile(r"\b([Tt])he\s+(Lender|Bank|Company|Insurer|NBFC|Seller|Merchant|Platform|Issuing Bank|Financier)\b"),
     lambda m: m.group(1) + "he " + m.group(2).lower()),
    # "the credit score of you" -> "your credit score"
    (re.compile(r"\bthe ((?:\w+ ){0,2}\w+) of you\b"), r"your \1"),
    (re.compile(r"\bany ((?:\w+ )?\w+) of you\b"), r"any of your \1"),
    # --- legal phrases ---
    (re.compile(r"\bin the event that\b", _I), "if"),
    (re.compile(r"\b(?:in the event of|in case of)\b", _I), "if there is"),
    (re.compile(r"\bprior to\b", _I), "before"),
    (re.compile(r"\bsubsequent to\b", _I), "after"),
    (re.compile(r"\bpursuant to\b", _I), "under"),
    (re.compile(r"\bin accordance with\b", _I), "as per"),
    (re.compile(r"\b(?:in respect of|with respect to|in relation to)\b", _I), "for"),
    (re.compile(r"\bin lieu of\b", _I), "instead of"),
    (re.compile(r"\binter alia\b", _I), "among other things"),
    (re.compile(r"\b(?:forthwith|with immediate effect)\b", _I), "immediately"),
    (re.compile(r"\b(?:on account of|by virtue of)\b", _I), "because of"),
    (re.compile(r"\bfor the purpose of\b", _I), "for"),
    (re.compile(r"\bin addition to\b", _I), "on top of"),
    (re.compile(r"\bat (?:its|their) sole (?:and absolute )?discretion\b", _I), "entirely at its own choice"),
    (re.compile(r"\bsole (?:and absolute )?discretion\b", _I), "own choice"),
    (re.compile(r"\bwithout (?:any )?(?:prior )?(?:notice|intimation)\b", _I), "without telling you first"),
    (re.compile(r"\breserves? the right to\b", _I), "may"),
    (re.compile(r"\b(?:shall|will) be required to\b|\bis required to\b", _I), "must"),
    (re.compile(r"\bshall be liable to pay\b", _I), "must pay"),
    (re.compile(r"\bshall be liable to\b", _I), "may have to"),
    (re.compile(r"\b(?:shall|will) be entitled to\b|\bis entitled to\b", _I), "can"),
    (re.compile(r"\bundertakes? to\b", _I), "agree to"),
    (re.compile(r"\backnowledges? and agrees?\b", _I), "agree"),
    (re.compile(r"\b(?:shall|will) not be liable to make any payment for\b", _I), "will not pay for"),
    (re.compile(r"\b(?:shall|will) not be liable (?:to|for)\b", _I), "will not be responsible for"),
    (re.compile(r"\bshall fall due\b", _I), "is due"),
    (re.compile(r"\bshall become\b", _I), "becomes"),
    (re.compile(r"\bshall constitute\b", _I), "counts as"),
    (re.compile(r"\bshall have\b", _I), "has"),
    (re.compile(r"\bshall apply\b", _I), "applies"),
    (re.compile(r"\bshall carry\b", _I), "carries"),
    (re.compile(r"\bshall attract\b", _I), "attracts"),
    (re.compile(r"\bshall not\b", _I), "will not"),
    (re.compile(r"\bshall be\b", _I), "will be"),
    (re.compile(r"\bshall\b", _I), "must"),
    (re.compile(r"\bhereunder\b", _I), "under this document"),
    (re.compile(r"\bherein\b", _I), "in this document"),
    (re.compile(r"\bhereof\b", _I), "of this document"),
    (re.compile(r"\bhereby\s+", _I), ""),
    (re.compile(r"\bthereof\b", _I), "of it"),
    (re.compile(r"\bthereon\b", _I), "on it"),
    (re.compile(r"\btherein\b", _I), "in it"),
    (re.compile(r"\bthe aforesaid\b|\bthe said\b", _I), "the"),
    (re.compile(r"\baforesaid\b", _I), "above"),
    (re.compile(r"\bcommencement\b", _I), "start"),
    (re.compile(r"\bcommence(?:s|d)?\b", _I), "start"),
    (re.compile(r"\btermination\b", _I), "ending"),
    (re.compile(r"\butili[sz]e(?:d)?\b", _I), "use"),
    (re.compile(r"\bremit(?:ted)?\b", _I), "pay"),
    (re.compile(r"\blevied\b", _I), "charged"),
    (re.compile(r"\blevy\b", _I), "charge"),
    (re.compile(r"\bdisburs(?:al|ement)\b", _I), "payout"),
    (re.compile(r"\bdisbursed\b", _I), "paid out"),
    (re.compile(r"\bavail(?:s|ed)?\b", _I), "take"),
    (re.compile(r"\bintimation\b", _I), "notice"),
    (re.compile(r"\bintimate\b", _I), "inform"),
    (re.compile(r"\bappropriated\b", _I), "adjusted"),
]

#: Jargon explained once per text: (pattern, note appended after the first match).
_GLOSSARY: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bforeclos(?:ure|e|ed|ing)\b|\bpre-?closure\b", _I), "closing the loan early"),
    (re.compile(r"\b(?:part[-\s])?pre-?payment\b", _I), "paying back part of the loan early"),
    (re.compile(r"\bpenal interest\b|\bpenal charges?\b", _I), "extra charge for paying late"),
    (re.compile(r"\bco-?payment\b|\bco-pay\b", _I), "the share of each claim you pay yourself"),
    (re.compile(r"\bsum insured\b|\bsum assured\b", _I), "the maximum the insurer will pay"),
    (re.compile(r"\bpre-existing (?:diseases?|conditions?|illness)\b", _I), "illness you already had before buying the policy"),
    (re.compile(r"\bwaiting period\b", _I), "time before this cover starts"),
    (re.compile(r"\bfree[-\s]look period\b", _I), "time in which you can cancel for a refund"),
    (re.compile(r"\bmoratorium\b", _I), "a break from payments"),
    (re.compile(r"\bhypothecat\w+\b", _I), "pledged as security"),
    (re.compile(r"\bNACH\b"), "automatic debit from your bank account"),
    (re.compile(r"\brepo rate\b", _I), "RBI's benchmark rate"),
    (re.compile(r"\bMCLR\b"), "the bank's benchmark lending rate"),
    (re.compile(r"\bcredit information compan(?:y|ies)\b|\bcredit bureaus?\b|\bCIBIL\b", _I), "agencies that keep your credit score"),
    (re.compile(r"\bset[-\s]?off\b", _I), "using money in your other accounts to recover dues"),
    (re.compile(r"\bno[-\s]cost EMI\b", _I), "interest is given back to you as a discount"),
]

# Only where "you" is the subject: sentence start or after a conjunction/punctuation.
_YOU_VERB = re.compile(
    r"(?:(?<=^)|(?<=[.;:,] )|(?<=\b[Ii]f )|(?<=\bthat )|(?<=\b[Ww]hen )|(?<=\b[Uu]nless )|(?<=\band )|(?<=\b[Ww]here ))"
    r"(you|You) (is|has|was|does|agrees|acknowledges|confirms|fails|pays|defaults|wishes|opts|requests|chooses|"
    r"makes|needs|intends|desires|repays|prepays|forecloses|cancels|returns|purchases|buys|submits|receives|"
    r"continues|remains|holds|keeps|takes|gets|uses|signs|accepts|seeks|claims|applies)\b"
)
_IRREGULAR = {"is": "are", "has": "have", "was": "were", "does": "do"}
_MARKER = re.compile(r"^\s*(?:(?:section|clause|article)\s+\w+\s*[.:-]\s+|\d+(?:\.\d+)*\.?\s+|\(?[a-z]{1,4}\)\s+|[•\-*]\s+)", _I)
_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _verb_fix(m: re.Match) -> str:
    verb = m.group(2)
    if verb in _IRREGULAR:
        base = _IRREGULAR[verb]
    elif verb.endswith("ies"):
        base = verb[:-3] + "y"
    elif verb.endswith(("ches", "shes", "sses", "xes")):
        base = verb[:-2]
    else:
        base = verb[:-1]
    return f"{m.group(1)} {base}"


def _capitalise_sentences(text: str) -> str:
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def simplify_text(text: str, glossary: bool = True, strip_marker: bool = True) -> str:
    """Rewrite legal text in plain English (see module docstring).

    Args:
        text: Cleaned clause/sentence text.
        glossary: Append short explanations after the first use of jargon terms.
        strip_marker: Drop a leading clause number such as ``3.2`` or ``(a)``.
    """
    if not text:
        return text
    out = _MARKER.sub("", text, count=1) if strip_marker else text
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out)
    out = _YOU_VERB.sub(_verb_fix, out)
    out = re.sub(r"\b(you|You) has\b", r"\1 have", out)  # "You has" left by 'shall have' rules
    if glossary:
        for pattern, note in _GLOSSARY:
            m = pattern.search(out)
            if m and f"({note})" not in out:
                out = out[:m.end()] + f" ({note})" + out[m.end():]
    out = re.sub(r"\byou you\b", "you", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:])", r"\1", out).strip()
    out = _capitalise_sentences(out)
    # Safety: numbers must be preserved exactly; otherwise fall back to the input.
    if _NUMBER_TOKEN.findall(out) != _NUMBER_TOKEN.findall(_MARKER.sub("", text, count=1) if strip_marker else text):
        return text
    return out
