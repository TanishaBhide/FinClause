"""Keyword-based document type detection (transparent: returns the matched keywords)."""

from __future__ import annotations

import re

#: type -> (keywords, weight). The type with the highest weighted score wins.
_TYPES: dict[str, list[tuple[str, int]]] = {
    "key_fact_statement": [(r"key fact(?:s)? statement", 5), (r"\bkfs\b", 4), (r"most important terms", 4),
                           (r"annual percentage rate", 2)],
    "sanction_letter": [(r"sanction letter", 5), (r"we are pleased to (?:inform|sanction|convey)", 4),
                        (r"\bsanctioned\b", 2), (r"letter of (?:offer|sanction)", 4)],
    "insurance_policy": [(r"sum insured|sum assured", 4), (r"\binsurer\b", 2), (r"\bpolicy(?:holder)?\b", 1),
                         (r"\bclaims?\b", 1), (r"\bpremium\b", 2), (r"waiting period", 3), (r"\bexclusions?\b", 2)],
    "emi_terms": [(r"no[-\s]cost emi", 5), (r"\bemi option", 3), (r"\border\b", 1), (r"\bcheckout\b", 3),
                  (r"\bmerchant\b|\bseller\b", 2), (r"credit card", 2)],
    "loan_agreement": [(r"loan agreement", 5), (r"\bborrower\b", 2), (r"\blender\b", 2), (r"\bloan\b", 1)],
}


def detect_document_type(text: str) -> tuple[str, list[str]]:
    """Return ``(document_type, evidence_keywords)``; ``"unknown"`` if nothing matches."""
    lowered = text.lower()
    best, best_score, best_evidence = "unknown", 0, []
    for doc_type, keywords in _TYPES.items():
        score, evidence = 0, []
        for pattern, weight in keywords:
            hits = len(re.findall(pattern, lowered))
            if hits:
                score += weight * min(hits, 3)
                evidence.append(re.findall(pattern, lowered)[0] if isinstance(re.findall(pattern, lowered)[0], str) else pattern)
        if score > best_score:
            best, best_score, best_evidence = doc_type, score, evidence
    return (best, best_evidence) if best_score >= 4 else ("unknown", best_evidence)
