"""Plain-language layer: rule-based simplification and key points (deterministic, traceable).

This is a reading aid built only from extracted facts. It is not legal advice and it never
labels a term as unfair, illegal or fraudulent. Richer explanations are Person 3's scope.
"""

from .doc_type import detect_document_type
from .key_points import CATEGORY_ORDER, CATEGORY_TITLES, build_key_points
from .simplifier import simplify_text

__all__ = ["detect_document_type", "build_key_points", "simplify_text", "CATEGORY_ORDER", "CATEGORY_TITLES"]
