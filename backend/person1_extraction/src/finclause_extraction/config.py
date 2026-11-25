"""Configuration for the FinClause extraction pipeline.

All tunable thresholds live in :class:`ExtractionConfig`. The dataclass is frozen
(and therefore hashable) so that a configured pipeline can be cached and reused
across requests without reloading the spaCy model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

EXTRACTION_VERSION = "1.1.0"

#: Environment variable that may point at the Tesseract executable
#: (e.g. ``C:\\Program Files\\Tesseract-OCR\\tesseract.exe``) when it is not on PATH.
TESSERACT_CMD_ENV = "FINCLAUSE_TESSERACT_CMD"

#: Human readable description of how offsets are defined in the output.
OFFSET_REFERENCE = (
    "Entity.start_offset/end_offset and SentenceSpan.start_offset/end_offset are "
    "0-based, end-exclusive character offsets into Clause.original_text. "
    "The cleaned_start_offset/cleaned_end_offset fields index into Clause.cleaned_text. "
    "SourceLocation.start_offset/end_offset index into Clause.original_text. "
    "Word.char_start/char_end index into TextLine.text."
)


@dataclass(frozen=True)
class ExtractionConfig:
    """Tunable settings for every stage of the pipeline.

    Attributes:
        use_ocr: Run OCR on pages whose native text layer is missing or insufficient.
        force_ocr: OCR every page even if native text is good (useful for broken
            text layers). Ignored when ``use_ocr`` is False.
        min_native_chars: A page with fewer non-whitespace native characters than
            this is treated as having insufficient text.
        max_garbage_ratio: If more than this fraction of native characters are
            unmappable glyphs (U+FFFD / private-use), the text layer is unusable.
        ocr_dpi: Rendering resolution used for OCR.
        ocr_language: Tesseract language code(s), e.g. ``"eng"`` or ``"eng+hin"``.
        tesseract_cmd: Explicit path to the tesseract executable. Falls back to the
            ``FINCLAUSE_TESSERACT_CMD`` environment variable, then PATH.
        use_spacy: Use spaCy for trigger matching and NER fallback. When False the
            extractor uses a blank English tokenizer (no pretrained model needed).
        spacy_model: Name or filesystem path of the spaCy pipeline to load. A custom
            trained model can be plugged in here (see README).
        include_words: Include word-level bounding boxes in ``pages`` output.
        date_order: How to read ambiguous numeric dates such as 05/06/2024:
            ``"DMY"`` (Indian convention, default) or ``"MDY"``.
        detect_headers_footers: Exclude repeated running headers/footers and
            page-number lines from clause segmentation (they stay in ``pages``).
        header_footer_margin_ratio: Fraction of page height treated as the
            header/footer band.
        context_window_chars: Max distance (characters) between a financial value
            and the trigger phrase that gives it meaning.
        low_ocr_confidence: OCR words below this confidence are flagged with locations.
        generate_summary: Build ``key_points`` and ``Clause.simplified_text``.
    """

    use_ocr: bool = True
    force_ocr: bool = False
    min_native_chars: int = 25
    max_garbage_ratio: float = 0.30
    ocr_dpi: int = 300
    ocr_language: str = "eng"
    #: Clean the page image before OCR (contrast, speckle removal, deskew, binarisation).
    ocr_preprocess: bool = True
    tesseract_cmd: Optional[str] = None
    use_spacy: bool = True
    spacy_model: str = "en_core_web_sm"
    include_words: bool = True
    date_order: str = "DMY"
    detect_headers_footers: bool = True
    header_footer_margin_ratio: float = 0.08
    context_window_chars: int = 150
    #: OCR words below this Tesseract confidence (0-100) are reported in a LOW_OCR_CONFIDENCE warning.
    low_ocr_confidence: float = 60.0
    #: Generate plain-language key points and simplified clause text.
    generate_summary: bool = True

    def resolved_tesseract_cmd(self) -> Optional[str]:
        """Return the explicitly configured tesseract path, if any."""
        return self.tesseract_cmd or os.environ.get(TESSERACT_CMD_ENV) or None
