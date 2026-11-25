"""Pydantic output schema for the FinClause extraction module.

Design notes
------------
* **Single source of truth for entities.** Entities are stored once, in
  ``ExtractionResult.entities``. Each :class:`Clause` lists the IDs of its
  entities in ``entity_ids`` and each :class:`Entity` carries its ``clause_id``.
  Use :meth:`ExtractionResult.entities_for_clause` to resolve them.
* **Offsets.** See :data:`finclause_extraction.config.OFFSET_REFERENCE`.
  ``Clause.original_text[e.start_offset:e.end_offset] == e.surface_text`` always
  holds, and ``Clause.cleaned_text[e.cleaned_start_offset:e.cleaned_end_offset]
  == e.cleaned_surface_text``.
* **Coordinates.** Bounding boxes are in PDF points (1/72 inch) with the origin
  at the top-left of the page, as reported by PyMuPDF. OCR coordinates are
  converted to the same space.
* **No fabricated confidence.** The only confidence values reported are
  Tesseract's own per-word OCR confidences.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ExtractionMethod(str, Enum):
    """How a piece of text was obtained."""

    NATIVE = "native"  #: PDF text layer via PyMuPDF
    OCR = "ocr"  #: Tesseract OCR of the rendered page
    TEXT_INPUT = "text_input"  #: plain text passed to ``analyze_text``
    NONE = "none"  #: no text could be obtained for the page


class ExtractionStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"  #: result usable but some content could not be extracted
    FAILED = "failed"  #: document could not be processed (see ``errors``)


class ClauseType(str, Enum):
    SECTION = "section"  #: "Section 7 - Fees and Charges", "Article II", "Schedule A"
    HEADING = "heading"  #: unnumbered heading such as "FEES AND CHARGES"
    CLAUSE = "clause"  #: top-level numbered clause: "1.", "Clause 5"
    SUBCLAUSE = "subclause"  #: "1.1", "1.1.1", "2(a)", "(i)", "a)"
    BULLET = "bullet"  #: bullet list item
    PARAGRAPH = "paragraph"  #: unnumbered paragraph


class EntityLabel(str, Enum):
    """Entity labels. The last four are additions documented in the README."""

    INTEREST_RATE = "INTEREST_RATE"
    PROCESSING_FEE = "PROCESSING_FEE"
    PENALTY = "PENALTY"
    FORECLOSURE_CHARGE = "FORECLOSURE_CHARGE"
    PREPAYMENT_CHARGE = "PREPAYMENT_CHARGE"
    LOAN_AMOUNT = "LOAN_AMOUNT"
    EMI = "EMI"
    TENURE = "TENURE"
    DATE = "DATE"
    OTHER_FINANCIAL_AMOUNT = "OTHER_FINANCIAL_AMOUNT"
    REPAYMENT_CONDITION = "REPAYMENT_CONDITION"
    VARIABLE_INTEREST_CONDITION = "VARIABLE_INTEREST_CONDITION"
    # --- additional value labels ---
    TAX = "TAX"  #: GST / tax rate or amount
    OTHER_PERCENTAGE = "OTHER_PERCENTAGE"  #: percentage whose meaning is not determinable
    DURATION = "DURATION"  #: time period that is not the loan tenure (lock-in, notice period)
    GRACE_PERIOD = "GRACE_PERIOD"  #: period allowed after a due date before consequences apply
    PREMIUM = "PREMIUM"  #: insurance premium
    SUM_INSURED = "SUM_INSURED"  #: insurance cover amount (sum insured / sum assured)
    CO_PAYMENT = "CO_PAYMENT"  #: share of each claim paid by the policyholder
    WAITING_PERIOD = "WAITING_PERIOD"  #: insurance waiting period before cover starts
    COVERAGE_LIMIT = "COVERAGE_LIMIT"  #: insurance sub-limit (e.g. room rent cap)
    # --- additional sentence-level condition labels ---
    DEFAULT_CONDITION = "DEFAULT_CONDITION"  #: consequences of default (recall, legal action, credit bureau)
    EXCLUSION = "EXCLUSION"  #: what is not covered / not payable / not liable
    CANCELLATION_CONDITION = "CANCELLATION_CONDITION"  #: cancellation, refund, free-look, returns
    DISCRETIONARY_CHANGE_CONDITION = "DISCRETIONARY_CHANGE_CONDITION"  #: terms changeable at one party's discretion
    SECURITY_CONDITION = "SECURITY_CONDITION"  #: guarantor, collateral, lien, hypothecation, mortgage


#: Labels whose entity spans a whole sentence (a condition) rather than a single value.
SENTENCE_LEVEL_LABELS = frozenset({
    "REPAYMENT_CONDITION", "VARIABLE_INTEREST_CONDITION", "DEFAULT_CONDITION", "EXCLUSION",
    "CANCELLATION_CONDITION", "DISCRETIONARY_CHANGE_CONDITION", "SECURITY_CONDITION",
})


class BoundingBox(_Model):
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def union(cls, boxes: Iterable["BoundingBox"]) -> Optional["BoundingBox"]:
        """Smallest box enclosing all ``boxes`` (None if empty)."""
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return None
        return cls(
            x0=min(b.x0 for b in boxes),
            y0=min(b.y0 for b in boxes),
            x1=max(b.x1 for b in boxes),
            y1=max(b.y1 for b in boxes),
        )


class Word(_Model):
    text: str
    bbox: BoundingBox
    char_start: int = Field(description="Offset of the word in TextLine.text")
    char_end: int
    ocr_confidence: Optional[float] = Field(
        default=None, description="Tesseract word confidence (0-100); None for native text"
    )


class TextLine(_Model):
    line_id: str
    text: str
    bbox: BoundingBox
    words: list[Word] = Field(default_factory=list)
    font_size: Optional[float] = None
    is_bold: Optional[bool] = None
    is_artifact: bool = Field(
        default=False,
        description="True for running headers/footers/page numbers excluded from clauses",
    )


class TextBlock(_Model):
    block_id: str
    document_id: str
    page_number: int
    text: str = Field(description="Original block text: line texts joined with '\\n'")
    bbox: BoundingBox
    extraction_method: ExtractionMethod
    lines: list[TextLine] = Field(default_factory=list)
    ocr_mean_confidence: Optional[float] = None


class Page(_Model):
    page_number: int = Field(description="1-based page number")
    width: float
    height: float
    rotation: int = 0
    extraction_method: ExtractionMethod
    native_char_count: int = Field(description="Non-whitespace characters in the native text layer")
    has_images: bool = False
    is_empty: bool = Field(description="No text, images or drawings on the page")
    needs_ocr: bool = Field(description="Native text missing or insufficient")
    ocr_applied: bool = False
    assessment_reason: Optional[str] = Field(
        default=None, description="Why the page was treated as scanned/empty (None for good native text)")
    ocr_mean_confidence: Optional[float] = Field(default=None, description="Mean Tesseract word confidence (OCR pages)")
    ocr_word_count: Optional[int] = None
    ocr_deskew_angle: Optional[float] = Field(default=None, description="Rotation applied before OCR (degrees)")
    ocr_preprocessing: list[str] = Field(default_factory=list, description="Image steps applied before OCR")
    text: str = Field(default="", description="Original page text (blocks joined by blank lines)")
    blocks: list[TextBlock] = Field(default_factory=list)


class SourceLocation(_Model):
    """A contiguous part of a clause that comes from one text block."""

    page_number: int
    block_id: str
    line_ids: list[str]
    bbox: Optional[BoundingBox] = None
    start_offset: int = Field(description="Offset into Clause.original_text")
    end_offset: int


class SentenceSpan(_Model):
    """A financially relevant sentence inside a clause."""

    text: str = Field(description="Cleaned sentence text")
    start_offset: int = Field(description="Offset into Clause.original_text")
    end_offset: int
    cleaned_start_offset: int = Field(description="Offset into Clause.cleaned_text")
    cleaned_end_offset: int
    entity_ids: list[str] = Field(default_factory=list)


class Clause(_Model):
    clause_id: str = Field(description="Stable sequential ID (document order)")
    parent_clause_id: Optional[str] = None
    child_clause_ids: list[str] = Field(default_factory=list)
    clause_type: ClauseType
    clause_number: Optional[str] = Field(default=None, description="Marker as written: '1.1', '(a)', 'Section 7'")
    numbering_path: Optional[str] = Field(default=None, description="Full path, e.g. '7.1(a)(ii)'")
    level: int = Field(description="Depth in the clause tree (0 = top level)")
    heading: Optional[str] = Field(default=None, description="Own title or nearest ancestor heading")
    heading_clause_id: Optional[str] = None
    original_text: str
    cleaned_text: str
    page_numbers: list[int]
    source_locations: list[SourceLocation] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    financial_sentences: list[SentenceSpan] = Field(default_factory=list)
    is_financially_relevant: bool = False
    simplified_text: Optional[str] = Field(
        default=None, description="Rule-based plain-language rewrite of cleaned_text (for reading aid only)")


class Evidence(_Model):
    """Why an entity received its label (rule-based evidence, not a probability)."""

    rule: str
    trigger_text: Optional[str] = None
    trigger_label: Optional[str] = None
    trigger_source: Optional[str] = Field(
        default=None,
        description="sentence_preceding | sentence_following | parent_clause | heading | pattern | model",
    )


class Entity(_Model):
    entity_id: str
    label: EntityLabel
    surface_text: str = Field(description="Exact original text: Clause.original_text[start_offset:end_offset]")
    cleaned_surface_text: str
    normalized_value: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    start_offset: int
    end_offset: int
    cleaned_start_offset: int
    cleaned_end_offset: int
    clause_id: str
    page_number: int
    bboxes: list[BoundingBox] = Field(
        default_factory=list, description="One box per source line the entity spans"
    )
    extraction_method: str = Field(
        description="regex_context | regex_pattern | spacy_span_ruler | spacy_ner | spacy_custom_model"
    )
    evidence: Optional[Evidence] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class WarningLocation(_Model):
    """Where on the page a warning applies."""

    page_number: int
    bbox: Optional[BoundingBox] = None
    text: Optional[str] = Field(default=None, description="The text at this location, if any")
    entity_id: Optional[str] = None
    clause_id: Optional[str] = None


class ExtractionWarning(_Model):
    code: str
    message: str = Field(description="What happened, why, and what to do")
    severity: str = Field(default="warning", description="info | warning")
    page_number: Optional[int] = None
    clause_id: Optional[str] = None
    locations: list[WarningLocation] = Field(default_factory=list)


class KeyPoint(_Model):
    """A plain-language point derived (by rules/templates) from extracted entities and clauses.

    Every point links to its source clauses/entities so it can be verified against the
    original text. Points never judge whether a term is fair, legal or fraudulent.
    """

    point_id: str
    category: str = Field(description="costs | interest | repayment | penalties | default | cover | "
                                      "exclusions | cancellation | changes | dates | check")
    text: str
    importance: str = Field(default="info", description="info | attention (worth checking before signing)")
    clause_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_text: str = Field(default="", description="Cleaned source sentence/clause the point is based on")
    method: str = Field(description="template | simplified_sentence")


class ExtractionError(_Model):
    code: str
    message: str


class ArtifactLine(_Model):
    page_number: int
    line_id: str
    text: str
    reason: str


class DocumentInfo(_Model):
    """File-level information about the processed document."""

    document_id: str
    filename: str
    file_size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    page_count: int = 0
    is_encrypted: bool = False
    pdf_metadata: dict[str, str] = Field(default_factory=dict)


class ExtractionMetadata(_Model):
    extraction_version: str
    ocr_enabled: bool
    ocr_used: bool = False
    ocr_pages: list[int] = Field(default_factory=list)
    native_pages: list[int] = Field(default_factory=list)
    empty_pages: list[int] = Field(default_factory=list)
    unextracted_pages: list[int] = Field(
        default_factory=list, description="Pages that needed OCR but OCR was disabled/unavailable/failed"
    )
    tesseract_version: Optional[str] = None
    nlp_pipeline: Optional[str] = Field(default=None, description="spaCy pipeline used, e.g. 'en_core_web_sm 3.8.0'")
    offset_reference: str
    excluded_artifact_lines: list[ArtifactLine] = Field(default_factory=list)
    label_counts: dict[str, int] = Field(default_factory=dict)
    document_type: Optional[str] = Field(
        default=None, description="loan_agreement | key_fact_statement | sanction_letter | insurance_policy | "
                                  "emi_terms | unknown (keyword based)")
    document_type_evidence: list[str] = Field(default_factory=list)
    processing_time_seconds: Optional[float] = None


class ExtractionResult(_Model):
    """Top-level output of :func:`finclause_extraction.analyze_pdf`."""

    document_id: str
    filename: str
    total_pages: int
    extraction_status: ExtractionStatus
    document: Optional[DocumentInfo] = None
    pages: list[Page] = Field(default_factory=list)
    clauses: list[Clause] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    errors: list[ExtractionError] = Field(default_factory=list)
    metadata: ExtractionMetadata

    # ----- convenience accessors (not serialized) -----
    def get_clause(self, clause_id: str) -> Optional[Clause]:
        return next((c for c in self.clauses if c.clause_id == clause_id), None)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return next((e for e in self.entities if e.entity_id == entity_id), None)

    def entities_for_clause(self, clause_id: str) -> list[Entity]:
        return [e for e in self.entities if e.clause_id == clause_id]

    def entities_by_label(self, label: str | EntityLabel) -> list[Entity]:
        value = label.value if isinstance(label, EntityLabel) else label
        return [e for e in self.entities if e.label == value]

    def to_json(self, indent: Optional[int] = 2) -> str:
        return self.model_dump_json(indent=indent)
