"""End-to-end extraction pipeline and public entry points.

    PDF validation -> native extraction -> scanned-page detection -> OCR (if needed)
    -> header/footer detection -> clause segmentation (+ cleaning)
    -> entity extraction + normalisation -> traceability mapping -> Pydantic result

Public functions:
    * :func:`analyze_pdf` - path to a PDF
    * :func:`analyze_pdf_bytes` - raw bytes (e.g. a FastAPI ``UploadFile``)
    * :func:`analyze_text` - plain text (no PDF), mainly for testing/experiments

None of these raise for bad input by default; they return an
``ExtractionResult`` with ``extraction_status="failed"`` and ``errors`` filled.
Pass ``raise_on_error=True`` to get exceptions instead.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import pymupdf

from .config import EXTRACTION_VERSION, OFFSET_REFERENCE, ExtractionConfig
from .entities.ner import ClauseContext, FinancialEntityExtractor, RawEntity
from .pdf.extractor import (
    PDFExtractionError,
    assess_page,
    build_document_info,
    build_page,
    default_document_id,
    extract_native_blocks,
    open_pdf,
    read_pdf_bytes,
)
from .pdf.ocr import OCRProcessingError, OCRUnavailableError, check_tesseract, ocr_page_with_info
from .preprocessing.segmenter import SegmentedClause, segment_pages
from .schemas import (
    BoundingBox,
    ClauseType,
    DocumentInfo,
    Entity,
    ExtractionError,
    ExtractionMetadata,
    ExtractionMethod,
    ExtractionResult,
    ExtractionStatus,
    ExtractionWarning,
    Page,
    SentenceSpan,
    WarningLocation,
)
from .summary import build_key_points, detect_document_type, simplify_text
from .text_input import pages_from_text

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class ExtractionPipeline:
    """Reusable pipeline. Holds the (expensive) spaCy pipeline between calls."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._extractor: Optional[FinancialEntityExtractor] = None

    @property
    def extractor(self) -> FinancialEntityExtractor:
        if self._extractor is None:
            logger.info("Loading NLP pipeline (spaCy model: %s)", self.config.spacy_model if self.config.use_spacy else "none")
            self._extractor = FinancialEntityExtractor(self.config)
        return self._extractor

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #
    def analyze_pdf(
        self,
        source: Union[PathLike, bytes],
        *,
        filename: Optional[str] = None,
        document_id: Optional[str] = None,
        password: Optional[str] = None,
        raise_on_error: bool = False,
    ) -> ExtractionResult:
        """Run the full pipeline on a PDF path or PDF bytes."""
        started = time.perf_counter()
        name = filename or (Path(source).name if not isinstance(source, (bytes, bytearray)) else "document.pdf")
        doc_id = document_id or "doc_unknown"
        try:
            data, detected = read_pdf_bytes(source)
            name = filename or detected
            doc_id = document_id or default_document_id(data)
            logger.info("Processing %s (%d bytes) as %s", name, len(data), doc_id)
            doc = open_pdf(data, name, password)
        except PDFExtractionError as exc:
            logger.error("%s: %s", exc.code, exc.message)
            if raise_on_error:
                raise
            return self._failed(doc_id, name, exc.code, exc.message, started)

        try:
            info = build_document_info(doc, data, name, doc_id)
            warnings: list[ExtractionWarning] = []
            pages, ocr_state = self._extract_pages(doc, doc_id, warnings)
            return self._finish(pages, info, warnings, ocr_state, started)
        except Exception as exc:  # unexpected: never crash the caller by default
            logger.exception("Unexpected failure while processing %s", name)
            if raise_on_error:
                raise
            return self._failed(doc_id, name, "INTERNAL_ERROR", f"{exc.__class__.__name__}: {exc}", started)
        finally:
            doc.close()

    def analyze_text(
        self, text: str, *, document_id: str = "doc_text", filename: str = "text_input.txt"
    ) -> ExtractionResult:
        """Run segmentation + entity extraction on plain text (no PDF/OCR).

        Pages are separated by form feeds (``\\f``), paragraphs by blank lines.
        Bounding boxes in the result are synthetic (see ``text_input`` module).
        """
        started = time.perf_counter()
        pages = pages_from_text(text, document_id)
        info = DocumentInfo(document_id=document_id, filename=filename, page_count=len(pages))
        warnings = [ExtractionWarning(code="SYNTHETIC_COORDINATES",
                                      message="Input was plain text; bounding boxes are synthetic layout positions.")]
        state = {"ocr_pages": [], "native_pages": [], "empty_pages": [p.page_number for p in pages if p.is_empty],
                 "unextracted_pages": [], "tesseract_version": None}
        return self._finish(pages, info, warnings, state, started)

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #
    def _extract_pages(self, doc: pymupdf.Document, document_id: str, warnings: list[ExtractionWarning]):
        cfg = self.config
        state: dict = {"ocr_pages": [], "native_pages": [], "empty_pages": [], "unextracted_pages": [],
                       "tesseract_version": None}
        ocr_available: Optional[bool] = None
        pages: list[Page] = []

        for index in range(doc.page_count):
            number = index + 1
            try:
                page = doc[index]
                blocks = extract_native_blocks(page, number, document_id, include_words=True)
                assessment = assess_page(page, blocks, cfg)
            except Exception as exc:
                logger.exception("Page %d could not be read", number)
                warnings.append(ExtractionWarning(code="PAGE_EXTRACTION_FAILED", message=str(exc), page_number=number))
                state["unextracted_pages"].append(number)
                pages.append(Page(page_number=number, width=0, height=0, extraction_method=ExtractionMethod.NONE,
                                  native_char_count=0, is_empty=False, needs_ocr=False))
                continue

            want_ocr = cfg.use_ocr and (assessment.needs_ocr or (cfg.force_ocr and not assessment.is_empty))
            method, ocr_applied = ExtractionMethod.NATIVE, False
            whole_page = [WarningLocation(page_number=number, bbox=BoundingBox(
                x0=0, y0=0, x1=round(page.rect.width, 2), y1=round(page.rect.height, 2)))]

            def warn(code: str, message: str, severity: str = "warning") -> None:
                warnings.append(ExtractionWarning(code=code, message=message, severity=severity,
                                                  page_number=number, locations=whole_page))

            if assessment.is_empty:
                state["empty_pages"].append(number)
                warn("EMPTY_PAGE", "This page has no text, images or drawings, so nothing was extracted from it. "
                                   "If the printed document has content here, the PDF may be incomplete.", "info")
            elif assessment.needs_ocr and not cfg.use_ocr:
                state["unextracted_pages"].append(number)
                warn("OCR_DISABLED", f"This page looks scanned ({assessment.reason}), but OCR is switched off, so its "
                                     "text was NOT read and any terms on it are missing from the results. "
                                     "Turn OCR on to read it.")
            elif 0 < assessment.native_char_count < cfg.min_native_chars and not assessment.needs_ocr:
                warn("LOW_TEXT_PAGE", f"Only {assessment.native_char_count} characters of text were found on this page. "
                                      "Check that no content is missing.", "info")

            if want_ocr:
                if ocr_available is None:
                    try:
                        state["tesseract_version"] = check_tesseract(cfg)
                        ocr_available = True
                    except OCRUnavailableError as exc:
                        ocr_available = False
                        warnings.append(ExtractionWarning(code="OCR_UNAVAILABLE", message=str(exc)))
                if ocr_available:
                    try:
                        logger.info("Running OCR on page %d (%s)", number, assessment.reason or "force_ocr")
                        ocr_result = ocr_page_with_info(page, number, document_id, replace(cfg, include_words=True))
                        ocr_blocks = ocr_result.blocks
                        ocr_chars = sum(len("".join(b.text.split())) for b in ocr_blocks)
                        if assessment.needs_ocr or ocr_chars > assessment.native_char_count:
                            blocks, method, ocr_applied = ocr_blocks, ExtractionMethod.OCR, True
                        else:
                            warn("OCR_NOT_USED", "OCR was run but produced less text than the PDF's own text layer, "
                                                 "so the text layer was kept.", "info")
                    except (OCRProcessingError, OCRUnavailableError) as exc:
                        warn(getattr(exc, "code", "OCR_FAILED"), str(exc))
                if assessment.needs_ocr and not ocr_applied:
                    state["unextracted_pages"].append(number)
                    if ocr_available is False:
                        warn("PAGE_NOT_EXTRACTED", f"This page looks scanned ({assessment.reason}) but Tesseract OCR is "
                                                   "not installed, so its text was not read. See the OCR_UNAVAILABLE "
                                                   "warning for installation steps.")

            if ocr_applied:
                state["ocr_pages"].append(number)
            elif blocks:
                state["native_pages"].append(number)
            if not blocks and not assessment.is_empty and not assessment.needs_ocr:
                method = ExtractionMethod.NONE
            if assessment.needs_ocr and not ocr_applied:
                method = ExtractionMethod.NATIVE if blocks else ExtractionMethod.NONE
            built = build_page(page, number, blocks, assessment, method, ocr_applied)
            built.assessment_reason = assessment.reason
            if ocr_applied:
                built.ocr_deskew_angle = ocr_result.deskew_angle
                built.ocr_preprocessing = ocr_result.preprocessing
                confs = [w.ocr_confidence for b in blocks for l in b.lines for w in l.words if w.ocr_confidence is not None]
                built.ocr_word_count = sum(len(l.words) for b in blocks for l in b.lines)
                built.ocr_mean_confidence = round(sum(confs) / len(confs), 1) if confs else None
            pages.append(built)
        return pages, state

    def _finish(self, pages: list[Page], info: DocumentInfo, warnings: list[ExtractionWarning],
                state: dict, started: float) -> ExtractionResult:
        cfg = self.config
        extractor = self.extractor
        for code, message in extractor.load_warnings:
            warnings.append(ExtractionWarning(code=code, message=message))

        segmentation = segment_pages(pages, cfg)
        segs = segmentation.clauses
        line_boxes = {l.line_id: l.bbox for p in pages for b in p.blocks for l in b.lines}
        if segmentation.artifacts:
            warnings.append(ExtractionWarning(
                code="HEADER_FOOTER_EXCLUDED", severity="info",
                message=f"{len(segmentation.artifacts)} running header/footer or page-number lines were left out of the "
                        "clauses so they do not break clauses that continue across pages. They are still in 'pages'.",
                locations=[WarningLocation(page_number=a.page_number, bbox=line_boxes.get(a.line_id), text=a.text)
                           for a in segmentation.artifacts]))
        for page in pages:
            if not page.ocr_applied:
                continue
            low = [(w, l) for b in page.blocks for l in b.lines for w in l.words
                   if w.ocr_confidence is not None and w.ocr_confidence < cfg.low_ocr_confidence and any(c.isalnum() for c in w.text)]
            if low:
                warnings.append(ExtractionWarning(
                    code="LOW_OCR_CONFIDENCE", page_number=page.page_number,
                    message=f"{len(low)} word(s) on page {page.page_number} were read by OCR with confidence below "
                            f"{cfg.low_ocr_confidence:g}%. They may be misread (e.g. 'I' vs 'l', '0' vs 'O'); "
                            "check them against the scan, especially numbers.",
                    locations=[WarningLocation(page_number=page.page_number, bbox=w.bbox,
                                               text=f"{w.text} ({w.ocr_confidence:g}%)") for w, _ in low]))

        by_id = {s.clause.clause_id: s.clause for s in segs}
        contexts = [
            ClauseContext(
                text=s.clause.cleaned_text,
                heading=s.clause.heading,
                parent_text=by_id[s.clause.parent_clause_id].cleaned_text if s.clause.parent_clause_id else None,
                is_heading_clause=s.clause.clause_type in (ClauseType.HEADING.value, ClauseType.SECTION.value),
                clause_number=s.clause.clause_number,
            )
            for s in segs
        ]
        extractions = extractor.extract_batch(contexts)

        entities: list[Entity] = []
        for seg, extraction in zip(segs, extractions):
            clause_entities = [self._to_entity(raw, seg, len(entities) + i + 1) for i, raw in enumerate(extraction.entities)]
            entities.extend(clause_entities)
            seg.clause.entity_ids = [e.entity_id for e in clause_entities]
            sentences: list[SentenceSpan] = []
            for s, e in extraction.sentences:
                ids = [x.entity_id for x in clause_entities
                       if x.cleaned_start_offset < e and s < x.cleaned_end_offset]
                has_trigger = any(s <= t.start < e for t in extraction.triggers)
                if ids or has_trigger:
                    o_s, o_e = seg.cleaned.to_original_span(s, e)
                    sentences.append(SentenceSpan(text=seg.clause.cleaned_text[s:e], start_offset=o_s, end_offset=o_e,
                                                  cleaned_start_offset=s, cleaned_end_offset=e, entity_ids=ids))
            seg.clause.financial_sentences = sentences
            seg.clause.is_financially_relevant = bool(clause_entities or sentences)

        if not segs:
            warnings.append(ExtractionWarning(code="NO_TEXT_EXTRACTED", message="No text could be extracted from the document."))

        def entity_location(e: Entity) -> WarningLocation:
            return WarningLocation(page_number=e.page_number, bbox=BoundingBox.union(e.bboxes),
                                   text=e.surface_text, entity_id=e.entity_id, clause_id=e.clause_id)

        ambiguous = [e for e in entities if e.attributes.get("date_order_assumed")]
        if ambiguous:
            warnings.append(ExtractionWarning(
                code="AMBIGUOUS_DATE",
                message=f"{len(ambiguous)} date(s) such as '{ambiguous[0].surface_text}' could be read day-first or "
                        f"month-first. They were read as {cfg.date_order} (Indian convention). Confirm the intended date.",
                locations=[entity_location(e) for e in ambiguous]))
        unknown = [e for e in entities if e.label in ("OTHER_PERCENTAGE", "OTHER_FINANCIAL_AMOUNT")
                   and not e.attributes.get("related_to")]
        if unknown:
            warnings.append(ExtractionWarning(
                code="UNCLASSIFIED_VALUE", severity="info",
                message=f"{len(unknown)} amount(s)/percentage(s) were found, but the surrounding words do not say what "
                        "they are for, so they were given a generic label instead of guessing. Read these clauses manually.",
                locations=[entity_location(e) for e in unknown]))

        doc_type, doc_evidence = detect_document_type("\n".join(s.clause.cleaned_text for s in segs))
        key_points = []
        if cfg.generate_summary:
            for seg in segs:
                if seg.clause.clause_type not in (ClauseType.HEADING.value, ClauseType.SECTION.value):
                    seg.clause.simplified_text = simplify_text(seg.clause.cleaned_text)
            key_points = build_key_points([s.clause for s in segs], entities)

        if not cfg.include_words:
            for page in pages:
                for block in page.blocks:
                    for line in block.lines:
                        line.words = []

        status = ExtractionStatus.PARTIAL if state["unextracted_pages"] or not segs else ExtractionStatus.SUCCESS
        metadata = ExtractionMetadata(
            extraction_version=EXTRACTION_VERSION,
            ocr_enabled=cfg.use_ocr,
            ocr_used=bool(state["ocr_pages"]),
            ocr_pages=state["ocr_pages"],
            native_pages=state["native_pages"],
            empty_pages=state["empty_pages"],
            unextracted_pages=sorted(set(state["unextracted_pages"])),
            tesseract_version=state["tesseract_version"],
            nlp_pipeline=extractor.nlp_description,
            offset_reference=OFFSET_REFERENCE,
            excluded_artifact_lines=segmentation.artifacts,
            label_counts=dict(sorted(Counter(e.label for e in entities).items())),
            document_type=doc_type,
            document_type_evidence=doc_evidence,
            processing_time_seconds=round(time.perf_counter() - started, 3),
        )
        result = ExtractionResult(
            document_id=info.document_id,
            filename=info.filename,
            total_pages=len(pages),
            extraction_status=status,
            document=info,
            pages=pages,
            clauses=[s.clause for s in segs],
            entities=entities,
            key_points=key_points,
            warnings=warnings,
            metadata=metadata,
        )
        logger.info("Extracted %d clauses and %d entities from %d pages (status=%s)",
                    len(result.clauses), len(entities), len(pages), status.value)
        # Re-validate the assembled object graph (models were mutated after creation).
        return ExtractionResult.model_validate(result.model_dump())

    @staticmethod
    def _to_entity(raw: RawEntity, seg: SegmentedClause, number: int) -> Entity:
        clause = seg.clause
        o_start, o_end = seg.cleaned.to_original_span(raw.start, raw.end)
        bboxes: list[BoundingBox] = []
        page_number: Optional[int] = None
        for span in seg.line_spans:
            if not (span.start < o_end and o_start < span.end):
                continue
            page_number = page_number or span.line.page_number
            rel_s, rel_e = o_start - span.start, o_end - span.start
            words = [w.bbox for w in span.line.words if w.char_start < rel_e and rel_s < w.char_end]
            box = BoundingBox.union(words) or span.line.bbox
            bboxes.append(box)
        norm = raw.normalized
        attributes = dict(raw.attributes)
        if norm is not None and norm.attributes:
            attributes.update({k: v for k, v in norm.attributes.items() if k not in attributes})
        return Entity(
            entity_id=f"entity_{number:04d}",
            label=raw.label,
            surface_text=clause.original_text[o_start:o_end],
            cleaned_surface_text=clause.cleaned_text[raw.start:raw.end],
            normalized_value=norm.normalized if norm else None,
            numeric_value=float(norm.numeric) if norm is not None and norm.numeric is not None else None,
            unit=norm.unit if norm else None,
            currency=norm.currency if norm else None,
            start_offset=o_start,
            end_offset=o_end,
            cleaned_start_offset=raw.start,
            cleaned_end_offset=raw.end,
            clause_id=clause.clause_id,
            page_number=page_number or (clause.page_numbers[0] if clause.page_numbers else 0),
            bboxes=bboxes,
            extraction_method=raw.extraction_method,
            evidence=raw.evidence,
            attributes=attributes,
        )

    def _failed(self, doc_id: str, filename: str, code: str, message: str, started: float) -> ExtractionResult:
        return ExtractionResult(
            document_id=doc_id,
            filename=filename,
            total_pages=0,
            extraction_status=ExtractionStatus.FAILED,
            errors=[ExtractionError(code=code, message=message)],
            metadata=ExtractionMetadata(
                extraction_version=EXTRACTION_VERSION,
                ocr_enabled=self.config.use_ocr,
                offset_reference=OFFSET_REFERENCE,
                processing_time_seconds=round(time.perf_counter() - started, 3),
            ),
        )


# ---------------------------------------------------------------------- #
# Module-level convenience API
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def get_pipeline(config: ExtractionConfig) -> ExtractionPipeline:
    """Return a cached pipeline for ``config`` (configs are frozen/hashable)."""
    return ExtractionPipeline(config)


def analyze_pdf(
    pdf_path: PathLike,
    use_ocr: bool = True,
    *,
    config: Optional[ExtractionConfig] = None,
    document_id: Optional[str] = None,
    password: Optional[str] = None,
    raise_on_error: bool = False,
) -> ExtractionResult:
    """Extract clauses and financial entities from a loan-agreement PDF.

    Args:
        pdf_path: Path to the PDF file.
        use_ocr: Run OCR on scanned/low-text pages (overrides ``config.use_ocr``).
        config: Optional :class:`ExtractionConfig` for other settings.
        document_id: Optional ID; defaults to ``doc_`` + first 12 hex chars of the file SHA-256.
        password: Password for encrypted PDFs.
        raise_on_error: Raise :class:`PDFExtractionError` instead of returning a failed result.

    Returns:
        A validated :class:`ExtractionResult` (``.model_dump()`` gives JSON-ready dicts).
    """
    cfg = replace(config or ExtractionConfig(), use_ocr=use_ocr)
    return get_pipeline(cfg).analyze_pdf(pdf_path, document_id=document_id, password=password,
                                         raise_on_error=raise_on_error)


def analyze_pdf_bytes(
    data: bytes,
    filename: str = "document.pdf",
    use_ocr: bool = True,
    *,
    config: Optional[ExtractionConfig] = None,
    document_id: Optional[str] = None,
    password: Optional[str] = None,
    raise_on_error: bool = False,
) -> ExtractionResult:
    """Same as :func:`analyze_pdf` but for in-memory bytes (e.g. an uploaded file)."""
    cfg = replace(config or ExtractionConfig(), use_ocr=use_ocr)
    return get_pipeline(cfg).analyze_pdf(data, filename=filename, document_id=document_id, password=password,
                                         raise_on_error=raise_on_error)


def analyze_text(
    text: str, *, config: Optional[ExtractionConfig] = None, document_id: str = "doc_text",
    filename: str = "text_input.txt",
) -> ExtractionResult:
    """Run segmentation and entity extraction on plain text."""
    return get_pipeline(config or ExtractionConfig()).analyze_text(text, document_id=document_id, filename=filename)
