"""OCR for scanned pages using Tesseract (via pytesseract).

OCR is only run on pages flagged by :func:`..extractor.assess_page` (or on every
page when ``force_ocr`` is set). Word boxes are converted from image pixels back
to PDF points so OCR output is in the same coordinate space as native text.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import pymupdf

from ..config import ExtractionConfig
from ..schemas import BoundingBox, ExtractionMethod, TextBlock, TextLine, Word

logger = logging.getLogger(__name__)

TESSERACT_INSTALL_HELP = (
    "Tesseract OCR is required to read scanned pages but was not found.\n"
    "  Windows: install from https://github.com/UB-Mannheim/tesseract/wiki "
    "(default path C:\\Program Files\\Tesseract-OCR\\tesseract.exe), then either\n"
    "    - add the install folder to PATH and restart the terminal, or\n"
    "    - set the environment variable FINCLAUSE_TESSERACT_CMD to the full path of tesseract.exe, or\n"
    "    - pass ExtractionConfig(tesseract_cmd=...) / --tesseract-cmd on the CLI.\n"
    "  Linux: sudo apt install tesseract-ocr    macOS: brew install tesseract\n"
    "Digital (non-scanned) PDFs can still be processed with OCR disabled (use_ocr=False / --no-ocr)."
)


class OCRUnavailableError(RuntimeError):
    """Raised when Tesseract / pytesseract cannot be used."""

    code = "OCR_UNAVAILABLE"


class OCRProcessingError(RuntimeError):
    """Raised when OCR of a specific page fails."""

    code = "OCR_FAILED"


def _import_pytesseract():
    try:
        import pytesseract  # noqa: WPS433 (optional dependency)
    except ImportError as exc:  # pragma: no cover - dependency listed in requirements
        raise OCRUnavailableError("pytesseract is not installed: pip install pytesseract\n" + TESSERACT_INSTALL_HELP) from exc
    return pytesseract


def default_install_locations() -> list[Path]:
    """Standard Windows install folders (derived from environment variables, not hard-coded drives)."""
    if os.name != "nt":
        return []
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
             str(Path(os.environ["LOCALAPPDATA"]) / "Programs") if os.environ.get("LOCALAPPDATA") else None]
    return [Path(r) / "Tesseract-OCR" / "tesseract.exe" for r in roots if r]


def resolve_tesseract_cmd(config: ExtractionConfig) -> Optional[str]:
    """Find the tesseract executable.

    Order: ``config.tesseract_cmd`` -> ``FINCLAUSE_TESSERACT_CMD`` -> PATH -> standard
    Windows install folders (covers "installed but terminal not restarted").
    An explicitly configured path is returned as-is even if it does not exist, so
    the resulting error names the path the user gave.
    """
    explicit = config.resolved_tesseract_cmd()
    if explicit:
        return explicit
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in default_install_locations():
        if candidate.is_file():
            logger.info("Tesseract not on PATH; using %s", candidate)
            return str(candidate)
    return None


def _configure(pytesseract, config: ExtractionConfig) -> None:
    """Point pytesseract at the right executable.

    ``pytesseract.pytesseract.tesseract_cmd`` is process-global state, so it is
    set explicitly on every call instead of only when a custom path is given;
    otherwise a path from an earlier call/request would leak into this one.
    """
    cmd = resolve_tesseract_cmd(config)
    if cmd is None:
        raise OCRUnavailableError(TESSERACT_INSTALL_HELP)
    pytesseract.pytesseract.tesseract_cmd = cmd


def check_tesseract(config: ExtractionConfig) -> str:
    """Verify Tesseract is usable and return its version string.

    Raises:
        OCRUnavailableError: with actionable installation instructions.
    """
    pytesseract = _import_pytesseract()
    _configure(pytesseract, config)
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:  # TesseractNotFoundError, OSError, ...
        raise OCRUnavailableError(f"{exc}\n{TESSERACT_INSTALL_HELP}") from exc
    return version


def _render_page(page: pymupdf.Page, dpi: int):
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def ocr_page(
    page: pymupdf.Page, page_number: int, document_id: str, config: ExtractionConfig
) -> list[TextBlock]:
    """OCR a single page and return blocks/lines/words with PDF-space boxes.

    Raises:
        OCRUnavailableError: Tesseract not installed / not configured.
        OCRProcessingError: Tesseract failed on this page (e.g. missing language data).
    """
    pytesseract = _import_pytesseract()
    _configure(pytesseract, config)
    image = _render_page(page, config.ocr_dpi)
    try:
        data = pytesseract.image_to_data(
            image, lang=config.ocr_language, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError(TESSERACT_INSTALL_HELP) from exc
    except Exception as exc:
        raise OCRProcessingError(f"Tesseract failed on page {page_number}: {exc}") from exc
    return blocks_from_tesseract_data(data, page_number, document_id, config.ocr_dpi, page.rect.x0, page.rect.y0, config.include_words)


def blocks_from_tesseract_data(
    data: dict,
    page_number: int,
    document_id: str,
    dpi: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    include_words: bool = True,
) -> list[TextBlock]:
    """Convert ``pytesseract.image_to_data`` output into :class:`TextBlock` objects.

    Tesseract's (block_num, par_num) pairs become blocks and line_num becomes
    lines. Words keep Tesseract's own confidence value.
    """
    scale = 72.0 / dpi
    grouped: dict[tuple[int, int], dict[int, list[tuple[str, BoundingBox, Optional[float]]]]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf: Optional[float] = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = None
        if conf is not None and conf < 0:
            conf = None
        left, top = data["left"][i] * scale + origin_x, data["top"][i] * scale + origin_y
        box = BoundingBox(
            x0=round(left, 2),
            y0=round(top, 2),
            x1=round(left + data["width"][i] * scale, 2),
            y1=round(top + data["height"][i] * scale, 2),
        )
        key = (int(data["block_num"][i]), int(data["par_num"][i]))
        grouped.setdefault(key, {}).setdefault(int(data["line_num"][i]), []).append((text, box, conf))

    blocks: list[TextBlock] = []
    for key in sorted(grouped):
        block_id = f"p{page_number:03d}_ocr_b{len(blocks):03d}"
        lines: list[TextLine] = []
        confs: list[float] = []
        for line_num in sorted(grouped[key]):
            entries = grouped[key][line_num]
            words: list[Word] = []
            parts: list[str] = []
            pos = 0
            for text, box, conf in entries:
                if parts:
                    pos += 1
                words.append(Word(text=text, bbox=box, char_start=pos, char_end=pos + len(text), ocr_confidence=conf))
                parts.append(text)
                pos += len(text)
                if conf is not None:
                    confs.append(conf)
            line_box = BoundingBox.union(w.bbox for w in words)
            lines.append(
                TextLine(
                    line_id=f"{block_id}_l{len(lines):03d}",
                    text=" ".join(parts),
                    bbox=line_box,
                    words=words if include_words else [],
                )
            )
        blocks.append(
            TextBlock(
                block_id=block_id,
                document_id=document_id,
                page_number=page_number,
                text="\n".join(l.text for l in lines),
                bbox=BoundingBox.union(l.bbox for l in lines),
                extraction_method=ExtractionMethod.OCR,
                lines=lines,
                ocr_mean_confidence=round(sum(confs) / len(confs), 2) if confs else None,
            )
        )
    return blocks
