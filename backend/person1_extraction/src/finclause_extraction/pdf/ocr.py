"""OCR for scanned pages using Tesseract (via pytesseract).

OCR is only run on pages flagged by :func:`..extractor.assess_page` (or on every
page when ``force_ocr`` is set). Word boxes are converted from image pixels back
to PDF points so OCR output is in the same coordinate space as native text.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
from dataclasses import dataclass, field
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


@dataclass
class OCRPageResult:
    """OCR output for one page plus what was done to the image first."""

    blocks: list[TextBlock]
    deskew_angle: float = 0.0
    preprocessing: list[str] = field(default_factory=list)


def otsu_threshold(arr) -> int:
    """Otsu's global threshold for a uint8 grayscale array."""
    import numpy as np

    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    total = arr.size
    cum_w = np.cumsum(hist)
    cum_mu = np.cumsum(hist * np.arange(256))
    mu_t = cum_mu[-1]
    w0 = cum_w
    w1 = total - cum_w
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 / total - cum_mu) ** 2 / (w0 * w1 / total)
    between[~np.isfinite(between)] = 0
    return int(np.argmax(between))


def estimate_skew(gray, max_angle: float = 3.0) -> float:
    """Estimate page skew (degrees) with a projection-profile search.

    Text lines produce sharp peaks in the horizontal ink profile when the page is
    straight; the angle that maximises the profile's variance is returned. The
    value is the rotation (PIL ``rotate`` convention, counter-clockwise) that
    straightens the page.
    """
    import numpy as np
    from PIL import Image

    small = gray.copy()
    small.thumbnail((900, 900))
    arr = np.asarray(small)
    ink = Image.fromarray(((arr < otsu_threshold(arr)) * 255).astype("uint8"))

    def score(angle: float) -> float:
        rotated = np.asarray(ink.rotate(angle, resample=Image.NEAREST, fillcolor=0), dtype=float)
        profile = rotated.sum(axis=1)
        return float(np.var(profile))

    coarse = max(np.arange(-max_angle, max_angle + 1e-9, 0.25), key=score)
    fine = max(np.arange(coarse - 0.25, coarse + 0.25 + 1e-9, 0.05), key=score)
    return round(float(fine), 2)


def preprocess_for_ocr(image):
    """Clean a rendered page before OCR.

    Steps: grayscale -> contrast stretch -> 3x3 median filter (removes toner speckles)
    -> deskew (projection profile) -> Otsu binarisation.
    Returns ``(processed_image, deskew_angle, steps)``.
    """
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    steps = ["grayscale", "autocontrast", "median_filter_3x3"]
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=1).filter(ImageFilter.MedianFilter(3))
    angle = estimate_skew(gray)
    if abs(angle) >= 0.2:  # below this is measurement noise; leave straight pages untouched
        gray = gray.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
        steps.append(f"deskew_{angle:+.2f}deg")
    arr = np.asarray(gray)
    binary = Image.fromarray(((arr > otsu_threshold(arr)) * 255).astype("uint8"))
    steps.append("otsu_binarisation")
    return binary, angle, steps


def ocr_page(
    page: pymupdf.Page, page_number: int, document_id: str, config: ExtractionConfig
) -> list[TextBlock]:
    """OCR a single page and return blocks/lines/words with PDF-space boxes.

    Raises:
        OCRUnavailableError: Tesseract not installed / not configured.
        OCRProcessingError: Tesseract failed on this page (e.g. missing language data).
    """
    return ocr_page_with_info(page, page_number, document_id, config).blocks


def ocr_page_with_info(
    page: pymupdf.Page, page_number: int, document_id: str, config: ExtractionConfig
) -> OCRPageResult:
    """Like :func:`ocr_page` but also reports the preprocessing applied (deskew angle, steps)."""
    pytesseract = _import_pytesseract()
    _configure(pytesseract, config)
    image = _render_page(page, config.ocr_dpi)
    angle, steps = 0.0, []
    if config.ocr_preprocess:
        image, angle, steps = preprocess_for_ocr(image)
    try:
        data = pytesseract.image_to_data(
            image, lang=config.ocr_language, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError(TESSERACT_INSTALL_HELP) from exc
    except Exception as exc:
        raise OCRProcessingError(f"Tesseract failed on page {page_number}: {exc}") from exc
    blocks = blocks_from_tesseract_data(
        data, page_number, document_id, config.ocr_dpi, page.rect.x0, page.rect.y0, config.include_words,
        rotation=(angle, image.width / 2, image.height / 2) if angle else None)
    noise = sum(1 for t, c in zip(data.get("text", []), data.get("conf", []))
                if (t or "").strip() and is_noise_token(t.strip(), _as_conf(c)))
    if noise:
        steps.append(f"dropped_{noise}_noise_tokens")
    return OCRPageResult(blocks, angle, steps)


def _as_conf(value) -> Optional[float]:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return conf if conf >= 0 else None


def is_noise_token(text: str, conf: Optional[float]) -> bool:
    """A lone punctuation mark read with low confidence is almost always a speckle."""
    return (len(text) <= 2 and not any(c.isalnum() for c in text) and text not in ("%", "/-")
            and conf is not None and conf < 50)


def unrotate_box(left: float, top: float, width: float, height: float, angle: float, cx: float, cy: float):
    """Map a box found in an image rotated by ``angle`` (PIL, counter-clockwise, about (cx, cy))
    back to the coordinates of the unrotated image. Returns (x0, y0, x1, y1)."""
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    xs, ys = [], []
    for x, y in ((left, top), (left + width, top), (left, top + height), (left + width, top + height)):
        dx, dy = x - cx, y - cy
        # inverse of PIL's rotation (image y axis points down)
        xs.append(cx + dx * cos_a - dy * sin_a)
        ys.append(cy + dx * sin_a + dy * cos_a)
    return min(xs), min(ys), max(xs), max(ys)


def blocks_from_tesseract_data(
    data: dict,
    page_number: int,
    document_id: str,
    dpi: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    include_words: bool = True,
    rotation: Optional[tuple[float, float, float]] = None,
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
        if is_noise_token(text, conf):
            continue  # counted and reported by ocr_page_with_info
        px0, py0 = float(data["left"][i]), float(data["top"][i])
        px1, py1 = px0 + float(data["width"][i]), py0 + float(data["height"][i])
        if rotation is not None:  # OCR ran on a deskewed image: map back to the original page
            px0, py0, px1, py1 = unrotate_box(px0, py0, px1 - px0, py1 - py0, *rotation)
        box = BoundingBox(
            x0=round(px0 * scale + origin_x, 2),
            y0=round(py0 * scale + origin_y, 2),
            x1=round(px1 * scale + origin_x, 2),
            y1=round(py1 * scale + origin_y, 2),
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
