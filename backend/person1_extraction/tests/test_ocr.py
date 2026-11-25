"""OCR tests.

Tesseract is an external binary, so most tests mock pytesseract. One test runs
real OCR and is skipped automatically when Tesseract is not installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pymupdf
import pytest

from finclause_extraction.config import ExtractionConfig
from finclause_extraction.pdf import ocr as ocr_mod
from finclause_extraction.pdf.ocr import (
    OCRUnavailableError,
    TESSERACT_INSTALL_HELP,
    blocks_from_tesseract_data,
    check_tesseract,
    ocr_page,
)

FAKE_DATA = {
    "block_num": [1, 1, 1, 1, 1, 2],
    "par_num": [1, 1, 1, 1, 1, 1],
    "line_num": [1, 1, 1, 2, 2, 1],
    "word_num": [1, 2, 3, 1, 2, 1],
    "left": [300, 450, 600, 300, 420, 300],
    "top": [300, 300, 300, 360, 360, 600],
    "width": [120, 120, 90, 100, 200, 150],
    "height": [40, 40, 40, 40, 40, 40],
    "conf": ["95.5", "91", "88", "-1", "77", "60"],
    "text": ["Interest", "at", "12%", "", "p.a.", "Clause"],
}


def test_tesseract_data_converted_to_blocks_in_pdf_points() -> None:
    blocks = blocks_from_tesseract_data(FAKE_DATA, 3, "doc_x", dpi=300)
    assert len(blocks) == 2
    b0 = blocks[0]
    assert b0.extraction_method == "ocr" and b0.page_number == 3
    assert [l.text for l in b0.lines] == ["Interest at 12%", "p.a."]
    first = b0.lines[0].words[0]
    assert first.text == "Interest" and first.ocr_confidence == 95.5
    assert first.bbox.x0 == pytest.approx(300 * 72 / 300)
    assert first.bbox.y1 == pytest.approx(340 * 72 / 300)
    # word offsets index into the line text
    line = b0.lines[0]
    for w in line.words:
        assert line.text[w.char_start:w.char_end] == w.text
    # empty token with conf -1 is skipped; mean conf only over real confidences
    assert b0.ocr_mean_confidence == pytest.approx((95.5 + 91 + 88 + 77) / 4, rel=1e-3)


def simulate_no_tesseract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_mod, "default_install_locations", lambda: [])
    monkeypatch.delenv("FINCLAUSE_TESSERACT_CMD", raising=False)


def test_missing_tesseract_gives_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    simulate_no_tesseract(monkeypatch)
    with pytest.raises(OCRUnavailableError) as info:
        check_tesseract(ExtractionConfig())
    msg = str(info.value)
    assert "FINCLAUSE_TESSERACT_CMD" in msg and "PATH" in msg and "--no-ocr" in msg


def test_bad_tesseract_cmd_reported(tmp_path: Path) -> None:
    cfg = ExtractionConfig(tesseract_cmd=str(tmp_path / "nope" / "tesseract.exe"))
    with pytest.raises(OCRUnavailableError):
        check_tesseract(cfg)


def test_ocr_page_with_mocked_tesseract(monkeypatch: pytest.MonkeyPatch, scanned_pdf: Path) -> None:
    import pytesseract

    calls = {}

    def fake_image_to_data(image, lang, output_type):
        calls["size"] = image.size
        calls["lang"] = lang
        return FAKE_DATA

    monkeypatch.setattr(pytesseract, "image_to_data", fake_image_to_data)
    monkeypatch.setattr(ocr_mod, "resolve_tesseract_cmd", lambda cfg: "tesseract")
    doc = pymupdf.open(scanned_pdf)
    blocks = ocr_page(doc[0], 1, "doc_s", ExtractionConfig(ocr_dpi=100, ocr_preprocess=False))
    assert calls["lang"] == "eng"
    assert calls["size"][0] == pytest.approx(doc[0].rect.width * 100 / 72, abs=2)
    # coordinates are scaled with the configured DPI
    assert blocks[0].lines[0].words[0].bbox.x0 == pytest.approx(300 * 72 / 100)


def test_install_help_mentions_windows() -> None:
    assert "Windows" in TESSERACT_INSTALL_HELP and "tesseract.exe" in TESSERACT_INSTALL_HELP


def test_tesseract_cmd_is_reset_on_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: pytesseract's tesseract_cmd is process-global; a path from an earlier
    call (e.g. another request with a custom path) must not leak into later calls."""
    import pytesseract

    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "C:/stale/from/earlier/tesseract.exe")
    monkeypatch.delenv("FINCLAUSE_TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(pytesseract, "get_tesseract_version", lambda: "5.0-test")
    assert check_tesseract(ExtractionConfig()) == "5.0-test"
    assert pytesseract.pytesseract.tesseract_cmd == "/usr/bin/tesseract"


def test_explicit_cmd_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINCLAUSE_TESSERACT_CMD", "D:/env/tesseract.exe")
    assert ocr_mod.resolve_tesseract_cmd(ExtractionConfig(tesseract_cmd="D:/cfg/tesseract.exe")) == "D:/cfg/tesseract.exe"
    assert ocr_mod.resolve_tesseract_cmd(ExtractionConfig()) == "D:/env/tesseract.exe"


def test_windows_install_folder_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "Tesseract-OCR" / "tesseract.exe"
    fake.parent.mkdir()
    fake.write_bytes(b"")
    monkeypatch.delenv("FINCLAUSE_TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_mod, "default_install_locations", lambda: [tmp_path / "missing" / "tesseract.exe", fake])
    assert ocr_mod.resolve_tesseract_cmd(ExtractionConfig()) == str(fake)


_REAL_TESSERACT = ocr_mod.resolve_tesseract_cmd(ExtractionConfig()) is not None


@pytest.mark.skipif(not _REAL_TESSERACT, reason="Tesseract OCR binary not installed")
def test_real_tesseract_version() -> None:  # pragma: no cover - environment dependent
    assert check_tesseract(ExtractionConfig())[0].isdigit()


@pytest.mark.skipif(not _REAL_TESSERACT, reason="Tesseract OCR binary not installed")
def test_real_ocr_on_scanned_pdf(scanned_pdf: Path) -> None:  # pragma: no cover - environment dependent
    doc = pymupdf.open(scanned_pdf)
    blocks = ocr_page(doc[0], 1, "doc_real", ExtractionConfig())
    text = " ".join(b.text for b in blocks)
    assert "PERSONAL LOAN AGREEMENT" in text.upper()
    assert "5,00,000" in text
    words = [w for b in blocks for l in b.lines for w in l.words]
    assert words and all(w.ocr_confidence is None or 0 <= w.ocr_confidence <= 100 for w in words)
    page_w, page_h = doc[0].rect.width, doc[0].rect.height
    assert all(0 <= w.bbox.x0 < w.bbox.x1 <= page_w and 0 <= w.bbox.y0 < w.bbox.y1 <= page_h for w in words)


# ---- image preprocessing before OCR ----
def _text_image(angle: float = 0.0):
    from PIL import Image

    from finclause_extraction.samples import SYNTHETIC_AGREEMENT, build_pdf

    doc = pymupdf.open(stream=build_pdf(None, [SYNTHETIC_AGREEMENT[0]]), filetype="pdf")
    pix = doc[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    return img.rotate(angle, resample=Image.BICUBIC, fillcolor=255) if angle else img


@pytest.mark.parametrize("skew", [-1.5, -0.8, 1.2, 2.0])
def test_deskew_estimates_rotation(skew: float) -> None:
    from finclause_extraction.pdf.ocr import estimate_skew

    # a page rotated by `skew` needs a rotation of -skew to be straight
    assert estimate_skew(_text_image(skew)) == pytest.approx(-skew, abs=0.15)


def test_straight_page_is_not_rotated() -> None:
    from finclause_extraction.pdf.ocr import preprocess_for_ocr

    _, angle, steps = preprocess_for_ocr(_text_image())
    assert abs(angle) < 0.2 and not any(s.startswith("deskew") for s in steps)
    assert steps[0] == "grayscale" and steps[-1] == "otsu_binarisation"


def test_unrotate_box_inverts_pil_rotation() -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from finclause_extraction.pdf.ocr import unrotate_box

    for angle in (10, -7, 1.3):
        img = Image.new("L", (400, 300), 255)
        ImageDraw.Draw(img).rectangle((300, 200, 304, 204), fill=0)
        arr = np.asarray(img.rotate(angle, resample=Image.NEAREST, fillcolor=255))
        ys, xs = np.where(arr < 128)
        x0, y0, x1, y1 = unrotate_box(xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min(), angle, 200, 150)
        assert (x0, y0) == pytest.approx((300, 200), abs=1.5) and (x1, y1) == pytest.approx((304, 204), abs=1.5)


def test_noise_tokens_dropped_but_symbols_kept() -> None:
    from finclause_extraction.pdf.ocr import is_noise_token

    assert is_noise_token(".", 20) and is_noise_token(")", 10)
    assert not is_noise_token("%", 20) and not is_noise_token(".", 95) and not is_noise_token("1.", 20)
