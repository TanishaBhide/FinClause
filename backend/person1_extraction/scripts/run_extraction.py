"""Command-line runner for the FinClause extraction pipeline.

Examples (run from backend/person1_extraction):

    python scripts/run_extraction.py --input data/raw/sample_loan.pdf --output data/processed/result.json
    python scripts/run_extraction.py --input data/raw/scan.pdf --no-ocr
    python scripts/run_extraction.py --input scan.pdf --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

Exit codes: 0 = success or partial, 1 = extraction failed, 2 = invalid arguments.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent

try:
    import finclause_extraction  # noqa: F401  (installed with `pip install -e .`)
except ImportError:  # running from a source checkout without installation
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from finclause_extraction import ExtractionConfig, analyze_pdf  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract clauses and financial entities from a loan agreement PDF into JSON.")
    p.add_argument("--input", "-i", required=True, type=Path, help="Path to the input PDF")
    p.add_argument("--output", "-o", type=Path,
                   help="Output JSON path (default: data/processed/<input-name>.json)")
    ocr = p.add_mutually_exclusive_group()
    ocr.add_argument("--no-ocr", action="store_true", help="Disable OCR (scanned pages will be reported, not read)")
    ocr.add_argument("--force-ocr", action="store_true", help="OCR every page even if it has a text layer")
    p.add_argument("--tesseract-cmd", help="Full path to tesseract(.exe) if it is not on PATH")
    p.add_argument("--ocr-lang", default="eng", help="Tesseract language(s), e.g. eng or eng+hin (default: eng)")
    p.add_argument("--spacy-model", default="en_core_web_sm",
                   help="spaCy model name or path (default: en_core_web_sm)")
    p.add_argument("--no-spacy-model", action="store_true",
                   help="Do not load a pretrained spaCy model (rule-based extraction only)")
    p.add_argument("--no-words", action="store_true", help="Omit word-level boxes from 'pages' to shrink the JSON")
    p.add_argument("--date-order", choices=["DMY", "MDY"], default="DMY",
                   help="How to read ambiguous dates like 05/03/2025 (default: DMY)")
    p.add_argument("--document-id", help="Override the generated document ID")
    p.add_argument("--password", help="Password for an encrypted PDF")
    p.add_argument("--compact", action="store_true", help="Write compact JSON (no indentation)")
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    verbosity.add_argument("--quiet", "-q", action="store_true", help="Only warnings and errors")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may not be able to print every character (e.g. the rupee sign).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    args = _parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("run_extraction")

    input_path: Path = args.input
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1
    output_path: Path = args.output or (MODULE_ROOT / "data" / "processed" / f"{input_path.stem}.json")

    config = ExtractionConfig(
        use_ocr=not args.no_ocr,
        force_ocr=args.force_ocr,
        tesseract_cmd=args.tesseract_cmd,
        ocr_language=args.ocr_lang,
        use_spacy=not args.no_spacy_model,
        spacy_model=args.spacy_model,
        include_words=not args.no_words,
        date_order=args.date_order,
    )

    log.info("Extracting %s (OCR %s)", input_path, "disabled" if args.no_ocr else "enabled")
    result = analyze_pdf(input_path, use_ocr=config.use_ocr, config=config,
                         document_id=args.document_id, password=args.password)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=None if args.compact else 2), encoding="utf-8")

    md = result.metadata
    print()
    print("=" * 64)
    print(f" FinClause extraction summary  (v{md.extraction_version})")
    print("=" * 64)
    print(f" File              : {result.filename}")
    print(f" Document ID       : {result.document_id}")
    print(f" Status            : {result.extraction_status}")
    print(f" Total pages       : {result.total_pages}")
    print(f" Total clauses     : {len(result.clauses)}")
    print(f" Financial entities: {len(result.entities)}")
    for label, count in md.label_counts.items():
        print(f"     {label:<29}{count}")
    print(f" OCR pages         : {md.ocr_pages or 'none'}")
    if md.unextracted_pages:
        print(f" Unextracted pages : {md.unextracted_pages}")
    print(f" NLP pipeline      : {md.nlp_pipeline}")
    print(f" Warnings          : {len(result.warnings)}")
    for w in result.warnings:
        where = f" (page {w.page_number})" if w.page_number else ""
        first_line = w.message.splitlines()[0] if w.message else ""
        print(f"   - {w.code}{where}: {first_line}")
    for e in result.errors:
        print(f" ERROR {e.code}: {e.message}", file=sys.stderr)
    print(f" Output            : {output_path}")
    print("=" * 64)
    return 1 if result.extraction_status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
