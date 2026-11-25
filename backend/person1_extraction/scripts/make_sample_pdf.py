"""Generate synthetic (fictional) loan-agreement PDFs for demos.

    python scripts/make_sample_pdf.py                  # -> data/raw/synthetic_loan_agreement.pdf
    python scripts/make_sample_pdf.py --scanned        # -> data/raw/synthetic_loan_agreement_scanned.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent
try:
    import finclause_extraction  # noqa: F401
except ImportError:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

from finclause_extraction.samples import SYNTHETIC_AGREEMENT, build_pdf, build_scanned_pdf  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scanned", action="store_true", help="Create an image-only (scanned-like) PDF")
    p.add_argument("--output", "-o", type=Path)
    args = p.parse_args()
    name = "synthetic_loan_agreement_scanned.pdf" if args.scanned else "synthetic_loan_agreement.pdf"
    out = args.output or MODULE_ROOT / "data" / "raw" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.scanned:
        build_scanned_pdf(out, SYNTHETIC_AGREEMENT)
    else:
        build_pdf(out, SYNTHETIC_AGREEMENT, footer_page_numbers=True)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
