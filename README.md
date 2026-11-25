# FinClause

**An NLP-based system for detecting hidden charges, financial obligations and contradictory conditions in
personal loan agreements.**

FinClause reads a loan-agreement PDF, extracts its clauses and financial terms (interest rates, fees, penalties,
foreclosure/prepayment charges, EMI, tenure, repayment and floating-rate conditions), and helps users understand
what they are agreeing to. It organises information for human review. It does not decide that any charge is
fraudulent or illegal.

## Modules

| Module | Owner | Status | Location |
|---|---|---|---|
| PDF/OCR extraction, clause segmentation, financial entity extraction, plain-language key points | Person 1 | ✅ implemented, 223 tests | [`backend/person1_extraction/`](backend/person1_extraction/README.md) |
| Clause classification, semantic retrieval, contradiction detection | Person 2 | planned | – |
| Plain-English explanations, FastAPI integration, React frontend | Person 3 | planned | – |

## Quick start (Windows)

```powershell
git clone https://github.com/TanishaBhide/FinClause.git
cd FinClause
powershell -ExecutionPolicy Bypass -File backend\person1_extraction\setup_windows.ps1 -InstallTesseract
```

The script creates `venv\`, installs all Python dependencies (including the spaCy English model), installs Tesseract
OCR with winget, and runs the tests. To do the same by hand:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
winget install --id UB-Mannheim.TesseractOCR -e     # only needed for scanned PDFs
```

See it working (one-click demo documents: loan agreement, Key Fact Statement, scanned sanction letter, health
insurance terms, No Cost EMI terms, each as a digital PDF or a scanned photocopy):

```powershell
cd backend\person1_extraction
python scripts\demo_server.py          # opens http://127.0.0.1:8765
```

Command line:

```powershell
python scripts\run_extraction.py --input path\to\loan.pdf --output data\processed\result.json
```

Use it from Python:

```python
from finclause_extraction import analyze_pdf
result = analyze_pdf("loan.pdf", use_ocr=True)
for e in result.entities:
    print(e.label, e.normalized_value, "page", e.page_number)
```

## Tech stack

Python 3.10+, FastAPI, PyMuPDF, Tesseract OCR (pytesseract), spaCy, Pydantic, pytest; React frontend.

## Data policy

Do not commit real loan agreements, personal information or large model files. PDFs and module `data/` folders are
git-ignored. Tests use synthetic documents only.

## License

MIT, see [LICENSE](LICENSE).
