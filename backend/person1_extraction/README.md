# FinClause – Person 1: PDF Processing, Text Extraction & Financial Entity Extraction

`finclause_extraction` turns a personal-loan-agreement PDF into structured, validated JSON:
pages → text blocks → clauses (with hierarchy) → financial entities (with normalised values).

**Core guarantee: every extracted entity can be traced to its exact clause, its character offsets inside
that clause, its page, and its bounding box on the page.**

> This module **extracts and organises** information for further analysis and human review.
> It never decides whether a charge is hidden, unfair, illegal or fraudulent.

---

## 1. Scope

| In scope (Person 1) | Out of scope |
|---|---|
| PDF loading/validation (missing, corrupted, encrypted, non-PDF, zero pages) | Clause classification, semantic retrieval, contradiction detection (**Person 2**) |
| Native text extraction with block/line/word boxes (PyMuPDF) | Plain-English explanations, reports, React UI (**Person 3**) |
| Scanned-page detection and OCR (Tesseract) | The FastAPI application itself (**Person 3**) |
| Cleaning with an offset map back to the original text | Model training |
| Clause & paragraph segmentation with parent/child links | |
| Financial entity extraction (regex + spaCy) and normalisation | |
| Pydantic output schema, CLI, tests | |

The package does **not** depend on FastAPI, Streamlit or any frontend code.

## 2. Folder structure

```
backend/person1_extraction/
├── pyproject.toml              # installable package (pip install -e .)
├── requirements.txt
├── README.md
├── .gitignore                  # keeps data/ and models/ contents out of git
├── src/finclause_extraction/
│   ├── __init__.py             # public API: analyze_pdf, analyze_pdf_bytes, analyze_text, schemas
│   ├── config.py               # ExtractionConfig (all thresholds), offset documentation
│   ├── schemas.py              # Pydantic models (Page, TextBlock, Clause, Entity, ExtractionResult, ...)
│   ├── pipeline.py             # orchestrates all stages, error handling, traceability mapping
│   ├── text_input.py           # plain text -> Page objects (for analyze_text / tests)
│   ├── samples.py              # synthetic (fictional) agreement PDFs for tests & demos
│   ├── pdf/
│   │   ├── extractor.py        # open/validate PDF, blocks/lines/words, page assessment
│   │   └── ocr.py              # Tesseract OCR, availability check, actionable install help
│   ├── preprocessing/
│   │   ├── cleaner.py          # cleaning with an offset map (never loses the original)
│   │   └── segmenter.py        # header/footer removal, clause tree, sentence splitting
│   └── entities/
│       ├── patterns.py         # value regexes, trigger lexicon, context regexes
│       ├── ner.py              # regex + spaCy span_ruler + context rules + spaCy NER fallback
│       └── normalizer.py       # money / % / bps / date / duration normalisation (Decimal-exact)
├── scripts/
│   ├── run_extraction.py       # CLI
│   ├── demo_server.py          # local visual viewer (stdlib HTTP server)
│   └── make_sample_pdf.py      # writes synthetic demo PDFs into data/raw/
├── demo/index.html             # viewer page (plain HTML/JS, no external dependencies)
├── tests/                      # 176 pytest tests (synthetic fixtures only)
├── data/{raw,processed,annotations}/   # git-ignored except .gitkeep
└── models/                     # git-ignored; place a custom spaCy model here later
```

## 3. Setup on Windows

From the repository root (`FinClause/`), in PowerShell:

```powershell
# 1. Create / activate a virtual environment (Python 3.10+)
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
#   (if activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# 2. Install dependencies and the package (editable)
python -m pip install --upgrade pip
pip install -r backend\person1_extraction\requirements.txt
pip install -e backend\person1_extraction

# 3. spaCy English model (separate download, ~13 MB)
python -m spacy download en_core_web_sm
```

No PyTorch, Transformers or other large models are needed.
If `en_core_web_sm` is missing, the pipeline still runs, falling back to a blank tokenizer. Rule-based extraction
still works, but the pretrained NER fallback is disabled, and the result carries a `SPACY_MODEL_UNAVAILABLE` warning.

### Tesseract OCR (only needed for scanned PDFs)

Tesseract is a separate program, not a pip package.

1. Download the Windows installer from <https://github.com/UB-Mannheim/tesseract/wiki> and install it
   (default: `C:\Program Files\Tesseract-OCR\`).
2. Make it findable using **one** of these options:
   * add `C:\Program Files\Tesseract-OCR` to your **PATH** (System Properties → Environment Variables), then open a new terminal; or
   * set `FINCLAUSE_TESSERACT_CMD` to the full path of `tesseract.exe`:
     `setx FINCLAUSE_TESSERACT_CMD "C:\Program Files\Tesseract-OCR\tesseract.exe"`; or
   * pass `--tesseract-cmd "C:\...\tesseract.exe"` (CLI) / `ExtractionConfig(tesseract_cmd=...)` (Python).
3. Check that it works: `tesseract --version` (in a **new** terminal)

   Quick install alternative: `winget install --id UB-Mannheim.TesseractOCR -e` (the English language data is included).

If Tesseract is not on PATH, the pipeline also checks the standard install folders
(`%ProgramFiles%\Tesseract-OCR`, `%ProgramFiles(x86)%\Tesseract-OCR`, `%LOCALAPPDATA%\Programs\Tesseract-OCR`).
This means OCR also works in terminals that were opened before the install.

Without Tesseract, digital PDFs are processed normally. Scanned pages are reported (status `partial`,
warnings `OCR_UNAVAILABLE` + `PAGE_NOT_EXTRACTED` with these instructions) and the pipeline does not crash.

## 4. Running the tests

```powershell
cd backend\person1_extraction
python -m pytest            # all tests
python -m pytest -v tests\test_entity_extraction.py   # one file
```

Tests generate their own synthetic PDFs, so no confidential documents are needed. Most OCR tests mock Tesseract, so they
run anywhere. Three tests run the real Tesseract binary (version check, OCR of a scanned page, and a full scanned-PDF
pipeline run). They are reported as *skipped* on machines without Tesseract.

## 5. Command-line interface

```powershell
cd backend\person1_extraction
python scripts\make_sample_pdf.py                       # creates data\raw\synthetic_loan_agreement.pdf
python scripts\run_extraction.py --input data\raw\synthetic_loan_agreement.pdf --output data\processed\result.json
```

| Option | Meaning |
|---|---|
| `--input/-i` | PDF to process (required) |
| `--output/-o` | JSON output (default `data/processed/<name>.json`) |
| `--no-ocr` / `--force-ocr` | disable OCR / OCR every page |
| `--tesseract-cmd PATH`, `--ocr-lang eng+hin` | OCR configuration |
| `--spacy-model NAME_OR_PATH`, `--no-spacy-model` | choose spaCy pipeline / rules only |
| `--no-words` | drop word boxes from `pages` (entity boxes are kept) |
| `--date-order DMY\|MDY` | how to read ambiguous dates like 05/03/2025 (default DMY) |
| `--password`, `--document-id`, `--compact`, `-v`, `-q` | misc |

The CLI prints a summary showing pages, clauses, entities per label, OCR pages, unextracted pages and warnings.
Exit code: `0` success/partial, `1` failed, `2` bad arguments.

### Visual viewer (local demo)

```powershell
cd backend\person1_extraction
python scripts\demo_server.py          # opens http://127.0.0.1:8765 in your browser
```

Upload or drag in a PDF, or click **Sample (digital)** / **Sample (scanned)**. The viewer shows:

* each page with a coloured box around every extracted entity (dashed underline for sentence-level conditions);
* an entity table you can filter by label;
* the clause tree with values highlighted (hover a clause to outline it on the page);
* the evidence for any entity you click: original vs. normalised value, offsets, rule, trigger and attributes;
* warnings, excluded header/footer lines, and a JSON download.

It uses only the Python standard library (`http.server`) plus this package, listens on localhost only, and keeps
uploaded PDFs in memory (the 20 most recent documents). It is a developer tool for inspecting Person 1's output, **not**
the product frontend (Person 3's React app). URL shortcuts for demos: `/?sample=digital`, `/?sample=scanned&tab=clauses`,
`/?sample=digital&select=entity_0011`.

## 6. Python API

```python
from finclause_extraction import analyze_pdf, analyze_pdf_bytes, analyze_text, ExtractionConfig

result = analyze_pdf("data/raw/loan.pdf", use_ocr=True)        # -> ExtractionResult (Pydantic)
result.extraction_status            # "success" | "partial" | "failed"
for e in result.entities_by_label("PROCESSING_FEE"):
    clause = result.get_clause(e.clause_id)
    print(e.normalized_value, "| page", e.page_number, "| clause", clause.numbering_path, "|", clause.cleaned_text)

# custom settings
cfg = ExtractionConfig(min_native_chars=40, ocr_dpi=300, date_order="DMY", include_words=False)
result = analyze_pdf("loan.pdf", use_ocr=True, config=cfg, password=None)

# bytes (e.g. an upload) and plain text
result = analyze_pdf_bytes(pdf_bytes, filename="upload.pdf")
result = analyze_text("1.1 Processing fee of 2% of the loan amount.")
```

* The functions **do not raise** for bad input. They return `extraction_status="failed"` with `errors=[{code, message}]`
  (`FILE_NOT_FOUND`, `INVALID_PDF`, `UNSUPPORTED_FORMAT`, `ENCRYPTED_PDF`, `EMPTY_DOCUMENT`, `INTERNAL_ERROR`).
  Pass `raise_on_error=True` to get exceptions (`PDFNotFoundError`, `InvalidPDFError`, ...).
* The spaCy pipeline is loaded once per `ExtractionConfig` and cached (`get_pipeline`), so repeated calls are fast.
* `document_id` defaults to `doc_` + the first 12 hex characters of the file's SHA-256, so the same file always gets
  the same ID. Clause and entity IDs are sequential in reading order, so they are also stable for the same file and version.

## 7. Integration with the main FastAPI backend (for Person 3)

```python
# backend/app/main.py  (owned by Person 3 – example only, not implemented here)
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from finclause_extraction import analyze_pdf_bytes, ExtractionResult

app = FastAPI()

@app.post("/extract", response_model=ExtractionResult)
async def extract(file: UploadFile, use_ocr: bool = True):
    data = await file.read()
    result = await run_in_threadpool(analyze_pdf_bytes, data, file.filename or "upload.pdf", use_ocr)
    if result.extraction_status == "failed":
        raise HTTPException(status_code=422, detail=[e.model_dump() for e in result.errors])
    return result
```

Notes:

* Extraction is CPU-bound and synchronous. Run it in a thread pool (shown above) or a worker queue.
* `ExtractionResult` is a Pydantic v2 model, so it can be used directly as `response_model`.
  Use `result.model_dump(mode="json")` to get a plain dict.
* The output with word boxes is roughly 40–50 KB per page. Use `ExtractionConfig(include_words=False)` for API
  responses if the UI does not need word boxes (entity `bboxes` are always kept).

### How Person 2 should consume the output

* **Clause-level work** (classification, embeddings, FAISS, contradiction checks): iterate over `result.clauses` and use
  `cleaned_text` as the model input. Keep `clause_id` as the key. `heading`, `numbering_path`, `parent_clause_id` and
  `child_clause_ids` provide structure (e.g. combine a list item with its parent's lead-in sentence).
  `financial_sentences` gives sentence-level units that already contain financial terms.
* **Entity-level work**: `result.entities`, joined to clauses on `clause_id`. Compare like with like using `label`,
  `numeric_value` and `unit`/`currency`. For example, two `INTEREST_RATE` entities with different `unit`
  (`percent_per_annum` vs `percent_per_month`) are not directly comparable. Monthly rates are never converted to annual.
* `attributes` holds structured context for detecting overlooked conditions: `basis` ("loan amount", "outstanding principal"),
  `bound` (upper/lower), `taxes` ("additional"/"included"), `frequency`, `qualifiers` (e.g. "as per schedule of charges"),
  `value_specified: false` (a charge mentioned without an amount), `rate_type`, `benchmark`.

### How Person 3 should use it for highlighting and explanations

`entity.page_number` + `entity.bboxes` (PDF points, origin top-left) let you highlight the exact words on the rendered
page. `clause.source_locations` lets you highlight the whole clause. `evidence.trigger_text` shows why a label was assigned.

## 8. Output schema

`ExtractionResult` (all fields typed in `schemas.py`):

| Field | Description |
|---|---|
| `document_id`, `filename`, `total_pages`, `extraction_status` | summary |
| `document` | `DocumentInfo`: size, sha256, encryption flag, PDF metadata |
| `pages[]` | `Page`: size, `extraction_method` (`native`/`ocr`/`none`), `needs_ocr`, `ocr_applied`, `is_empty`, `blocks[]` → `lines[]` → `words[]` (with boxes; OCR words carry Tesseract `ocr_confidence`) |
| `clauses[]` | `Clause`: ids, type, numbering, hierarchy, heading, `original_text`, `cleaned_text`, pages, `source_locations`, `entity_ids`, `financial_sentences` |
| `entities[]` | `Entity`: label, surface/normalised values, offsets, `clause_id`, page, `bboxes`, method, `evidence`, `attributes` |
| `warnings[]`, `errors[]` | `{code, message, page_number?}` |
| `metadata` | version, OCR/native/empty/unextracted pages, Tesseract & spaCy versions, header/footer lines excluded, label counts, timing |

**Entities are stored once**, in `result.entities`. Each clause references its entities by `entity_ids`, so nothing is duplicated.

**Offsets** (also given in `metadata.offset_reference`):

* `Entity.start_offset/end_offset` index into `Clause.original_text`, and
  `clause.original_text[e.start_offset:e.end_offset] == e.surface_text` always holds.
* `Entity.cleaned_start_offset/cleaned_end_offset` index into `Clause.cleaned_text`.
* `SourceLocation.start_offset/end_offset` index into `Clause.original_text`. `Word.char_start/char_end` index into `TextLine.text`.
* `Clause.original_text` is the concatenation of the clause's source lines: `\n` joins lines within a block and `\n\n`
  separates blocks/pages. It is never modified by cleaning.

### Example (real output from `data/raw/synthetic_loan_agreement.pdf`, trimmed)

```json
{
  "document_id": "doc_65b0cd6c9499",
  "filename": "synthetic_loan_agreement.pdf",
  "total_pages": 2,
  "extraction_status": "success",
  "clauses": [{
    "clause_id": "clause_0012",
    "parent_clause_id": "clause_0010",
    "clause_type": "subclause",
    "clause_number": "3.2",
    "numbering_path": "3.2",
    "level": 2,
    "heading": "FEES AND CHARGES",
    "original_text": "3.2 Late payment charges of Rs. 500 per instance shall apply, and penal interest of 2% per\nmonth shall be levied on the overdue amount.",
    "cleaned_text": "3.2 Late payment charges of Rs. 500 per instance shall apply, and penal interest of 2% per month shall be levied on the overdue amount.",
    "page_numbers": [1],
    "source_locations": [{"page_number": 1, "block_id": "p001_b011", "line_ids": ["p001_b011_l000", "p001_b011_l001"],
                          "bbox": {"x0": 72.0, "y0": 367.15, "x1": 473.88, "y1": 394.39}, "start_offset": 0, "end_offset": 135}],
    "entity_ids": ["entity_0010", "entity_0011"],
    "is_financially_relevant": true
  }],
  "entities": [{
    "entity_id": "entity_0011",
    "label": "PENALTY",
    "surface_text": "2% per\nmonth",
    "cleaned_surface_text": "2% per month",
    "normalized_value": "2% per month",
    "numeric_value": 2.0,
    "unit": "percent_per_month",
    "currency": null,
    "start_offset": 84, "end_offset": 96,
    "cleaned_start_offset": 84, "cleaned_end_offset": 96,
    "clause_id": "clause_0012",
    "page_number": 1,
    "bboxes": [{"x0": 442.2, "y0": 367.15, "x1": 473.88, "y1": 380.89},
               {"x0": 72.0, "y0": 380.65, "x1": 99.79, "y1": 394.39}],
    "extraction_method": "regex_context",
    "evidence": {"rule": "value:percent+trigger:sentence_preceding", "trigger_text": "penal interest",
                 "trigger_label": "PENALTY", "trigger_source": "sentence_preceding"},
    "attributes": {"period": "per_month"}
  }],
  "warnings": [{"code": "HEADER_FOOTER_EXCLUDED", "message": "2 header/footer/page-number lines excluded from clauses ..."}],
  "errors": [],
  "metadata": {"extraction_version": "1.0.0", "ocr_used": false, "nlp_pipeline": "en_core_web_sm 3.8.0", "...": "..."}
}
```

The value wraps onto a second line, so it has two boxes.

## 9. Entity labels

| Label | Meaning | Example |
|---|---|---|
| `INTEREST_RATE` | rate of interest, APR, spread | "interest at **12.5% p.a.**" |
| `PROCESSING_FEE` | processing / login / origination fee | "processing fee of **2%**" |
| `PENALTY` | late payment, penal/default interest, bounce/dishonour charges | "penal interest of **2% per month**" |
| `FORECLOSURE_CHARGE` | foreclosure / pre-closure charge | "foreclosure charges of **4%**" |
| `PREPAYMENT_CHARGE` | prepayment / part-payment charge | "**prepayment charges** subject to applicable conditions" |
| `LOAN_AMOUNT` | sanctioned / principal / loan amount | "a loan of **Rs. 5,00,000**" |
| `EMI` | instalment amount | "36 EMIs of **Rs. 16,607**" |
| `TENURE` | loan term or number of instalments | "tenure of **36 months**", "repayable in **36 EMIs**" |
| `DATE` | calendar date (ISO-normalised) | "**15/01/2025**" → `2025-01-15` |
| `OTHER_FINANCIAL_AMOUNT` | money whose role is unknown or is another charge (`attributes.related_to`) | "documentation charges of **Rs. 1,000**" |
| `REPAYMENT_CONDITION` | sentence-level repayment terms (due day, mode, appropriation order) | "EMI payable on the fifth day of every month" |
| `VARIABLE_INTEREST_CONDITION` | sentence-level floating/benchmark/reset terms | "floating rate linked to the repo rate" |
| **`TAX`** *(added)* | GST / tax rate or amount, so tax is not mistaken for a fee or interest | "GST at **18%**" |
| **`OTHER_PERCENTAGE`** *(added)* | percentage whose meaning cannot be determined, so it is never wrongly called interest | "acknowledges **15%**" |
| **`DURATION`** *(added)* | time period that is not the loan tenure (grace, notice, lock-in) | "foreclosed within **12 months**" |

A charge mentioned **without a value** (e.g. "Prepayment charges as per the Schedule of Charges") is emitted with
`normalized_value: null`, `attributes.value_specified: false` and any `qualifiers`. These are exactly the references
that Person 2 may want to flag for human review.

### How labels are decided (documented rules)

1. **Values** are found by regex: `₹/Rs./INR/rupees`, Indian and Western digit grouping, `lakh/crore`, `/-`, `%`,
   `percent`, `per cent`, `bps`, `p.a./p.m./per annum/per month`, numeric and textual dates, and durations
   ("thirty (30) days"). Overlaps are resolved longest-first, then date > money > %/bps > duration.
2. **Triggers** ("processing fee", "penal interest", "tenure", ...) are matched by a spaCy `span_ruler` (see
   `TRIGGERS` in `patterns.py`). Nested matches keep the longest one, so "penal interest" is PENALTY, not interest.
3. **Context rules** pick the trigger that explains a value. R1: a trigger after the value with a connector
   ("Rs. 500 **as** late fee"). R2: the nearest applicable trigger before the value in the same sentence; a generic
   "charges/fee" yields to a specific trigger right after the value. R3: a specific trigger shortly after the value.
   R4: the parent clause's lead-in ("Foreclosure charges shall be levied as under:" → "(a) 4% …"). R5: the clause
   heading. R6: otherwise a generic label.
4. Triggers must suit the value type. "interest" makes a percentage an `INTEREST_RATE`, but a rupee amount near
   "interest" stays `OTHER_FINANCIAL_AMOUNT`. Words in a value's basis ("2% of the **loan amount**") never label
   another value.
5. The pretrained spaCy NER adds DATE/MONEY/PERCENT only where no regex entity overlaps (`extraction_method: spacy_ner`).

`extraction_method` values: `regex_context` (value + trigger), `regex_pattern` (pattern only),
`spacy_span_ruler` (value-less charge mention), `spacy_ner`, `spacy_custom_model`. `evidence` records the rule and
trigger. **No confidence scores are invented.** The only scores in the output are Tesseract's own OCR word confidences.

### Plugging in a custom spaCy model later

**No model has been trained in this project yet.** If one is trained later (e.g. on annotations saved in
`data/annotations/`), save it to `models/` and run with `ExtractionConfig(spacy_model="models/<name>")`. Any entity it
predicts with a label from the table above is added as `spacy_custom_model` wherever no rule-based entity overlaps.
The trigger ruler is added on top of the loaded pipeline automatically.

## 10. Normalisation rules

| Input | `normalized_value` | Notes |
|---|---|---|
| `₹5,000`, `Rs. 10,000`, `INR 1,25,000` | `5000 INR`, `10000 INR`, `125000 INR` | `currency: "INR"` |
| `Rs. 2.5 lakh`, `₹5 crore` | `250000 INR`, `50000000 INR` | lakh = 10⁵, crore = 10⁷ |
| `2.5 lakh`, `10,000/-` (no currency written) | `250000`, `10000` | `currency: null` + `currency_hint`. **The currency is never assumed** |
| `Rs. 5,000.50` | `5000.50 INR` | written decimal precision kept (Decimal arithmetic) |
| `12.5% p.a.` / `1.5 percent per month` | `12.5% per annum` / `1.5% per month` | `unit: percent_per_annum` / `percent_per_month`. **Never converted** between periods |
| `11% (fixed) per annum` | `11% per annum` | period taken from following text (`period_source: following_context`) |
| `250 basis points` | `250 bps (2.5%)` | `attributes.percent_equivalent` |
| `05/03/2025` | `2025-03-05` | ambiguous → configured order (DMY), flagged `date_order_assumed` |
| `3 years` | `3 years` | `attributes.derived_months: "36"` (explicitly derived, separate field) |

Note: the original specification table listed `2.5 lakh → 250000 INR`. Because the specification also says "do not assume
a missing currency", a bare `2.5 lakh` is normalised to `250000` with `currency: null` plus a hint. When a currency is written
(`Rs. 2.5 lakh`), the output is `250000 INR` as specified.

## 11. Limitations and known edge cases

* **Layout**: reading order is top-to-bottom, left-to-right. Multi-column pages and complex tables may interleave
  text. Table cells are extracted as text lines, not as table structure.
* **Inline enumerations** such as "the Borrower shall (a) pay …; (b) maintain …" written on one line are not split into
  sub-clauses. Only markers at the start of a line are recognised.
* **Headings** without bold/uppercase formatting (common in OCR output, which has no font information) are recognised
  only when they are a single-line block or directly followed by a list.
* **Unnumbered headings** close the open numbering context. A bold subheading placed between `4.2` and `4.3` becomes
  the parent of `4.3`.
* **Amounts in words** ("Rupees Five Lakh only") are not converted. The numeric form next to them is extracted.
* **Context rules are heuristics.** A value far from its trigger (more than 150 characters, configurable), or in a different sentence
  without a parent/heading cue, gets a generic label. `evidence` shows how every label was reached so reviewers can
  check it.
* **Sentence-level conditions** (`REPAYMENT_CONDITION`, `VARIABLE_INTEREST_CONDITION`) span the whole sentence.
* **Repeated header/footer detection** needs at least 2 pages. Page-number lines are detected on any page.
* **OCR quality** depends on the scan. OCR text is used as-is (no spell correction), and Tesseract confidences are reported per word.
  On the synthetic scanned agreement (`make_sample_pdf.py --scanned`, 150 dpi), Tesseract 5.4 read 248 words with a
  mean confidence of 95.6 and produced the same 20 entities as the digital PDF. The only OCR error was
  "EMIs" read as "EMls". The EMI patterns tolerate this kind of confusion (`EMl`, `EM1`) without changing the original
  text. Real scans (skewed, stamped, low-resolution) will be harder and have not been tested yet.
* **Languages**: English only (Hindi or other languages would need `--ocr-lang` and additional trigger phrases).
* **Accuracy has not been measured on real agreements yet.** Tests use synthetic documents. No precision/recall
  figures are claimed.

## 12. Remaining work before integration

1. Run the pipeline on a few real (redacted) **scanned** agreements to check OCR quality beyond synthetic scans.
2. Collect a small set of **non-confidential / redacted** real agreements, annotate expected entities in
   `data/annotations/`, and measure precision/recall per label. Then extend `TRIGGERS` and the patterns based on the misses.
3. Agree on the JSON contract with Person 2 and Person 3 (this README §8). Changes to `schemas.py` should bump
   `EXTRACTION_VERSION` in `config.py`.
4. Person 3: add `pip install -e backend/person1_extraction` (or the requirements) to the backend environment.
