# FinClause – Person 1: PDF Processing, OCR, Text Extraction & Financial Entity Extraction

`finclause_extraction` turns a loan agreement, Key Fact Statement, sanction letter, insurance policy or EMI-terms PDF
into structured, validated JSON. It extracts pages, text blocks, clauses (with hierarchy) and financial entities
(with normalised values), adds plain-language key points, and returns warnings that point to exact locations on the page.

**Core guarantee: every extracted entity, key point and warning can be traced to its exact clause, its character
offsets, its page, and its bounding box on the page.**

> This module **extracts and organises** information for further analysis and human review. It never decides whether
> a charge is hidden, unfair, illegal or fraudulent, and its plain-language output is a reading aid, not legal or financial advice.

## Contents
1. [Scope](#1-scope) · 2. [Folder structure](#2-folder-structure) · 3. [Setup on Windows](#3-setup-on-windows) ·
4. [Quick demo](#4-quick-demo-visual-viewer) · 5. [How the pipeline works](#5-how-the-pipeline-works) ·
6. [Entity labels](#6-entity-labels) · 7. [Plain-language key points](#7-plain-language-key-points) ·
8. [Warnings](#8-warnings) · 9. [Output schema](#9-output-schema) · 10. [Python API & CLI](#10-python-api-and-cli) ·
11. [Integration for Person 2 & 3](#11-integration-for-person-2-and-person-3) · 12. [Demo documents](#12-demo-documents) ·
13. [Tests](#13-tests) · 14. [Limitations](#14-limitations-and-known-edge-cases) · 15. [Remaining work](#15-remaining-work)

---

## 1. Scope

| In scope (Person 1) | Out of scope |
|---|---|
| PDF loading/validation (missing, corrupted, encrypted, non-PDF, zero pages) | Clause classification, semantic retrieval, contradiction detection (**Person 2**) |
| Native text extraction with block/line/word boxes (PyMuPDF) | Rich explanations, reports, the React product UI (**Person 3**) |
| Scanned-page detection, OCR image preprocessing (deskew, denoise, binarise), Tesseract OCR | The FastAPI application (**Person 3**) |
| Cleaning with an offset map back to the original text | Model training |
| Clause, paragraph and table-row segmentation with parent/child links | |
| Financial entity extraction (regex + spaCy) and normalisation | |
| Rule-based plain-language key points and simplified clause text (traceable, deterministic) | |
| Located warnings, Pydantic schema, CLI, local demo viewer, tests | |

**Note on scope:** the plain-language layer was added at the team's request. It is template-based and fully traceable
(each point lists its source clause and entities), so Person 3 can show it as-is or build richer explanations on top.

## 2. Folder structure

```
backend/person1_extraction/
├── pyproject.toml / requirements.txt / requirements-lock.txt / setup_windows.ps1 / .env.example
├── src/finclause_extraction/
│   ├── __init__.py             # public API: analyze_pdf, analyze_pdf_bytes, analyze_text, schemas
│   ├── config.py               # ExtractionConfig (all thresholds); offset documentation
│   ├── schemas.py              # Pydantic models (Page, TextBlock, Clause, Entity, KeyPoint, ExtractionResult, ...)
│   ├── pipeline.py             # orchestrates all stages; located warnings; traceability mapping
│   ├── pdf/
│   │   ├── extractor.py        # open/validate PDF, blocks/lines/words, page assessment (scanned? empty?)
│   │   └── ocr.py              # Tesseract discovery, image preprocessing (deskew etc.), OCR -> blocks in PDF coordinates
│   ├── preprocessing/
│   │   ├── cleaner.py          # cleaning with an offset map (never loses the original)
│   │   └── segmenter.py        # header/footer removal, table rows, clause tree, sentence splitting
│   ├── entities/
│   │   ├── patterns.py         # value regexes, trigger lexicon, condition cues
│   │   ├── ner.py              # regex + spaCy span_ruler + context rules + spaCy NER fallback
│   │   └── normalizer.py       # money / % / bps / date / duration normalisation (Decimal-exact)
│   ├── summary/
│   │   ├── simplifier.py       # legal -> plain English rewriting rules + glossary
│   │   ├── key_points.py       # key points from entities (templates)
│   │   └── doc_type.py         # keyword-based document type detection
│   ├── demo_documents.py       # 5 curated fictional demo documents (loan x3, insurance, EMI)
│   ├── samples.py              # PDF builders: text PDFs, tables, photocopy-style scans
│   └── text_input.py           # plain text -> Page objects (for analyze_text / tests)
├── scripts/
│   ├── demo_server.py          # local visual viewer (stdlib HTTP server)
│   ├── run_extraction.py       # CLI
│   └── make_sample_pdf.py      # writes synthetic PDFs into data/raw/
├── demo/index.html             # viewer page (plain HTML/JS, no external dependencies)
├── tests/                      # 223 pytest tests (synthetic fixtures only)
├── data/{raw,processed,annotations}/   # git-ignored except .gitkeep
└── models/                     # git-ignored; place a custom spaCy model here later
```

## 3. Setup on Windows

One command from the repository root (creates `venv`, installs everything, installs Tesseract, runs the tests):

```powershell
powershell -ExecutionPolicy Bypass -File backend\person1_extraction\setup_windows.ps1 -InstallTesseract
```

Manual steps:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1                 # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r backend\person1_extraction\requirements.txt   # includes the spaCy en_core_web_sm model
pip install -e backend\person1_extraction
winget install --id UB-Mannheim.TesseractOCR -e               # only needed for scanned PDFs
```

* `requirements-lock.txt` pins the exact versions the test suite passed with.
* No PyTorch, Transformers or other large models are needed.
* Without `en_core_web_sm`, the pipeline falls back to a blank tokenizer: rule-based extraction still works, and a
  `SPACY_MODEL_UNAVAILABLE` warning is returned.
* **Tesseract:** the pipeline looks for it in this order: `ExtractionConfig(tesseract_cmd=...)` / `--tesseract-cmd`,
  then `FINCLAUSE_TESSERACT_CMD`, then PATH, then the standard install folders (`%ProgramFiles%\Tesseract-OCR`,
  `%ProgramFiles(x86)%\...`, `%LOCALAPPDATA%\Programs\...`). Without Tesseract, digital PDFs still work; scanned pages are
  reported with `OCR_UNAVAILABLE` / `PAGE_NOT_EXTRACTED` warnings that include installation steps.

## 4. Quick demo (visual viewer)

```powershell
cd backend\person1_extraction
python scripts\demo_server.py          # opens http://127.0.0.1:8765
```

1. Click a **demo document**: **▶ Digital PDF** reads the text layer; **▶ Scanned (OCR)** runs the same document as a
   skewed, noisy photocopy that must be read by OCR. Or upload / drag in your own PDF.
2. **Summary** tab: what the document says in plain English, grouped (The basics, Fees and charges, Interest,
   Repayment, If you pay late, If you default, Security and guarantees, What is covered / not covered, Cancellation,
   Terms that can change, Key dates, Worth checking before you sign). ⚠ marks items worth checking. Click a point to
   see the original wording and highlight its source on the page.
3. **Pages**: each page has a **DIGITAL** (green) or **SCANNED** (amber) badge that explains how it was read, with the
   OCR confidence meter and deskew angle for scanned pages. Toggles control the overlays: entities, condition sentences,
   warnings, and **words read** (each OCR word coloured by confidence: green ≥ 90 %, amber 70–90 %, red < 70 %).
4. **Entities** (filter by label, click for evidence), **Clauses** (original and plain-English version side by side),
   **Warnings** (what, why, what to do, and clickable locations), **Pipeline** (what every stage did),
   **JSON** (download).

The server uses only the Python standard library plus this package, listens on localhost only, and keeps uploaded PDFs
in memory (20 most recent). URL shortcuts for demos:
`/?demo=northwind_bank_personal_loan_agreement`, `/?demo=kaveri_cooperative_bank_sanction_letter&scanned=1&tab=warnings&words=1`.
This viewer is a developer/demo tool for Person 1's output, **not** Person 3's product frontend.

## 5. How the pipeline works

```
PDF validation → native extraction (PyMuPDF) → scanned-page detection → [OCR image preprocessing → Tesseract]
→ header/footer detection → table rows + clause segmentation → cleaning (offset map)
→ entity extraction (regex + spaCy) → normalisation → key points & simplified text → located warnings → Pydantic result
```

**1. PDF processing.** Every page is read with PyMuPDF (`rawdict`, reading order top-to-bottom), giving blocks → lines →
words with boxes and font weight. A page is **scanned** if it has fewer than `min_native_chars` (25) characters but
contains images/graphics, or if > 30 % of its characters are unmappable glyphs. It is **empty** if it has no text,
images or drawings. Pages with good text are never OCR'd (unless `force_ocr`).

**2. OCR.** The page is rendered at 300 dpi and cleaned before Tesseract (`ocr_preprocess=True`):
grayscale → contrast stretch → 3×3 median filter (removes toner speckles) → **deskew** (projection-profile search
over ±3°, applied only if ≥ 0.2°) → Otsu binarisation. OCR boxes are rotated back and scaled, so they line up with
the original page. Lone punctuation read with < 50 % confidence (speckles) is dropped and counted. Each page records
`ocr_preprocessing`, `ocr_deskew_angle`, `ocr_mean_confidence` and `ocr_word_count`. On the demo photocopy,
preprocessing raised mean confidence from 90.3 % to 93.6 % and cut low-confidence words from 9 to 1.

**3. Cleaning** (`cleaner.py`) keeps a character-level map back to the original. It removes zero-width and control
characters, expands ligatures, normalises quotes/dashes/spaces, rejoins hyphenated words (`pay-\nment` → `payment`,
but `pre-\npayment` → `pre-payment`), turns line breaks into spaces, keeps paragraph breaks, and repairs OCR spaces
inside currency amounts (`Rs. 3, 000` → `Rs. 3,000`, only after ₹/Rs./INR so lists like `12, 24 and 36` are untouched).
Digits, currency symbols, `%`, decimals and clause numbers are never altered (enforced by tests).

**4. Segmentation** (`segmenter.py`) works on lines:
* running headers/footers (repeated in the top/bottom 8 % band on ≥ half the pages) and page numbers are excluded from
  clauses and reported;
* **table rows**: a line to the right of the previous one on the same row is a cell of that row; each row becomes one
  clause (`"Loan amount sanctioned Rs. 3,00,000"`), cells joined by `\t` in `original_text`;
* clause markers `Section 7 –`, `Article II`, `Clause 5`, `1.`, `1.1`, `1.1.1`, `2(a)`, `(i)`, `a)`, `a.`, bullets;
  decimals such as `2.5 lakh` at a line start are rejected; parent/child links follow numbering, list style and indentation;
* headings (bold/upper-case/title-like), paragraphs, and clauses that continue across blocks and pages.

**5. Entity extraction** (`ner.py`): regex finds values; a spaCy `span_ruler` finds trigger phrases; documented
context rules decide what each value means (R1 trigger after the value with a connector, "Rs. 500 **as** late fee";
R2 nearest trigger before it in the sentence; R3 a specific trigger shortly after; R4 the parent clause's lead-in,
"Foreclosure charges shall be levied as under:" → "(a) 4% ..."; R5 the heading; R6 a generic label). Text such as
"of the loan amount" is stored as the value's `basis` and is never used as a trigger; a tax word only labels the value
it directly precedes; negated mentions ("without any foreclosure charges") are ignored. The pretrained spaCy NER only
adds DATE/MONEY/PERCENT where no regex entity overlaps. No confidence scores are invented.

**6. Normalisation** (`normalizer.py`, Decimal-exact):

| Input | `normalized_value` | Notes |
|---|---|---|
| `₹5,000`, `Rs. 10,000`, `INR 1,25,000` | `5000 INR`, `10000 INR`, `125000 INR` | |
| `Rs. 2.5 lakh`, `₹5 crore` | `250000 INR`, `50000000 INR` | lakh = 10⁵, crore = 10⁷ |
| `2.5 lakh`, `10,000/-` | `250000`, `10000` | **currency never assumed**: `currency: null` + `currency_hint` |
| `12.5% p.a.` / `1.5 percent per month` / `1% of the Sum Insured per day` | `12.5% per annum` / `1.5% per month` / `1% per day` | periods never converted |
| `250 basis points` | `250 bps (2.5%)` | |
| `05/03/2025` | `2025-03-05` | ambiguous → DMY + `AMBIGUOUS_DATE` warning |
| `thirty (30) days`, `12 to 24 months`, `3, 6 and 9 months`, `36 EMIs` | `30 days`, `12 to 24 months`, `3, 6 or 9 months`, `36 instalments` | |

## 6. Entity labels

**Value labels** (one span = one value):

| Label | Example |
|---|---|
| `INTEREST_RATE` | interest at **12.5% p.a.**, APR **17.9%**, spread of **250 basis points** |
| `PROCESSING_FEE` | processing fee of **2%**, minimum **Rs. 3,000** |
| `PENALTY` | penal interest **2% per month**, bounce charges **Rs. 590** |
| `FORECLOSURE_CHARGE` / `PREPAYMENT_CHARGE` | foreclosure charges **4%**; "part-prepayment charges as per schedule" (no value) |
| `LOAN_AMOUNT` / `EMI` / `TENURE` | loan of **Rs. 5,00,000**; 36 EMIs of **Rs. 16,607**; **36 months** |
| `DATE` | **15/01/2025** → `2025-01-15` |
| `TAX` | GST at **18%** |
| `GRACE_PERIOD` | grace period of **30 days** |
| `PREMIUM` / `SUM_INSURED` | annual premium **Rs. 18,450**; sum insured **Rs. 5,00,000** |
| `CO_PAYMENT` / `COVERAGE_LIMIT` / `WAITING_PERIOD` | co-payment of **20%**; room rent capped at **1%** per day; waiting period of **36 months** |
| `OTHER_FINANCIAL_AMOUNT` / `OTHER_PERCENTAGE` | value with another stated purpose (`attributes.related_to`, e.g. "documentation charges") or no determinable purpose |
| `DURATION` | time period that is not tenure/grace/waiting (lock-in, validity, cancellation window) |

**Sentence-level condition labels** (span = the whole sentence; `attributes.cues` lists the matched words):

| Label | Detects |
|---|---|
| `REPAYMENT_CONDITION` | due day (`due_day_of_month`), payment modes (NACH/ECS/PDC), appropriation order |
| `VARIABLE_INTEREST_CONDITION` | floating rate, benchmark (`repo rate`, MCLR, EBLR…), reset/revision |
| `DEFAULT_CONDITION` | events of default, recall, legal action, credit-bureau reporting, set-off, recovery agents |
| `SECURITY_CONDITION` | guarantor, collateral, lien, hypothecation, mortgage |
| `EXCLUSION` | "not covered", "excluded", "not payable", "will not be liable" |
| `CANCELLATION_CONDITION` | cancellation, refund, free-look, cooling-off / look-up period, returns |
| `DISCRETIONARY_CHANGE_CONDITION` | "sole discretion", "reserves the right", "without prior notice" |

A charge mentioned **without an amount** is emitted with `normalized_value: null`, `attributes.value_specified: false`
and any `qualifiers` (e.g. "as per the schedule of charges").

## 7. Plain-language key points

`result.key_points` is a list of `KeyPoint {point_id, category, text, importance, clause_ids, entity_ids,
page_numbers, source_text, method}`, and every non-heading clause gets `simplified_text`. Example (Northwind demo):

```
Fees and charges
  Processing fee (one-time): 1.5% of the Loan amount + GST (non-refundable).
  ⚠ Foreclosure charge (closing the loan early): 4% of the principal outstanding, if the Loan is foreclosed between the 13th and 24th EMI.
If you pay late
  ⚠ Penal interest: 2% a month on the overdue amount (about 24% a year if charged for 12 months).
If you default
  ⚠ The lender can report the default to credit information companies (agencies that keep your credit score), which may adversely affect your credit score.
Worth checking before you sign
  ⚠ Prepayment charges apply, but no amount is stated in this clause (as per the schedule of charges; prevailing). Ask for the exact amount in writing.
```

How points are built (`summary/key_points.py`, all deterministic):
1. **Value points:** `<noun>: <value><details>[, <condition>]`. The noun comes from the label or trigger ("Bounce
   charges"). Details are basis, bound (up to / minimum), frequency, GST and refundability. A condition tail from the same sentence is
   kept ("if the loan is foreclosed within 12 months"). Monthly rates show a clearly marked approximate yearly figure.
2. **Ranges** with the same label are merged ("₹100 to ₹1,300", "13% to 16% a year"); GST in the same sentence is attached to the fee.
3. **Condition sentences** are rewritten by `simplifier.py` (e.g. "the Borrower shall" → "you must", "in the event of" →
   "if there is", "shall constitute" → "counts as", "sole discretion" → "own choice"), with one glossary note per jargon term
   (foreclosure, penal interest, co-payment, NACH, repo rate…). A final check falls back to the original if any number changed.
4. **Check points**: a charge with no stated amount, unless an amount for that charge exists elsewhere in the document.
5. **Dates** with a role ("First EMI due: 7 Apr 2025", "Policy period: 1 Apr 2025 to 31 Mar 2026"); **definitions**
   ("What it means: No Cost EMI means …").

Wording is neutral by design (a test checks that "fraud", "illegal", "unfair", "hidden" never appear).

## 8. Warnings

Every `ExtractionWarning` has `code`, `severity` (`info`/`warning`), a `message` that says what happened, why, and what
to do, and `locations` (`page_number`, `bbox`, `text`, `entity_id`, `clause_id`) so the UI can draw it on the page.
Whole-page warnings use the full-page box.

| Code | Severity | Meaning |
|---|---|---|
| `HEADER_FOOTER_EXCLUDED` | info | running header/footer/page-number lines left out of clauses (box per line) |
| `LOW_OCR_CONFIDENCE` | warning | OCR words below `low_ocr_confidence` (60 %); box + text + confidence per word |
| `AMBIGUOUS_DATE` | warning | e.g. 07/04/2025 could be 7 Apr or 4 Jul; read as DMY |
| `UNCLASSIFIED_VALUE` | info | amount/percentage whose purpose the text does not state |
| `OCR_DISABLED` / `PAGE_NOT_EXTRACTED` | warning | scanned page not read (OCR off / Tesseract missing); status `partial` |
| `OCR_UNAVAILABLE` / `OCR_FAILED` / `OCR_NOT_USED` | warning/info | Tesseract problems, with install instructions |
| `EMPTY_PAGE` / `LOW_TEXT_PAGE` | info | blank page / very little text |
| `SPACY_MODEL_UNAVAILABLE`, `NO_TEXT_EXTRACTED`, `SYNTHETIC_COORDINATES` | warning/info | environment / input notes |

## 9. Output schema

`ExtractionResult` (all fields typed in `schemas.py`; `extraction_version` = `1.1.0`):

| Field | Description |
|---|---|
| `document_id`, `filename`, `total_pages`, `extraction_status` | `success` / `partial` / `failed` |
| `document` | size, sha256, encryption flag, PDF metadata |
| `pages[]` | size, `extraction_method` (`native`/`ocr`/`none`), `needs_ocr`, `ocr_applied`, `assessment_reason`, `ocr_mean_confidence`, `ocr_deskew_angle`, `ocr_preprocessing`, `blocks[] → lines[] → words[]` (boxes; OCR words carry `ocr_confidence`) |
| `clauses[]` | ids, type, numbering, hierarchy, heading, `original_text`, `cleaned_text`, `simplified_text`, pages, `source_locations`, `entity_ids`, `financial_sentences` |
| `entities[]` | label, `surface_text`, `normalized_value`, `numeric_value`, `unit`, `currency`, offsets, `clause_id`, `page_number`, `bboxes`, `extraction_method`, `evidence`, `attributes` |
| `key_points[]` | plain-language points (section 7) |
| `warnings[]`, `errors[]` | section 8 |
| `metadata` | version, OCR/native/empty/unextracted pages, Tesseract & spaCy versions, header/footer lines, label counts, `document_type` (+ evidence), timing |

* Entities are stored once in `result.entities`; clauses reference them by `entity_ids`.
* `clause.original_text[e.start_offset:e.end_offset] == e.surface_text` always holds; `cleaned_*` offsets index `cleaned_text`.
* `original_text` joins lines with `\n`, blocks with `\n\n`, and table cells with `\t`. It is never modified.
* Boxes are in PDF points with a top-left origin; OCR boxes are converted to the same space.

## 10. Python API and CLI

```python
from finclause_extraction import analyze_pdf, analyze_pdf_bytes, analyze_text, ExtractionConfig

result = analyze_pdf("loan.pdf", use_ocr=True)
for p in result.key_points:
    print(p.category, "|", p.text, "| clause", p.clause_ids, "page", p.page_numbers)
fees = result.entities_by_label("PROCESSING_FEE")
clause = result.get_clause(fees[0].clause_id)

result = analyze_pdf("scan.pdf", config=ExtractionConfig(ocr_dpi=300, low_ocr_confidence=70, include_words=False))
result = analyze_pdf_bytes(upload_bytes, filename="upload.pdf")
```

Functions never raise for bad input by default: they return `extraction_status="failed"` with `errors` (`FILE_NOT_FOUND`,
`INVALID_PDF`, `UNSUPPORTED_FORMAT`, `ENCRYPTED_PDF`, `EMPTY_DOCUMENT`, `INTERNAL_ERROR`). Pass `raise_on_error=True`
to get exceptions instead. The spaCy pipeline is loaded once per config and cached. `document_id` = `doc_` + the first 12 hex characters of the file's SHA-256.

CLI:

```powershell
python scripts\run_extraction.py --input data\raw\loan.pdf --output data\processed\loan.json
python scripts\make_sample_pdf.py            # synthetic sample in data\raw\
```

Options: `--no-ocr`, `--force-ocr`, `--tesseract-cmd`, `--ocr-lang eng+hin`, `--spacy-model`, `--no-spacy-model`,
`--no-words`, `--date-order DMY|MDY`, `--password`, `--document-id`, `--compact`, `-v`, `-q`. The summary shows pages,
clauses, entities per label, OCR pages and warnings. Exit code `0` success/partial, `1` failed, `2` bad arguments.

## 11. Integration for Person 2 and Person 3

```python
# backend/app/main.py (owned by Person 3 – example only)
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from finclause_extraction import analyze_pdf_bytes, ExtractionConfig, ExtractionResult

app = FastAPI()
CONFIG = ExtractionConfig(include_words=False)   # word boxes are large; entity boxes are always kept

@app.post("/extract", response_model=ExtractionResult)
async def extract(file: UploadFile, use_ocr: bool = True):
    data = await file.read()
    result = await run_in_threadpool(analyze_pdf_bytes, data, file.filename or "upload.pdf", use_ocr, config=CONFIG)
    if result.extraction_status == "failed":
        raise HTTPException(status_code=422, detail=[e.model_dump() for e in result.errors])
    return result
```

* **Person 2:** use `clauses[].cleaned_text` (keyed by `clause_id`) for classification, embeddings and contradiction
  checks; `heading`, `numbering_path`, `parent_clause_id` give structure. Compare entities by `label` +
  `numeric_value` + `unit` (a `percent_per_month` rate is not comparable to a `percent_per_annum` one). `attributes`
  holds `basis`, `bound`, `taxes`, `frequency`, `qualifiers`, `value_specified`, `rate_type`, `benchmark`, `refundable`.
  Sentence-level conditions mark default, security, exclusion, cancellation and discretionary-change terms.
* **Person 3:** show `key_points` grouped by `category` (titles in `summary.CATEGORY_TITLES`), draw `entities[].bboxes`
  and `warnings[].locations` on rendered pages, and show `clauses[].simplified_text` next to the original.
  `demo/index.html` is a working reference for all of this.

## 12. Demo documents

`demo_documents.py` builds five documents on the fly, each available as a digital PDF or a photocopy-style scan.
**All institutions, names and figures are fictional.** They follow the *format* of real public document types, but no
real institution's document is reproduced, because attributing invented or outdated figures to a real bank or insurer
would be misleading.

| ID | Type | Exercises |
|---|---|---|
| `northwind_bank_personal_loan_agreement` | bank loan agreement, 2 pages | header/footer removal, hyphenation, cross-page clause, tiered foreclosure (a)/(b), default & credit-bureau, cooling-off |
| `sahyadri_finance_key_fact_statement` | NBFC KFS (RBI-style tables) | table rows, APR, floating rate + repo rate, bounce/penal charges, look-up period |
| `kaveri_cooperative_bank_sanction_letter` | sanction letter (scanned by default) | OCR on a skewed photocopy, guarantor (security), recall clause, validity |
| `suraksha_health_insurance_policy_terms` | health insurance terms, 2 pages | sum insured, premium, room-rent caps, co-payment, waiting periods, exclusions, free-look & short-period refunds |
| `shopkart_no_cost_emi_terms` | e-commerce EMI terms | No Cost EMI definition, GST on interest, bank processing fee, tenure lists/ranges, returns, withdrawable offer |

Download any of them from the viewer (⬇ PDF), or run `GET /api/demo/<id>.pdf?scanned=1`.

## 13. Tests

```powershell
cd backend\person1_extraction
python -m pytest          # 223 tests
```

Covered:
* PDF validation and extraction;
* OCR: mocked tests, plus real Tesseract tests that are skipped automatically when Tesseract is not installed. Deskew accuracy, box back-mapping and noise filtering are tested separately;
* cleaning, including value preservation and OCR digit-group repair;
* segmentation, including table rows and cross-page clauses;
* entity extraction, normalisation, the simplifier and key points;
* end-to-end checks on all five demo documents, digital and real-OCR scanned. The scan must reproduce the key figures of the digital version;
* the CLI and the demo server.

All fixtures are synthetic.

## 14. Limitations and known edge cases

* **Accuracy has not been measured on real agreements yet.** The rules were developed on synthetic/fictional documents. No precision/recall figures are claimed.
* **Layout:** multi-column pages may interleave. Tables are handled as rows of cells; inside a row whose cells wrap
  onto several lines, text is ordered line by line. Merged cells and nested tables are not reconstructed.
* **Inline enumerations** ("(a) …; (b) …" on one line) are not split. Only markers at a line start are recognised.
* **OCR:** preprocessing handles skew up to ±3°, speckles and uneven contrast. Handwriting, stamps over text, heavy
  blur and non-English scripts will reduce accuracy. Low-confidence words are always flagged with their location.
* **Plain-language output** is template/rule based. It can be grammatically clumsy for unusual sentences, and it only
  rewrites, never adds facts. It is not legal advice.
* **Context rules are heuristics.** Values far from their trigger get generic labels (and an `UNCLASSIFIED_VALUE` warning).
* **Running headers** are only detected on documents with ≥ 2 pages (on one page they cannot be told apart from body text).
* **Amounts in words** ("Rupees Five Lakh only") are not converted. The numeric form next to them is extracted.
* English only (other languages need `--ocr-lang` plus new trigger phrases).

## 15. Remaining work

1. Collect a few **redacted real** agreements, policies and KFS documents (digital and scanned), annotate the expected entities in
   `data/annotations/`, measure precision/recall per label, and extend `TRIGGERS` and the patterns based on the misses.
2. Agree on the JSON contract (section 9) with Person 2 and Person 3; bump `EXTRACTION_VERSION` on schema changes.
3. Person 3: install with `pip install -e backend/person1_extraction` in the backend environment.
