<#
.SYNOPSIS
    One-command setup for the FinClause Person 1 extraction module on Windows.

.DESCRIPTION
    Creates (or reuses) the virtual environment at the repository root (FinClause\venv),
    installs the Python dependencies including the spaCy English model, installs this
    package in editable mode, checks for Tesseract OCR (optionally installing it with
    winget), and runs the test suite.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File backend\person1_extraction\setup_windows.ps1
    powershell -ExecutionPolicy Bypass -File backend\person1_extraction\setup_windows.ps1 -InstallTesseract -Locked
#>
param(
    [switch]$InstallTesseract,   # install Tesseract with winget if it is missing
    [switch]$Locked,             # use requirements-lock.txt (exact tested versions)
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ModuleDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ModuleDir "..\..")).Path
$VenvDir = Join-Path $RepoRoot "venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# 1. Virtual environment
if (-not (Test-Path $Python)) {
    Step "Creating virtual environment at $VenvDir"
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & py -3.10 -m venv $VenvDir; if (-not $?) { & py -3 -m venv $VenvDir } }
    else { & python -m venv $VenvDir }
    if (-not (Test-Path $Python)) { throw "Could not create a virtual environment. Install Python 3.10+ from python.org." }
} else {
    Step "Using existing virtual environment at $VenvDir"
}
& $Python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'; print(sys.version)"
if (-not $?) { throw "Python 3.10 or newer is required." }

# 2. Python dependencies
$req = if ($Locked) { "requirements-lock.txt" } else { "requirements.txt" }
Step "Installing Python dependencies from $req"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ModuleDir $req)
if (-not $?) { throw "pip install failed." }

Step "Installing finclause_extraction (editable)"
& $Python -m pip install -e $ModuleDir
if (-not $?) { throw "Editable install failed." }

# 3. spaCy model check
Step "Checking spaCy model"
& $Python -c "import spacy; spacy.load('en_core_web_sm'); print('en_core_web_sm OK')"
if (-not $?) { & $Python -m spacy download en_core_web_sm }

# 4. Tesseract
Step "Checking Tesseract OCR"
$candidates = @(
    (Get-Command tesseract -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
    (Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Tesseract-OCR\tesseract.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Tesseract-OCR\tesseract.exe")
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $candidates -and $InstallTesseract) {
    Write-Host "Installing Tesseract with winget (a UAC prompt may appear)..."
    winget install --id UB-Mannheim.TesseractOCR -e --accept-source-agreements --accept-package-agreements
    $candidates = @(Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe") | Where-Object { Test-Path $_ }
}
if ($candidates) {
    $exe = @($candidates)[0]
    Write-Host "Tesseract found: $exe"
    & $exe --version | Select-Object -First 1
} else {
    Write-Warning ("Tesseract not found. Digital PDFs work; scanned PDFs need OCR. " +
        "Re-run with -InstallTesseract or: winget install --id UB-Mannheim.TesseractOCR -e")
}

# 5. Tests
if (-not $SkipTests) {
    Step "Running tests"
    Push-Location $ModuleDir
    try { & $Python -m pytest -q -rs } finally { Pop-Location }
    if (-not $?) { throw "Some tests failed (see output above)." }
}

Step "Done. Activate with:  $VenvDir\Scripts\Activate.ps1"
Write-Host "Try:  python $ModuleDir\scripts\make_sample_pdf.py; python $ModuleDir\scripts\run_extraction.py --input $ModuleDir\data\raw\synthetic_loan_agreement.pdf"
