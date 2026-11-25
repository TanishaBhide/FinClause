"""Tests for the command-line runner (executed as a real subprocess)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_extraction.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)


def test_cli_writes_json_and_summary(agreement_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "out" / "result.json"
    proc = _run("--input", str(agreement_pdf), "--output", str(out), "--no-ocr", "--quiet")
    assert proc.returncode == 0, proc.stderr
    assert "Total pages       : 2" in proc.stdout
    assert "Total clauses" in proc.stdout and "Financial entities" in proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["extraction_status"] == "success" and data["entities"]


def test_cli_missing_input(tmp_path: Path) -> None:
    proc = _run("--input", str(tmp_path / "nope.pdf"), "--output", str(tmp_path / "x.json"))
    assert proc.returncode == 1
    assert "not found" in proc.stderr


def test_cli_invalid_pdf(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    out = tmp_path / "bad.json"
    proc = _run("--input", str(bad), "--output", str(out), "--quiet")
    assert proc.returncode == 1
    assert json.loads(out.read_text(encoding="utf-8"))["extraction_status"] == "failed"
