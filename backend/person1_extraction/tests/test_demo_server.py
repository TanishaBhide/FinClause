"""Tests for the local demo viewer server (real HTTP requests to a server on a free port)."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "demo_server.py"
_spec = importlib.util.spec_from_file_location("demo_server", SCRIPT)
demo_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo_server)  # type: ignore[union-attr]


@pytest.fixture(scope="module")
def base_url():
    server = demo_server.create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _req(url: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.headers.get("Content-Type"), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type"), err.read()


def test_index_page_served(base_url: str) -> None:
    status, ctype, body = _req(base_url + "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"FinClause" in body and b"/api/analyze" in body


def test_sample_analysis_and_page_render(base_url: str) -> None:
    status, ctype, body = _req(base_url + "/api/sample?kind=digital&ocr=0", data=b"")
    assert status == 200 and ctype.startswith("application/json")
    result = json.loads(body)
    assert result["extraction_status"] == "success" and len(result["entities"]) == 20
    assert all(not l["words"] for p in result["pages"] for b in p["blocks"] for l in b["lines"])  # trimmed
    status, ctype, png = _req(f"{base_url}/api/page/{result['document_id']}/1.png?zoom=1")
    assert status == 200 and ctype == "image/png" and png.startswith(b"\x89PNG")
    assert _req(f"{base_url}/api/page/{result['document_id']}/99.png")[0] == 404


def test_upload_pdf(base_url: str, agreement_pdf: Path) -> None:
    status, _, body = _req(base_url + "/api/analyze?ocr=0", data=agreement_pdf.read_bytes(),
                           headers={"X-Filename": "my%20agreement.pdf", "Content-Type": "application/pdf"})
    result = json.loads(body)
    assert status == 200 and result["filename"] == "my agreement.pdf"
    assert result["extraction_status"] == "success"


def test_invalid_upload_returns_failed_result(base_url: str) -> None:
    status, _, body = _req(base_url + "/api/analyze", data=b"not a pdf", headers={"X-Filename": "x.pdf"})
    assert status == 200 and json.loads(body)["extraction_status"] == "failed"


def test_errors(base_url: str) -> None:
    assert _req(base_url + "/api/page/unknown_doc/1.png")[0] == 404
    assert _req(base_url + "/nope")[0] == 404
    assert _req(base_url + "/api/analyze", data=b"")[0] == 400
