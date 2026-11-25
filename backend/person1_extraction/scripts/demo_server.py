"""Local demo viewer for the Person 1 extraction module.

Starts a small local web server (Python standard library only, no FastAPI) that lets
you upload a PDF in the browser, runs ``analyze_pdf_bytes`` on it and shows the result:
page images with entity bounding boxes, the clause tree, entities and warnings.

This is a developer/demo tool for inspecting extraction output. It is NOT the
product frontend (that belongs to Person 3) and is bound to localhost only.

    python scripts/demo_server.py                # opens http://127.0.0.1:8765
    python scripts/demo_server.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

MODULE_ROOT = Path(__file__).resolve().parent.parent
try:
    import finclause_extraction  # noqa: F401
except ImportError:
    sys.path.insert(0, str(MODULE_ROOT / "src"))

import pymupdf  # noqa: E402

from finclause_extraction import ExtractionConfig, analyze_pdf_bytes  # noqa: E402
from finclause_extraction.demo_documents import DEMO_DOCUMENTS, list_demo_documents  # noqa: E402
from finclause_extraction.samples import SYNTHETIC_AGREEMENT, build_pdf, build_scanned_pdf  # noqa: E402

logger = logging.getLogger("demo_server")
DEMO_PAGE = MODULE_ROOT / "demo" / "index.html"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class DocumentStore:
    """Keeps the most recent uploaded PDFs in memory so pages can be rendered."""

    def __init__(self, capacity: int = 20) -> None:
        self._docs: "OrderedDict[str, bytes]" = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def put(self, doc_id: str, data: bytes) -> None:
        with self._lock:
            self._docs[doc_id] = data
            self._docs.move_to_end(doc_id)
            while len(self._docs) > self._capacity:
                self._docs.popitem(last=False)

    def get(self, doc_id: str) -> Optional[bytes]:
        with self._lock:
            return self._docs.get(doc_id)


def render_page_png(data: bytes, page_number: int, zoom: float) -> bytes:
    """Render a 1-based page to PNG."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if not 1 <= page_number <= doc.page_count:
            raise IndexError(page_number)
        pix = doc[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")


def make_handler(store: DocumentStore, base_config: ExtractionConfig):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FinClauseDemo/1.0"

        def log_message(self, fmt: str, *args) -> None:  # route to logging
            logger.info("%s - %s", self.address_string(), fmt % args)

        # ---- helpers ----
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _analyze(self, data: bytes, filename: str, use_ocr: bool) -> None:
            cfg = base_config  # word boxes are kept: the viewer shows what OCR read, word by word
            result = analyze_pdf_bytes(data, filename, use_ocr=use_ocr, config=cfg)
            if result.extraction_status != "failed":
                store.put(result.document_id, data)
            self._json(result.model_dump(mode="json"))

        # ---- routes ----
        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, DEMO_PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/health":
                self._json({"status": "ok"})
            elif url.path == "/api/demos":
                self._json(list_demo_documents())
            elif url.path.startswith("/api/demo/") and url.path.endswith(".pdf"):
                doc = DEMO_DOCUMENTS.get(url.path[len("/api/demo/"):-4])
                if doc is None:
                    return self._error(404, "unknown demo document")
                scanned = parse_qs(url.query).get("scanned", ["0"])[0] == "1"
                body = doc.build(scanned=scanned)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{doc.doc_id}{"_scanned" if scanned else ""}.pdf"')
                self.end_headers()
                self.wfile.write(body)
            elif url.path.startswith("/api/page/"):
                parts = url.path.split("/")  # ['', 'api', 'page', doc_id, '3.png']
                if len(parts) != 5 or not parts[4].endswith(".png"):
                    return self._error(404, "not found")
                data = store.get(parts[3])
                if data is None:
                    return self._error(404, "document not loaded (re-upload it)")
                try:
                    zoom = min(3.0, max(0.5, float(parse_qs(url.query).get("zoom", ["1.5"])[0])))
                    png = render_page_png(data, int(parts[4][:-4]), zoom)
                except (ValueError, IndexError):
                    return self._error(404, "page not found")
                self._send(200, png, "image/png")
            else:
                self._error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            query = parse_qs(url.query)
            use_ocr = query.get("ocr", ["1"])[0] != "0"
            try:
                if url.path == "/api/analyze":
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0:
                        return self._error(400, "empty upload")
                    if length > MAX_UPLOAD_BYTES:
                        return self._error(413, f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                    data = self.rfile.read(length)
                    filename = Path(unquote(self.headers.get("X-Filename") or "upload.pdf")).name
                    self._analyze(data, filename, use_ocr)
                elif url.path.startswith("/api/demo/"):
                    doc = DEMO_DOCUMENTS.get(url.path[len("/api/demo/"):])
                    if doc is None:
                        return self._error(404, "unknown demo document")
                    scanned = query.get("scanned", ["0"])[0] == "1"
                    name = f"{doc.doc_id}{'_scanned' if scanned else ''}.pdf"
                    self._analyze(doc.build(scanned=scanned), name, use_ocr)
                elif url.path == "/api/sample":
                    kind = query.get("kind", ["digital"])[0]
                    if kind == "scanned":
                        self._analyze(build_scanned_pdf(None, SYNTHETIC_AGREEMENT), "synthetic_scanned.pdf", use_ocr)
                    else:
                        self._analyze(build_pdf(None, SYNTHETIC_AGREEMENT, footer_page_numbers=True),
                                      "synthetic_loan_agreement.pdf", use_ocr)
                else:
                    self._error(404, "not found")
            except Exception as exc:  # keep the demo server alive
                logger.exception("request failed")
                self._error(500, f"{exc.__class__.__name__}: {exc}")

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  config: Optional[ExtractionConfig] = None) -> ThreadingHTTPServer:
    """Create (but do not start) the demo server. ``port=0`` picks a free port."""
    return ThreadingHTTPServer((host, port), make_handler(DocumentStore(), config or ExtractionConfig()))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Local viewer for FinClause extraction results.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    p.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    server = create_server(args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"\nFinClause extraction viewer running at {url}\nPress Ctrl+C to stop.\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
