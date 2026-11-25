"""Adapter that turns plain text into :class:`Page` objects.

Used by :func:`finclause_extraction.analyze_text` and by tests. Conventions:

* a form feed (``\\f``) separates pages;
* a blank line separates blocks (paragraphs);
* every text line becomes a :class:`TextLine`.

Coordinates are *synthetic* (derived from line/column positions: 6pt per
character, 14pt per line, 72pt margins) so that indentation-based rules behave
consistently. They do not correspond to any real document layout.
"""

from __future__ import annotations

import re

from .schemas import BoundingBox, ExtractionMethod, Page, TextBlock, TextLine, Word

_CHAR_W = 6.0
_LINE_H = 14.0
_MARGIN = 72.0
_WORD_RE = re.compile(r"\S+")


def pages_from_text(text: str, document_id: str = "doc_text") -> list[Page]:
    """Build pages/blocks/lines/words from plain text (see module docstring)."""
    pages: list[Page] = []
    for page_index, page_text in enumerate(text.split("\f"), start=1):
        blocks: list[TextBlock] = []
        y = _MARGIN
        for raw_block in re.split(r"\n[ \t]*\n", page_text):
            raw_lines = [l.rstrip() for l in raw_block.split("\n") if l.strip()]
            if not raw_lines:
                continue
            block_id = f"p{page_index:03d}_b{len(blocks):03d}"
            lines: list[TextLine] = []
            for raw_line in raw_lines:
                indent = len(raw_line) - len(raw_line.lstrip())
                content = raw_line.lstrip()
                x0 = _MARGIN + indent * _CHAR_W
                words = [
                    Word(
                        text=m.group(0),
                        bbox=BoundingBox(x0=x0 + m.start() * _CHAR_W, y0=y, x1=x0 + m.end() * _CHAR_W, y1=y + 10),
                        char_start=m.start(),
                        char_end=m.end(),
                    )
                    for m in _WORD_RE.finditer(content)
                ]
                lines.append(
                    TextLine(
                        line_id=f"{block_id}_l{len(lines):03d}",
                        text=content,
                        bbox=BoundingBox(x0=x0, y0=y, x1=x0 + len(content) * _CHAR_W, y1=y + 10),
                        words=words,
                    )
                )
                y += _LINE_H
            y += _LINE_H
            blocks.append(
                TextBlock(
                    block_id=block_id,
                    document_id=document_id,
                    page_number=page_index,
                    text="\n".join(l.text for l in lines),
                    bbox=BoundingBox.union(l.bbox for l in lines),
                    extraction_method=ExtractionMethod.TEXT_INPUT,
                    lines=lines,
                )
            )
        chars = sum(len(re.sub(r"\s", "", b.text)) for b in blocks)
        pages.append(
            Page(
                page_number=page_index,
                width=595.0,
                height=max(842.0, y + _MARGIN),
                extraction_method=ExtractionMethod.TEXT_INPUT,
                native_char_count=chars,
                is_empty=chars == 0,
                needs_ocr=False,
                text="\n\n".join(b.text for b in blocks),
                blocks=blocks,
            )
        )
    return pages
