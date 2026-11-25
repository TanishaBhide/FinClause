"""Clause segmentation for loan agreements.

Segmentation works on *lines* (not sentences): a new clause starts only at a line
that begins with a clause marker, is a heading, or starts a new paragraph. A
clause therefore keeps all of its sentences together and may span several
blocks and pages.

Recognised markers (at the start of a line)::

    Section 7 - Fees and Charges | Article II | Schedule A | Clause 5   (sections)
    1.   1.1   1.1.1   1)                                               (decimal)
    2(a)                                                                (number + letter)
    (a)  (iv)  (A)  (1)   a)  iv)   a.  iv.                             (list items)
    •  -  *  ▪                                                         (bullets)

Hierarchy rules (documented, deterministic):
    * a section or unnumbered heading closes everything below it;
    * a decimal clause ``N.M`` becomes a child of the nearest open clause whose
      number is a prefix (``N``), otherwise of the enclosing section/heading;
    * a list marker of a style already open (e.g. another ``(b)``) is a sibling;
      a new style (e.g. ``(i)`` under ``(a)``) nests one level deeper;
    * an unnumbered paragraph becomes a child of the innermost open clause,
      unless it is not indented relative to an open list item (then the list
      item is closed first).

Guards against false clause boundaries:
    * decimals / amounts (``2.5 lakh``, ``12.5%``, ``10.75 p.a.``) are rejected by
      a unit-word check and by requiring whitespace after the number;
    * a mid-sentence line wrap (previous line not ending in ``.;:!?``) only starts a
      new decimal clause if the number is a plausible successor of the last one;
    * ``Clause 5 of this Agreement ...`` (a cross reference) is not a heading.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..config import ExtractionConfig
from ..schemas import ArtifactLine, BoundingBox, Clause, ClauseType, Page, SourceLocation, Word
from .cleaner import CleanedText, clean_text

# --------------------------------------------------------------------------- #
# Data holders
# --------------------------------------------------------------------------- #


@dataclass
class SegLine:
    """A line in reading order with the context the segmenter needs."""

    text: str
    page_number: int
    block_id: str
    line_id: str
    bbox: BoundingBox
    words: list[Word]
    is_bold: Optional[bool]
    first_in_block: bool
    block_line_count: int
    next_text_in_block: Optional[str]
    next_is_bold: Optional[bool]


@dataclass
class LineSpan:
    """Where a source line sits inside ``Clause.original_text``."""

    line: SegLine
    start: int
    end: int


@dataclass
class Marker:
    kind: str  # "section" | "decimal" | "list:<style>_<type>" | "bullet"
    label: str  # marker as written, e.g. "1.1", "(a)", "Section 7"
    parts: tuple[int, ...] = ()
    value: Optional[str] = None  # list value: "a", "ii", "3"
    parent_number: Optional[int] = None  # for "2(a)"
    title: Optional[str] = None
    has_body: bool = False


@dataclass
class _Node:
    marker: Optional[Marker]
    kind: str
    clause_type: ClauseType
    lines: list[SegLine] = field(default_factory=list)
    title: Optional[str] = None
    numbering_path: Optional[str] = None
    parent: Optional["_Node"] = None
    awaiting_body: bool = False
    first_x0: float = 0.0
    index: int = 0

    @property
    def parts(self) -> tuple[int, ...]:
        return self.marker.parts if self.marker else ()


@dataclass
class SegmentedClause:
    """A clause plus the internal data needed to trace entities to the source."""

    clause: Clause
    cleaned: CleanedText
    line_spans: list[LineSpan]


@dataclass
class SegmentationResult:
    clauses: list[SegmentedClause]
    artifacts: list[ArtifactLine]


# --------------------------------------------------------------------------- #
# Regexes and helpers
# --------------------------------------------------------------------------- #
_SECTION_RE = re.compile(
    r"^(?P<kw>section|clause|article|schedule|annexure|annex|part|chapter)\s+"
    r"(?P<num>\d{1,3}(?:\.\d{1,3})*|[IVXLC]{1,6}|[A-Z])\b(?P<dot>\.(?!\d))?\s*(?P<sep>[-:–—]\s*)?(?P<rest>.*)$",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"^(?P<num>\d{1,3}(?:\.\d{1,3}){0,4})(?P<term>\.|\))?(?=\s|$)")
_NUM_PAREN_RE = re.compile(r"^(?P<num>\d{1,3})\s?\((?P<val>[a-z]{1,4}|[A-Z])\)(?=\s|$)")
_PAREN_RE = re.compile(r"^\(\s?(?P<val>[A-Za-z]{1,5}|\d{1,3})\s?\)(?=\s|$)")
_RPAREN_RE = re.compile(r"^(?P<val>[A-Za-z]{1,4}|\d{1,3})\)(?=\s)")
_DOT_RE = re.compile(r"^(?P<val>[A-Za-z]|[ivxlcIVXLC]{2,5})\.(?=\s)")
_BULLET_RE = re.compile(r"^(?P<b>[•●▪■◦▸►‣⁃∙·*]|-|–)\s+")
_ROMAN_RE = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")
_TERMINAL_RE = re.compile(r"[.;:!?][\"')\]]*\s*$")
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|-\s*\d+\s*-|\d+\s*/\s*\d+|\d{1,4})\s*$", re.IGNORECASE)
_PAGE_WORD_RE = re.compile(r"^\s*page\s*\d+", re.IGNORECASE)

#: Words that, immediately after a number at line start, show it is a value not a clause number.
UNIT_WORDS = {
    "%", "percent", "per", "pct", "lakh", "lakhs", "lac", "lacs", "crore", "crores", "cr", "cr.", "thousand",
    "million", "billion", "rupees", "rs", "rs.", "inr", "months", "month", "years", "year", "days", "day",
    "weeks", "p.a.", "p.a", "pa", "p.m.", "bps", "basis", "emis", "emi", "instalments", "installments", "times",
    "x", "to", "and", "or", "-", "/-",
}
#: Unit words rejected even when capitalised ("2.5 Lakh", "1.5 Crore", "2.5 INR").
ALWAYS_UNIT_WORDS = {"lakh", "lakhs", "lac", "lacs", "crore", "crores", "rupees", "rs", "rs.", "inr", "percent", "%", "p.a.", "bps"}
_STOPWORDS = {"of", "and", "the", "for", "to", "in", "on", "a", "an", "or", "by", "with", "at", "as", "under", "&"}


def roman_to_int(s: str) -> Optional[int]:
    s = s.lower()
    if not s or not _ROMAN_RE.match(s):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = values[ch]
        total += -v if i + 1 < len(s) and values[s[i + 1]] > v else v
    return total


def ends_terminal(text: str) -> bool:
    return bool(_TERMINAL_RE.search(text))


def _starts_lower(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and stripped[0].islower()


def is_title_like(text: str, bold: Optional[bool] = None, next_text: Optional[str] = None) -> bool:
    """Heuristic: short, unpunctuated, capitalised text that reads like a title."""
    s = text.strip()
    if not s or len(s) > 90 or not re.search(r"[A-Za-z]{2,}", s):
        return False
    words = s.split()
    if len(words) > 12 or s[-1] in ".;,":
        return False
    first_alpha = next((c for c in s if c.isalpha()), "")
    if not first_alpha.isupper():
        return False
    if next_text is not None and _starts_lower(next_text):
        return False  # the line continues as a sentence
    letters = [c for c in s if c.isalpha()]
    if all(c.isupper() for c in letters) or bold:
        return True
    significant = [w for w in words if len(re.sub(r"\W", "", w)) >= 3 and w.lower() not in _STOPWORDS]
    return bool(significant) and all(w[0].isupper() for w in significant)


def plausible_successor(prev: Optional[tuple[int, ...]], parts: tuple[int, ...]) -> bool:
    """True if ``parts`` could follow ``prev`` in a numbered list (1.2 -> 1.3, 1.2.1, 2, 2.1)."""
    if prev is None:
        return parts in {(1,), (1, 1)} or (len(parts) >= 2 and parts[-1] == 1)
    for k in range(len(prev)):
        if parts == prev[:k] + (prev[k] + 1,) or parts == prev[:k] + (prev[k] + 1, 1):
            return True
    return parts == prev + (1,)


# --------------------------------------------------------------------------- #
# Header / footer detection
# --------------------------------------------------------------------------- #
def detect_artifact_lines(pages: list[Page], config: ExtractionConfig) -> list[ArtifactLine]:
    """Mark running headers/footers and page-number lines (``TextLine.is_artifact``).

    They stay in ``pages`` but are excluded from clauses. Returned for reporting.
    """
    if not config.detect_headers_footers:
        return []
    artifacts: list[ArtifactLine] = []
    band: dict[str, list] = defaultdict(list)
    pages_with_text = [p for p in pages if p.blocks]
    for page in pages_with_text:
        top, bottom = page.height * config.header_footer_margin_ratio, page.height * (1 - config.header_footer_margin_ratio)
        for block in page.blocks:
            for line in block.lines:
                in_band = line.bbox.y1 <= top or line.bbox.y0 >= bottom
                if _PAGE_WORD_RE.match(line.text) and _PAGE_NUM_RE.match(line.text) or (in_band and _PAGE_NUM_RE.match(line.text)):
                    line.is_artifact = True
                    artifacts.append(ArtifactLine(page_number=page.page_number, line_id=line.line_id, text=line.text, reason="page_number"))
                elif in_band:
                    key = re.sub(r"\d+", "#", line.text.strip().lower())
                    band[key].append((page.page_number, line))
    threshold = max(2, math.ceil(0.5 * len(pages_with_text)))
    for key, items in band.items():
        if key and len({p for p, _ in items}) >= threshold:
            for page_number, line in items:
                line.is_artifact = True
                artifacts.append(ArtifactLine(page_number=page_number, line_id=line.line_id, text=line.text, reason="repeated_header_footer"))
    return artifacts


# --------------------------------------------------------------------------- #
# Marker detection
# --------------------------------------------------------------------------- #
class _MarkerDetector:
    def __init__(self) -> None:
        self.last_decimal: Optional[tuple[int, ...]] = None
        self.last_list: dict[str, tuple[str, str]] = {}  # style -> (type, value)

    def reset_lists(self) -> None:
        self.last_list.clear()

    def _list_type(self, style: str, val: str) -> Optional[str]:
        if val.isdigit():
            return "num"
        upper = val.isupper()
        low = val.lower()
        prev_type, prev_val = self.last_list.get(style, (None, None))
        roman = roman_to_int(low)
        if len(low) == 1:
            if roman is None:
                return "ALPHA" if upper else "alpha"
            # ambiguous single letter: i, v, x, l, c, d, m
            if prev_type in ("alpha", "ALPHA") and prev_val and ord(low) == ord(prev_val.lower()) + 1:
                return "ALPHA" if upper else "alpha"
            if low == "i":
                return "ROMAN" if upper else "roman"
            if prev_type in ("roman", "ROMAN") and prev_val and roman_to_int(prev_val) == roman - 1:
                return "ROMAN" if upper else "roman"
            return "ALPHA" if upper else "alpha"
        if roman is not None:
            return "ROMAN" if upper else "roman"
        if len(set(low)) == 1:  # "aa", "bb"
            return "ALPHA" if upper else "alpha"
        return None

    def detect(self, line: SegLine, prev: Optional[SegLine]) -> Optional[Marker]:
        raw = line.text
        s = raw.lstrip()
        boundary_ok = prev is None or prev.block_id != line.block_id or ends_terminal(prev.text)

        m = _SECTION_RE.match(s)
        if m:
            rest = m.group("rest").strip()
            sep = m.group("sep") or m.group("dot")
            kw = m.group("kw")
            label = f"{kw} {m.group('num')}"
            num = m.group("num")
            title = rest if rest and is_title_like(rest, line.is_bold or rest.isupper(), line.next_text_in_block) else None
            if not rest or sep or title and (line.is_bold or rest.isupper() or boundary_ok):
                if kw.lower() == "clause" and re.fullmatch(r"\d+(?:\.\d+)*", num):
                    parts = tuple(int(p) for p in num.split("."))
                    return Marker("decimal", label, parts=parts, title=title, has_body=bool(rest) and not title)
                return Marker("section", label, title=title, has_body=bool(rest) and not title)

        m = _NUM_PAREN_RE.match(s)
        if m:
            val = m.group("val")
            typ = self._list_type("paren", val)
            if typ:
                return Marker(f"list:paren_{typ}", m.group(0), value=val, parent_number=int(m.group("num")))

        m = _DECIMAL_RE.match(s)
        if m:
            num, term = m.group("num"), m.group("term")
            parts = tuple(int(p) for p in num.split("."))
            after = s[m.end():].split()
            raw_next = after[0].strip(",;:") if after else ""
            next_word = raw_next.lower()
            is_unit = next_word in UNIT_WORDS and (not raw_next[:1].isupper() or next_word in ALWAYS_UNIT_WORDS)
            valid = bool(len(parts) > 1 or term) and not is_unit and not next_word.startswith("%")
            if valid and len(parts) == 1 and term == "." and not boundary_ok and not plausible_successor(self.last_decimal, parts):
                valid = False
            if valid and not boundary_ok and not plausible_successor(self.last_decimal, parts):
                valid = False
            if valid:
                rest = s[m.end():].strip()
                title = rest if rest and is_title_like(rest, line.is_bold, line.next_text_in_block) else None
                return Marker("decimal", m.group(0), parts=parts, title=title, has_body=bool(rest) and not title)

        for style, regex in (("paren", _PAREN_RE), ("rparen", _RPAREN_RE), ("dot", _DOT_RE)):
            m = regex.match(s)
            if not m:
                continue
            if style != "paren" and not boundary_ok:
                continue
            val = m.group("val")
            typ = self._list_type(style, val)
            if typ is None:
                continue
            if style == "dot" and typ in ("ALPHA",) and not boundary_ok:
                continue
            return Marker(f"list:{style}_{typ}", m.group(0), value=val)

        m = _BULLET_RE.match(s)
        if m and (boundary_ok or m.group("b") not in "-–"):
            return Marker("bullet", m.group("b"))
        return None


# --------------------------------------------------------------------------- #
# Tree building
# --------------------------------------------------------------------------- #
class _TreeBuilder:
    def __init__(self) -> None:
        self.nodes: list[_Node] = []
        self.stack: list[_Node] = []
        self.current: Optional[_Node] = None
        self.prev: Optional[SegLine] = None
        self.detector = _MarkerDetector()

    # -- helpers --
    def _new(self, node: _Node, line: SegLine, push: bool) -> None:
        node.index = len(self.nodes)
        node.first_x0 = line.bbox.x0
        node.lines.append(line)
        self.nodes.append(node)
        if push:
            self.stack.append(node)
        self.current = node

    def _top(self) -> Optional[_Node]:
        return self.stack[-1] if self.stack else None

    @staticmethod
    def _path_of(node: Optional[_Node]) -> Optional[str]:
        while node is not None:
            if node.numbering_path:
                return node.numbering_path
            node = node.parent
        return None

    def _is_heading(self, line: SegLine) -> bool:
        if not line.first_in_block:
            return False
        if not is_title_like(line.text, line.is_bold, line.next_text_in_block):
            return False
        cur = self.current
        if cur is not None and not cur.awaiting_body and cur.lines and not ends_terminal(cur.lines[-1].text) \
                and cur.clause_type not in (ClauseType.HEADING, ClauseType.SECTION) and not cur.title:
            return False  # previous sentence is unfinished: treat as continuation
        letters = [c for c in line.text if c.isalpha()]
        upper = bool(letters) and all(c.isupper() for c in letters)
        if line.block_line_count == 1:
            return True
        nxt = (line.next_text_in_block or "").lstrip()
        if _BULLET_RE.match(nxt) or _PAREN_RE.match(nxt):
            return True  # a short title directly followed by a list ("Key Facts" + bullets)
        return bool(line.is_bold or upper) and line.next_is_bold is not True

    # -- main entry --
    def feed(self, line: SegLine) -> None:
        marker = self.detector.detect(line, self.prev)
        if marker is not None:
            self._start_marked(marker, line)
        elif self._is_heading(line):
            self._start_heading(line)
        else:
            self._add_text(line)
        self.prev = line

    def _start_heading(self, line: SegLine) -> None:
        while self.stack and self.stack[-1].kind != "section":
            self.stack.pop()
        node = _Node(None, "heading", ClauseType.HEADING, title=line.text.strip(), parent=self._top())
        self.detector.reset_lists()
        self._new(node, line, push=True)

    def _start_marked(self, marker: Marker, line: SegLine) -> None:
        det = self.detector
        if marker.kind == "section":
            self.stack.clear()
            node = _Node(marker, "section", ClauseType.SECTION, title=marker.title,
                         numbering_path=marker.label, parent=None)
            det.reset_lists()
            self._new(node, line, push=True)
            return

        if marker.kind == "decimal":
            parts = marker.parts
            while self.stack:
                top = self.stack[-1]
                if top.kind in ("section", "heading"):
                    break
                if top.kind == "decimal" and len(top.parts) < len(parts) and parts[: len(top.parts)] == top.parts:
                    break
                self.stack.pop()
            ctype = ClauseType.CLAUSE if len(parts) == 1 else ClauseType.SUBCLAUSE
            node = _Node(marker, "decimal", ctype, title=marker.title, numbering_path=".".join(map(str, parts)),
                         parent=self._top(), awaiting_body=not marker.has_body)
            det.last_decimal = parts
            det.reset_lists()
            self._new(node, line, push=True)
            return

        # list items and bullets
        if marker.parent_number is not None:
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i].kind == "decimal" and self.stack[i].parts == (marker.parent_number,):
                    del self.stack[i + 1:]
                    break
        same = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i].kind == marker.kind), None)
        if same is not None:
            del self.stack[same:]
        else:
            # a list item at the same or smaller indent than an open list item is not nested in it
            while self.stack and self.stack[-1].kind.startswith(("list:", "bullet")) \
                    and line.bbox.x0 < self.stack[-1].first_x0 - 2:
                self.stack.pop()
        parent = self._top()
        if marker.kind == "bullet":
            node = _Node(marker, "bullet", ClauseType.BULLET, parent=parent)
        else:
            base = self._path_of(parent)
            if marker.parent_number is not None and (base is None or base.split("(")[0] != str(marker.parent_number)):
                base = str(marker.parent_number)
            path = f"{base or ''}({marker.value})"
            node = _Node(marker, marker.kind, ClauseType.SUBCLAUSE, numbering_path=path, parent=parent)
            style = marker.kind.split(":")[1].split("_")[0]
            det.last_list[style] = (marker.kind.split("_", 1)[1], marker.value or "")
        self._new(node, line, push=True)

    def _new_paragraph(self, line: SegLine) -> None:
        while self.stack and self.stack[-1].kind.startswith(("list:", "bullet")) \
                and line.bbox.x0 <= self.stack[-1].first_x0 + 1.0:
            self.stack.pop()
        node = _Node(None, "paragraph", ClauseType.PARAGRAPH, parent=self._top())
        self._new(node, line, push=False)

    def _add_text(self, line: SegLine) -> None:
        cur = self.current
        if cur is None:
            self._new_paragraph(line)
            return
        if cur.awaiting_body:
            cur.lines.append(line)
            cur.awaiting_body = False
            return
        if cur.clause_type in (ClauseType.HEADING,) or (cur.clause_type == ClauseType.SECTION and not cur.marker.has_body):
            if self.prev is not None and self.prev.block_id == line.block_id and not ends_terminal(self.prev.text) \
                    and _starts_lower(line.text):
                cur.lines.append(line)  # wrapped heading
            else:
                self._new_paragraph(line)
            return
        same_block = self.prev is not None and self.prev.block_id == line.block_id
        if same_block or not ends_terminal(cur.lines[-1].text) or _starts_lower(line.text):
            cur.lines.append(line)
        else:
            self._new_paragraph(line)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _collect_lines(pages: list[Page]) -> list[SegLine]:
    out: list[SegLine] = []
    for page in pages:
        for block in page.blocks:
            lines = [l for l in block.lines if not l.is_artifact and l.text.strip()]
            for i, line in enumerate(lines):
                nxt = lines[i + 1] if i + 1 < len(lines) else None
                out.append(
                    SegLine(
                        text=line.text,
                        page_number=page.page_number,
                        block_id=block.block_id,
                        line_id=line.line_id,
                        bbox=line.bbox,
                        words=line.words,
                        is_bold=line.is_bold,
                        first_in_block=i == 0,
                        block_line_count=len(lines),
                        next_text_in_block=nxt.text if nxt else None,
                        next_is_bold=nxt.is_bold if nxt else None,
                    )
                )
    return out


def _materialize(nodes: list[_Node]) -> list[SegmentedClause]:
    ids = {id(n): f"clause_{i + 1:04d}" for i, n in enumerate(nodes)}
    results: list[SegmentedClause] = []
    children: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        if n.parent is not None:
            children[id(n.parent)].append(ids[id(n)])

    for n in nodes:
        parts: list[str] = []
        spans: list[LineSpan] = []
        pos = 0
        for i, line in enumerate(n.lines):
            if i > 0:
                sep = "\n" if n.lines[i - 1].block_id == line.block_id else "\n\n"
                parts.append(sep)
                pos += len(sep)
            spans.append(LineSpan(line, pos, pos + len(line.text)))
            parts.append(line.text)
            pos += len(line.text)
        original = "".join(parts)
        cleaned = clean_text(original, preserve_paragraphs=True)

        locations: list[SourceLocation] = []
        for span in spans:
            last = locations[-1] if locations else None
            if last and last.block_id == span.line.block_id:
                last.line_ids.append(span.line.line_id)
                last.end_offset = span.end
                last.bbox = BoundingBox.union([last.bbox, span.line.bbox])
            else:
                locations.append(SourceLocation(page_number=span.line.page_number, block_id=span.line.block_id,
                                                line_ids=[span.line.line_id], bbox=span.line.bbox,
                                                start_offset=span.start, end_offset=span.end))

        level, heading, heading_id, anc = 0, n.title, ids[id(n)] if n.title else None, n.parent
        while anc is not None:
            level += 1
            if heading is None and anc.title:
                heading, heading_id = anc.title, ids[id(anc)]
            anc = anc.parent

        clause = Clause(
            clause_id=ids[id(n)],
            parent_clause_id=ids[id(n.parent)] if n.parent is not None else None,
            child_clause_ids=children.get(id(n), []),
            clause_type=n.clause_type,
            clause_number=n.marker.label.strip() if n.marker else None,
            numbering_path=n.numbering_path,
            level=level,
            heading=heading,
            heading_clause_id=heading_id,
            original_text=original,
            cleaned_text=cleaned.text,
            page_numbers=sorted({s.line.page_number for s in spans}),
            source_locations=locations,
        )
        results.append(SegmentedClause(clause=clause, cleaned=cleaned, line_spans=spans))
    return results


def segment_pages(pages: list[Page], config: Optional[ExtractionConfig] = None) -> SegmentationResult:
    """Segment extracted pages into a clause tree.

    Header/footer lines are detected first (see :func:`detect_artifact_lines`)
    and excluded from clauses.
    """
    config = config or ExtractionConfig()
    artifacts = detect_artifact_lines(pages, config)
    builder = _TreeBuilder()
    for line in _collect_lines(pages):
        builder.feed(line)
    return SegmentationResult(clauses=_materialize(builder.nodes), artifacts=artifacts)


# --------------------------------------------------------------------------- #
# Sentence splitting (abbreviation, decimal and amount aware)
# --------------------------------------------------------------------------- #
_NO_SPLIT_ABBREV = {
    "rs", "re", "no", "nos", "mr", "mrs", "ms", "dr", "sr", "jr", "st", "vs", "viz", "cl", "sec", "art", "fig",
    "vol", "approx", "i.e", "e.g", "u/s", "w.e.f", "a/c", "ref", "pvt", "sch", "para", "max", "min", "incl",
    "excl", "govt", "dept", "yr", "yrs", "mo", "hrs", "amt", "bal", "int", "nach", "ecs",
}
_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s+[\"'(\[]?[A-Z0-9₹])|\n")
_LAST_TOKEN_RE = re.compile(r"(\S+)$")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split text into sentence spans ``[(start, end), ...]``.

    Does not split on decimals (``12.5``), amounts (``Rs. 500``), common
    abbreviations (``viz.``, ``i.e.``, ``Pvt.``), initials, or a leading clause
    number (``1. DEFINITIONS``). Newlines (paragraph breaks) always split.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        if m.group(0) == "\n":
            end = m.start()
        else:
            before = text[start:m.start()]
            tok_m = _LAST_TOKEN_RE.search(before)
            token = tok_m.group(1) if tok_m else ""
            bare = token.lstrip("(\"'").lower()
            if bare in _NO_SPLIT_ABBREV:
                continue
            if len(bare) == 1 and bare.isalpha():
                continue  # initial or list letter
            if before.strip() == token and re.fullmatch(r"\(?[\dA-Za-z]{1,4}(\.\d{1,3})*\)?", token):
                continue  # leading clause number "1." / "(a)."
            end = m.end()
        seg = text[start:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((start + lead, start + len(seg.rstrip())))
        start = end if m.group(0) != "\n" else m.end()
    seg = text[start:]
    if seg.strip():
        lead = len(seg) - len(seg.lstrip())
        spans.append((start + lead, start + len(seg.rstrip())))
    return spans
