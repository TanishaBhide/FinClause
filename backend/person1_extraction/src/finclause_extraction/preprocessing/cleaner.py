"""Text cleaning that never loses track of the original text.

:func:`clean_text` returns a :class:`CleanedText` holding the original string,
the cleaned string and an offset map so that any span in the cleaned text can
be mapped back to the exact span in the original text.

What cleaning does
    * removes zero-width characters, soft hyphens and control characters;
    * expands typographic ligatures (``ﬁ`` -> ``fi``);
    * normalises non-breaking/thin spaces, smart quotes, Unicode minus and dashes;
    * re-joins words hyphenated across line breaks (``pay-\\nment`` -> ``payment``)
      while keeping real compound hyphens (``pre-\\npayment`` -> ``pre-payment``);
    * turns single line breaks into spaces, keeps paragraph breaks as ``\\n``;
    * collapses repeated spaces.

What cleaning never does
    It never removes or rewrites digits, currency symbols (``₹``, ``Rs.``, ``INR``),
    ``%``, decimal points, commas inside numbers, ``/-``, signs, dates, clause
    numbers, or letters. The tests in ``tests/test_preprocessing.py`` guard this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Union

#: A replacement segment: ("lit", "text") or ("grp", group_number)
Segment = tuple[str, Union[str, int]]

_ZERO_WIDTH = "​‌‍⁠﻿"
_SPACES = "              　\t\f\v"
_CHAR_MAP: dict[str, str] = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "−": "-", "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "…": "...",
    "\r": "\n",
}
for _c in _SPACES:
    _CHAR_MAP[_c] = " "

# Prefixes for which a line-final hyphen is a real compound hyphen.
_KEEP_HYPHEN_PREFIXES = {
    "pre", "non", "co", "re", "sub", "self", "part", "post", "semi", "multi", "anti", "inter", "over",
    "under", "ex", "e", "third", "cross", "follow", "pro", "quasi", "auto", "on", "off", "per",
}

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")
_SOFT_HYPHEN_BREAK_RE = re.compile(r"­[ \t]*\n[ \t]*")
_HYPHEN_BREAK_RE = re.compile(r"(\b[A-Za-z]+)-[ \t]*\n[ \t]*([a-z])")
_NEWLINES_RE = re.compile(r"[ \t]*\n[\s]*")
_MULTI_SPACE_RE = re.compile(r" {2,}")


@dataclass
class CleanedText:
    """Cleaned text plus a map back to the original.

    ``offset_map[i]`` is the index in ``original`` of the character that produced
    ``text[i]``. ``offset_map`` has ``len(text) + 1`` entries; the last one is
    ``len(original)``-equivalent sentinel for end positions.
    """

    original: str
    text: str
    offset_map: list[int] = field(repr=False)

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        """Map a ``[start, end)`` span of the cleaned text to the original text."""
        if not 0 <= start <= end <= len(self.text):
            raise ValueError(f"span ({start}, {end}) outside cleaned text of length {len(self.text)}")
        if start == end:
            pos = self.offset_map[start]
            return pos, pos
        return self.offset_map[start], self.offset_map[end - 1] + 1


def _apply(
    text: str, mapping: list[int], pattern: re.Pattern, repl: Callable[[re.Match], list[Segment]]
) -> tuple[str, list[int]]:
    """Regex substitution that keeps ``mapping`` (one original index per char) in sync."""
    out: list[str] = []
    out_map: list[int] = []
    pos = 0
    for m in pattern.finditer(text):
        out.append(text[pos:m.start()])
        out_map.extend(mapping[pos:m.start()])
        for kind, value in repl(m):
            if kind == "grp":
                g0, g1 = m.span(value)  # type: ignore[arg-type]
                out.append(text[g0:g1])
                out_map.extend(mapping[g0:g1])
            else:
                anchor = mapping[m.start()] if m.start() < len(mapping) else (mapping[-1] if mapping else 0)
                out.append(value)  # type: ignore[arg-type]
                out_map.extend([anchor] * len(value))  # type: ignore[arg-type]
        pos = m.end()
    out.append(text[pos:])
    out_map.extend(mapping[pos:])
    return "".join(out), out_map


def _char_level(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    out_map: list[int] = []
    for i, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        repl = _CHAR_MAP.get(ch)
        if repl is None:
            out.append(ch)
            out_map.append(i)
        else:
            out.append(repl)
            out_map.extend([i] * len(repl))
    return "".join(out), out_map


def _dehyphenate(m: re.Match) -> list[Segment]:
    if m.group(1).lower() in _KEEP_HYPHEN_PREFIXES:
        return [("grp", 1), ("lit", "-"), ("grp", 2)]
    return [("grp", 1), ("grp", 2)]


def _newlines(preserve_paragraphs: bool) -> Callable[[re.Match], list[Segment]]:
    def repl(m: re.Match) -> list[Segment]:
        if preserve_paragraphs and m.group(0).count("\n") >= 2:
            return [("lit", "\n")]
        return [("lit", " ")]

    return repl


def clean_text(original: str, preserve_paragraphs: bool = True) -> CleanedText:
    """Clean ``original`` and return the cleaned text with an offset map.

    Args:
        original: Raw extracted text. It is never modified.
        preserve_paragraphs: Keep blank-line paragraph breaks as a single ``\\n``.
            Single line breaks always become spaces.
    """
    text, mapping = _char_level(original)
    text, mapping = _apply(text, mapping, _CONTROL_RE, lambda m: [])
    text, mapping = _apply(text, mapping, _SOFT_HYPHEN_BREAK_RE, lambda m: [])
    text, mapping = _apply(text, mapping, re.compile("­"), lambda m: [])
    text, mapping = _apply(text, mapping, _HYPHEN_BREAK_RE, _dehyphenate)
    text, mapping = _apply(text, mapping, _NEWLINES_RE, _newlines(preserve_paragraphs))
    text, mapping = _apply(text, mapping, _MULTI_SPACE_RE, lambda m: [("lit", " ")])

    # strip leading/trailing whitespace
    start = 0
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    text, mapping = text[start:end], mapping[start:end]
    sentinel = (mapping[-1] + 1) if mapping else 0
    return CleanedText(original=original, text=text, offset_map=mapping + [sentinel])


def normalize_whitespace(text: str) -> str:
    """Convenience wrapper returning only the cleaned string."""
    return clean_text(text).text
