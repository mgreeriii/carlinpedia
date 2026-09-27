"""Deterministic transcript cleanup and sentence-unit splitting.

Every later step refers to text only by sentence-unit index, which is what keeps quotes verbatim.
"""

from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass

NORMALIZER_VERSION = "1"

_CUE = re.compile(
    r"[\[(]\s*(?:audience\s+)?(applause|applauds|laughter|laughs|laughing|cheering|cheers|clapping)[^\])\n]*[\])]",
    re.IGNORECASE,
)
_TIMESTAMP = re.compile(r"(?:^[ \t]*\[?|\[)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,]\d+)?\]?", re.MULTILINE)
_BRACKETED = re.compile(r"\[[^\]\n]{1,80}\]")
_SPEAKER = re.compile(
    r"^[ \t]*(?:george carlin|george|carlin|announcer|audience member|audience)[ \t]*:[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
_SENTENCE_END = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019)\]]*(\s+)|(\n\n)")
_LAUGH_KINDS = {"laughter", "laughs", "laughing"}


@dataclass(frozen=True)
class Unit:
    index: int
    char_start: int
    char_end: int
    ts: float | None = None


@dataclass(frozen=True)
class Cue:
    kind: str  # "laughter" or "applause"
    unit_index: int  # the unit the cue follows


@dataclass(frozen=True)
class NormalizedDoc:
    text: str
    units: tuple[Unit, ...]
    cues: tuple[Cue, ...]

    def unit_text(self, index: int) -> str:
        unit = self.units[index]
        return self.text[unit.char_start:unit.char_end]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{NORMALIZER_VERSION}\n{self.text}".encode()).hexdigest()


def _remove_markup(raw: str, has_timestamps: bool, strip_patterns: list[str]) -> tuple[str, list[tuple[int, str, object]]]:
    """Strip cues, timestamps, stage directions, and speaker labels. Returns text plus (position, kind, value) markers."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    for pattern in strip_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    text = _SPEAKER.sub("", text)

    patterns = [("cue", _CUE)]
    if has_timestamps:
        patterns.append(("ts", _TIMESTAMP))
    patterns.append(("drop", _BRACKETED))

    matches = []
    for kind, pattern in patterns:
        for m in pattern.finditer(text):
            matches.append((m.start(), m.end(), kind, m))
    matches.sort(key=lambda item: (item[0], -item[1]))

    out, markers, pos = [], [], 0
    for start, end, kind, m in matches:
        if start < pos:  # overlaps an earlier match
            continue
        out.append(text[pos:start])
        cursor = sum(len(piece) for piece in out)
        if kind == "cue":
            markers.append((cursor, "cue", "laughter" if m.group(1).lower() in _LAUGH_KINDS else "applause"))
        elif kind == "ts":
            hours, minutes, seconds = m.group(1), m.group(2), m.group(3)
            markers.append((cursor, "ts", int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)))
        out.append(" ")
        pos = end
    out.append(text[pos:])
    return "".join(out), markers


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space (or a paragraph break). Returns text and an old->new offset map."""
    out: list[str] = []
    mapping = [0] * (len(text) + 1)
    size = 0
    for m in re.finditer(r"\s+|\S+", text):
        token = m.group()
        if token[0].isspace():
            replacement = "" if size == 0 else ("\n\n" if token.count("\n") >= 2 else " ")
            for i in range(m.start(), m.end()):
                mapping[i] = size
            out.append(replacement)
            size += len(replacement)
        else:
            for i in range(m.start(), m.end()):
                mapping[i] = size + (i - m.start())
            out.append(token)
            size += len(token)
    mapping[len(text)] = size
    collapsed = "".join(out)
    stripped = collapsed.rstrip()
    if len(stripped) < len(collapsed):
        mapping = [min(x, len(stripped)) for x in mapping]
    return stripped, mapping


def _split_long(start: int, end: int, text: str, max_chars: int) -> list[tuple[int, int]]:
    spans = []
    while end - start > max_chars:
        cut = text.rfind(" ", start, start + max_chars)
        if cut <= start:
            cut = start + max_chars
        spans.append((start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if start < end:
        spans.append((start, end))
    return spans


def _sentence_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    raw_spans, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        end = m.start(1) if m.group(1) is not None else m.start(2)
        raw_spans.append((start, end))
        start = m.end()
    raw_spans.append((start, len(text)))
    spans = []
    for s, e in raw_spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            spans.extend(_split_long(s, e, text, max_chars))
    return spans


def normalize(
    raw: str,
    has_timestamps: bool = False,
    strip_patterns: list[str] | None = None,
    max_unit_chars: int = 400,
) -> NormalizedDoc:
    cleaned, markers = _remove_markup(raw, has_timestamps, strip_patterns or [])
    text, mapping = _collapse_whitespace(cleaned)
    spans = _sentence_spans(text, max_unit_chars)

    ts_markers = sorted((mapping[pos], value) for pos, kind, value in markers if kind == "ts")
    ts_positions = [pos for pos, _ in ts_markers]
    units = []
    for index, (start, end) in enumerate(spans):
        ts = None
        if ts_markers:
            i = bisect.bisect_right(ts_positions, start) - 1
            ts = float(ts_markers[i][1]) if i >= 0 else None
        units.append(Unit(index, start, end, ts))

    ends = [u.char_end for u in units]
    cues = []
    for pos, kind, value in markers:
        if kind != "cue" or not units:
            continue
        unit_index = max(0, bisect.bisect_right(ends, mapping[pos]) - 1)
        cues.append(Cue(value, unit_index))
    return NormalizedDoc(text=text, units=tuple(units), cues=tuple(cues))
