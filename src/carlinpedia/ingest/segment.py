"""Split a normalized transcript into bits and passages with Claude, validate the spans, and store them."""

from __future__ import annotations

import hashlib
import sqlite3

from pydantic import BaseModel

from carlinpedia.db.schema import delete_index_rows
from carlinpedia.ingest.normalize import NormalizedDoc
from carlinpedia.llm.client import LLM, load_prompt

SEGMENT_PROMPT = "segment_v1"
_CUE_MARK = {"laughter": "⟨laugh⟩", "applause": "⟨applause⟩"}


class PassageSpan(BaseModel):
    unit_start: int
    unit_end: int


class BitSegment(BaseModel):
    title: str
    summary: str
    unit_start: int
    unit_end: int
    passages: list[PassageSpan]


class SegmentationResult(BaseModel):
    bits: list[BitSegment]


class SegmentationFailed(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("segmentation failed validation twice:\n" + "\n".join(f"- {e}" for e in errors))
        self.errors = errors


def validate_segmentation(result: SegmentationResult, n_units: int, max_passage_units: int = 40) -> list[str]:
    """Check that bits tile units 0..n-1 and passages tile each bit. Returns human-readable problems."""
    if not result.bits:
        return ["no bits returned"]
    errors: list[str] = []
    expected = 0
    for bi, bit in enumerate(result.bits):
        label = f"bit {bi} ({bit.title!r})"
        if bit.unit_start != expected:
            errors.append(f"{label} starts at unit {bit.unit_start}, expected {expected}")
        if bit.unit_end < bit.unit_start:
            errors.append(f"{label} ends at unit {bit.unit_end}, before it starts")
        if not bit.passages:
            errors.append(f"{label} has no passages")
        p_expected = bit.unit_start
        for pi, p in enumerate(bit.passages):
            p_label = f"{label} passage {pi}"
            if p.unit_start != p_expected:
                errors.append(f"{p_label} starts at unit {p.unit_start}, expected {p_expected}")
            length = p.unit_end - p.unit_start + 1
            if length < 1:
                errors.append(f"{p_label} ends at unit {p.unit_end}, before it starts")
            elif length > max_passage_units:
                errors.append(f"{p_label} spans {length} units; the maximum is {max_passage_units}")
            p_expected = p.unit_end + 1
        if bit.passages and p_expected != bit.unit_end + 1:
            errors.append(f"{label} passages end at unit {p_expected - 1}, but the bit ends at {bit.unit_end}")
        expected = bit.unit_end + 1
    if expected != n_units:
        errors.append(f"bits cover units 0..{expected - 1}, but the transcript has units 0..{n_units - 1}")
    return errors


def render_units(doc: NormalizedDoc) -> str:
    marks: dict[int, list[str]] = {}
    for cue in doc.cues:
        marks.setdefault(cue.unit_index, []).append(_CUE_MARK[cue.kind])
    lines = []
    for unit in doc.units:
        suffix = "".join(f" {m}" for m in marks.get(unit.index, []))
        lines.append(f"[{unit.index}] {doc.unit_text(unit.index)}{suffix}")
    return "\n".join(lines)


def segment_document(
    llm: LLM, doc: NormalizedDoc, *, work_title: str, key: str, max_passage_units: int = 40
) -> SegmentationResult:
    """Ask Claude for a segmentation, retrying once with the validator's complaints."""
    n = len(doc.units)
    if n == 0:
        raise SegmentationFailed(["the transcript has no text after normalization"])
    system = load_prompt(SEGMENT_PROMPT)
    user = (
        f"Work: {work_title}\n"
        f"Units: {n} (numbered 0 to {n - 1})\n"
        f"Maximum passage length: {max_passage_units} units\n\n"
        f"{render_units(doc)}"
    )
    result = llm.parse(prompt_name=SEGMENT_PROMPT, key=key, system=system, user=user, output_model=SegmentationResult)
    errors = validate_segmentation(result, n, max_passage_units)
    if not errors:
        return result
    retry = (
        f"{user}\n\nYour previous segmentation had these problems. "
        "Return a corrected segmentation of the whole transcript:\n" + "\n".join(f"- {e}" for e in errors)
    )
    result = llm.parse(prompt_name=SEGMENT_PROMPT, key=key, system=system, user=retry, output_model=SegmentationResult)
    errors = validate_segmentation(result, n, max_passage_units)
    if errors:
        raise SegmentationFailed(errors)
    return result


def delete_work_content(conn: sqlite3.Connection, work_id: int) -> None:
    ids = [r["id"] for r in conn.execute(
        "SELECT p.id FROM passage p JOIN bit b ON b.id = p.bit_id WHERE b.work_id = ?", (work_id,)
    )]
    delete_index_rows(conn, ids)
    conn.execute("DELETE FROM bit WHERE work_id = ?", (work_id,))


def store_segmentation(conn: sqlite3.Connection, work_id: int, doc: NormalizedDoc, result: SegmentationResult) -> int:
    """Replace the work's bits and passages. The caller owns the transaction. Returns the passage count."""
    delete_work_content(conn, work_id)
    count = 0
    for bit_ordinal, bit in enumerate(result.bits):
        bit_id = conn.execute(
            "INSERT INTO bit (work_id, ordinal, title, summary) VALUES (?, ?, ?, ?)",
            (work_id, bit_ordinal, bit.title.strip(), bit.summary.strip()),
        ).lastrowid
        for ordinal, span in enumerate(bit.passages):
            first, last = doc.units[span.unit_start], doc.units[span.unit_end]
            text = doc.text[first.char_start:last.char_end]
            following = doc.units[span.unit_end + 1] if span.unit_end + 1 < len(doc.units) else None
            conn.execute(
                "INSERT INTO passage (bit_id, ordinal, text, content_hash, unit_start, unit_end, char_start, "
                "char_end, start_ts, end_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (bit_id, ordinal, text, hashlib.sha256(text.encode()).hexdigest(), span.unit_start, span.unit_end,
                 first.char_start, last.char_end, first.ts, following.ts if following else None),
            )
            count += 1
    return count
