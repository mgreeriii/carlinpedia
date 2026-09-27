import pytest

from carlinpedia.db.schema import connect
from carlinpedia.ingest.normalize import normalize
from carlinpedia.ingest.segment import (
    SegmentationFailed, SegmentationResult, render_units, segment_document, store_segmentation,
    validate_segmentation,
)
from carlinpedia.llm.client import FakeLLM

DOC = normalize("One. Two. [laughter] Three. Four. Five. Six.")  # 6 units


def seg(*bits):
    """bits: (title, start, end, [(p_start, p_end), ...])"""
    return SegmentationResult.model_validate({"bits": [
        {"title": t, "summary": f"{t} summary", "unit_start": s, "unit_end": e,
         "passages": [{"unit_start": a, "unit_end": b} for a, b in ps]}
        for t, s, e, ps in bits
    ]})


GOOD = seg(("A", 0, 2, [(0, 1), (2, 2)]), ("B", 3, 5, [(3, 5)]))


def test_valid_segmentation_has_no_errors():
    assert validate_segmentation(GOOD, 6) == []


@pytest.mark.parametrize("result,fragment", [
    (seg(("A", 0, 2, [(0, 2)]), ("B", 4, 5, [(4, 5)])), "starts at unit 4, expected 3"),
    (seg(("A", 0, 3, [(0, 3)]), ("B", 3, 5, [(3, 5)])), "starts at unit 3, expected 4"),
    (seg(("A", 0, 5, [(0, 1), (3, 5)])), "passage 1 starts at unit 3, expected 2"),
    (seg(("A", 0, 5, [(0, 4)])), "passages end at unit 4, but the bit ends at 5"),
    (seg(("A", 0, 4, [(0, 4)])), "transcript has units 0..5"),
    (seg(("A", 0, 7, [(0, 7)])), "bits cover units 0..7"),
    (seg(("A", 0, 5, [])), "has no passages"),
    (seg(("A", 0, 5, [(0, 5)])), None),
    (SegmentationResult(bits=[]), "no bits"),
])
def test_validator_reports_problems(result, fragment):
    errors = validate_segmentation(result, 6, max_passage_units=40)
    if fragment is None:
        assert errors == []
    else:
        assert any(fragment in e for e in errors), errors


def test_validator_enforces_max_passage_length():
    errors = validate_segmentation(seg(("A", 0, 5, [(0, 5)])), 6, max_passage_units=3)
    assert any("spans 6 units; the maximum is 3" in e for e in errors)


def test_render_units_numbers_lines_and_marks_cues():
    assert render_units(DOC).splitlines()[:3] == ["[0] One.", "[1] Two. ⟨laugh⟩", "[2] Three."]


def test_segment_document_retries_once_with_errors():
    bad = GOOD.model_dump()
    bad["bits"][1]["unit_start"] = 4
    fake = FakeLLM({"segment_v1:w": [bad, GOOD.model_dump()]})
    result = segment_document(fake, DOC, work_title="W", key="w")
    assert result == GOOD
    assert len(fake.calls) == 2
    assert "starts at unit 4, expected 3" in fake.calls[1]["user"]
    assert "Units: 6 (numbered 0 to 5)" in fake.calls[0]["user"]


def test_segment_document_gives_up_after_second_failure():
    bad = GOOD.model_dump()
    bad["bits"][1]["unit_end"] = 9
    fake = FakeLLM({"segment_v1:w": [bad]})
    with pytest.raises(SegmentationFailed) as excinfo:
        segment_document(fake, DOC, work_title="W", key="w")
    assert any("0..9" in e for e in excinfo.value.errors)


def test_segment_document_rejects_empty_transcript():
    with pytest.raises(SegmentationFailed, match="no text"):
        segment_document(FakeLLM({}), normalize("  "), work_title="W", key="w")


def test_store_segmentation_writes_verbatim_passages_and_replaces_old_ones():
    conn = connect(":memory:", embedding_dim=8)
    work_id = conn.execute("INSERT INTO work (slug, title, kind) VALUES ('w', 'W', 'special')").lastrowid
    assert store_segmentation(conn, work_id, DOC, GOOD) == 3
    rows = conn.execute("SELECT b.title, p.ordinal, p.text FROM passage p JOIN bit b ON b.id = p.bit_id "
                        "ORDER BY b.ordinal, p.ordinal").fetchall()
    assert [tuple(r) for r in rows] == [("A", 0, "One. Two."), ("A", 1, "Three."), ("B", 0, "Four. Five. Six.")]
    store_segmentation(conn, work_id, DOC, seg(("Only", 0, 5, [(0, 5)])))
    assert conn.execute("SELECT count(*) FROM passage").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM bit").fetchone()[0] == 1


def test_store_segmentation_records_timestamps():
    conn = connect(":memory:", embedding_dim=8)
    work_id = conn.execute("INSERT INTO work (slug, title, kind) VALUES ('w', 'W', 'special')").lastrowid
    doc = normalize("[0:05] One.\n[0:09] Two.", has_timestamps=True)
    store_segmentation(conn, work_id, doc, seg(("A", 0, 1, [(0, 0), (1, 1)])))
    rows = conn.execute("SELECT start_ts, end_ts FROM passage ORDER BY ordinal").fetchall()
    assert [tuple(r) for r in rows] == [(5.0, 9.0), (9.0, None)]
