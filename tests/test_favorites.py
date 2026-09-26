import pytest

from carlinpedia.db.schema import connect
from carlinpedia.ingest.favorites import export_favorites, list_orphaned, restore_favorites, snapshot_favorites


@pytest.fixture
def db():
    conn = connect(":memory:", embedding_dim=8)
    work_id = conn.execute("INSERT INTO work (slug, title, kind) VALUES ('w', 'W', 'special')").lastrowid
    return conn, work_id


def add_passages(conn, work_id, spans, favorites=()):
    """spans: list of (char_start, char_end, text)."""
    conn.execute("DELETE FROM bit WHERE work_id = ?", (work_id,))
    bit_id = conn.execute("INSERT INTO bit (work_id, ordinal, title, summary) VALUES (?, 0, 'B', 's')",
                          (work_id,)).lastrowid
    for i, (start, end, text) in enumerate(spans):
        conn.execute("INSERT INTO passage (bit_id, ordinal, text, content_hash, unit_start, unit_end, char_start, "
                     "char_end, favorite) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (bit_id, i, text, f"hash-{text}", i, i, start, end, int(i in favorites)))


def favorite_texts(conn):
    return [r[0] for r in conn.execute("SELECT text FROM passage WHERE favorite = 1 ORDER BY char_start")]


def test_identical_text_keeps_its_favorite(db):
    conn, work_id = db
    add_passages(conn, work_id, [(0, 10, "alpha"), (10, 20, "beta")], favorites={1})
    snaps = snapshot_favorites(conn, work_id)
    add_passages(conn, work_id, [(0, 5, "al"), (5, 10, "pha"), (10, 20, "beta")])
    assert restore_favorites(conn, work_id, snaps) == []
    assert favorite_texts(conn) == ["beta"]


def test_mostly_overlapping_passage_inherits_the_favorite(db):
    conn, work_id = db
    add_passages(conn, work_id, [(0, 100, "old long passage")], favorites={0})
    snaps = snapshot_favorites(conn, work_id)
    add_passages(conn, work_id, [(0, 30, "head"), (30, 100, "most of it")])
    assert restore_favorites(conn, work_id, snaps) == []
    assert favorite_texts(conn) == ["most of it"]


def test_unmatched_favorite_is_recorded_not_dropped(db):
    conn, work_id = db
    add_passages(conn, work_id, [(0, 100, "old")], favorites={0})
    snaps = snapshot_favorites(conn, work_id)
    add_passages(conn, work_id, [(0, 40, "a"), (40, 70, "b"), (70, 100, "c")])
    unmatched = restore_favorites(conn, work_id, snaps)
    assert [u.text for u in unmatched] == ["old"]
    assert favorite_texts(conn) == []
    assert [o["text"] for o in list_orphaned(conn)] == ["old"]


def test_export_favorites(db):
    conn, work_id = db
    add_passages(conn, work_id, [(0, 10, "alpha"), (10, 20, "beta")], favorites={0})
    assert export_favorites(conn) == [{"work": "w", "bit": "B", "content_hash": "hash-alpha", "text": "alpha"}]
