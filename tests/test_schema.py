import sqlite3

from carlinpedia.db.schema import connect, delete_index_rows, get_meta, migrate


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")}


def test_connect_creates_schema_and_vector_tables(tmp_path):
    conn = connect(tmp_path / "sub" / "carlin.db", embedding_dim=8)
    names = table_names(conn)
    for expected in ("work", "source_document", "bit", "passage", "theme", "target", "passage_theme",
                     "passage_target", "pipeline_step", "orphaned_favorite", "passage_fts",
                     "vec_passage_text", "vec_passage_thesis"):
        assert expected in names
    assert get_meta(conn, "schema_version") == "1"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "carlin.db", embedding_dim=8)
    assert migrate(conn) == 1
    conn.close()
    conn = connect(tmp_path / "carlin.db", embedding_dim=8)
    assert get_meta(conn, "schema_version") == "1"


def test_delete_index_rows_clears_fts_and_vectors():
    import sqlite_vec

    conn = connect(":memory:", embedding_dim=2)
    conn.execute("INSERT INTO passage_fts (rowid, text, thesis, bit_title) VALUES (7, 'a', 'b', 'c')")
    conn.execute("INSERT INTO vec_passage_text (passage_id, embedding) VALUES (7, ?)",
                 (sqlite_vec.serialize_float32([1.0, 0.0]),))
    delete_index_rows(conn, [7])
    assert conn.execute("SELECT count(*) FROM passage_fts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM vec_passage_text").fetchone()[0] == 0
