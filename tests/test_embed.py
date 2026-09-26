import pytest

from carlinpedia.db.schema import connect
from carlinpedia.ingest.embed import (
    EmbeddingMismatch, HashEmbedder, check_embedding_space, index_work, prepare_embedding_space,
)


def test_hash_embedder_is_normalized_and_deterministic():
    e = HashEmbedder(dim=16)
    a, b = e.embed_documents(["soft language", "soft language"])
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert e.embed_query("soft language") == a


def test_prepare_embedding_space_recreates_tables_on_model_change():
    conn = connect(":memory:", embedding_dim=768)
    assert prepare_embedding_space(conn, HashEmbedder(dim=16)) is False  # first use: nothing discarded
    assert prepare_embedding_space(conn, HashEmbedder(dim=16)) is False
    assert prepare_embedding_space(conn, HashEmbedder(dim=32)) is True


def test_check_embedding_space_rejects_a_different_model():
    conn = connect(":memory:", embedding_dim=16)
    check_embedding_space(conn, HashEmbedder(dim=16))  # nothing indexed yet: fine
    prepare_embedding_space(conn, HashEmbedder(dim=16))
    with pytest.raises(EmbeddingMismatch, match="carlin ingest --from embed"):
        check_embedding_space(conn, HashEmbedder(dim=32))


def test_index_work_writes_fts_and_vectors_and_is_repeatable():
    conn = connect(":memory:", embedding_dim=16)
    embedder = HashEmbedder(dim=16)
    prepare_embedding_space(conn, embedder)
    work_id = conn.execute("INSERT INTO work (slug, title, kind) VALUES ('w', 'W', 'special')").lastrowid
    bit_id = conn.execute("INSERT INTO bit (work_id, ordinal, title, summary) VALUES (?, 0, 'Euphemisms', 's')",
                          (work_id,)).lastrowid
    conn.execute("INSERT INTO passage (bit_id, ordinal, text, content_hash, unit_start, unit_end, char_start, "
                 "char_end, thesis) VALUES (?, 0, 'bathroom tissue', 'h0', 0, 0, 0, 1, 'soft words hide truth')", (bit_id,))
    conn.execute("INSERT INTO passage (bit_id, ordinal, text, content_hash, unit_start, unit_end, char_start, "
                 "char_end) VALUES (?, 1, 'pre-owned cars', 'h1', 1, 1, 1, 2)", (bit_id,))
    for _ in range(2):
        assert index_work(conn, embedder, work_id) == 2
    assert conn.execute("SELECT count(*) FROM passage_fts").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM vec_passage_text").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM vec_passage_thesis").fetchone()[0] == 1
    hit = conn.execute("SELECT rowid FROM passage_fts WHERE passage_fts MATCH 'euphemisms'").fetchall()
    assert len(hit) == 2  # bit title is indexed on every passage
