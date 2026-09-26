import sqlite_vec

from carlinpedia.db.schema import connect
from carlinpedia.search.dedup import collapse_duplicates
from carlinpedia.search.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_rankings():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 3, 1], [2, 1]], k=60)
    assert [pid for pid, _ in fused] == [2, 1, 3]
    assert abs(fused[0][1] - (1 / 62 + 1 / 61 + 1 / 61)) < 1e-12


def test_rrf_handles_empty_rankings_and_ties():
    assert reciprocal_rank_fusion([[], []]) == []
    assert [pid for pid, _ in reciprocal_rank_fusion([[5], [4]])] == [4, 5]


def make_db(passages):
    """passages: list of (passage_id, work_id, vector)."""
    conn = connect(":memory:", embedding_dim=2)
    for work_id in {w for _, w, _ in passages}:
        conn.execute("INSERT INTO work (id, slug, title, kind) VALUES (?, ?, ?, 'special')", (work_id, f"w{work_id}", f"W{work_id}"))
        conn.execute("INSERT INTO bit (id, work_id, ordinal, title, summary) VALUES (?, ?, 0, 'B', 's')", (work_id, work_id))
    for pid, work_id, vec in passages:
        conn.execute("INSERT INTO passage (id, bit_id, ordinal, text, content_hash, unit_start, unit_end, char_start, "
                     "char_end) VALUES (?, ?, ?, 't', 'h', 0, 0, 0, 1)", (pid, work_id, pid))
        conn.execute("INSERT INTO vec_passage_text (passage_id, embedding) VALUES (?, ?)",
                     (pid, sqlite_vec.serialize_float32(vec)))
    return conn


def test_duplicates_from_other_works_fold_into_the_higher_ranked_result():
    conn = make_db([(1, 1, [1.0, 0.0]), (2, 2, [1.0, 0.0]), (3, 2, [0.0, 1.0])])
    kept = collapse_duplicates(conn, [(1, 0.3), (2, 0.2), (3, 0.1)], threshold=0.92)
    assert [(k.passage_id, k.also_in) for k in kept] == [(1, [2]), (3, [])]


def test_similar_passages_in_the_same_work_are_kept_apart():
    conn = make_db([(1, 1, [1.0, 0.0]), (2, 1, [1.0, 0.0])])
    kept = collapse_duplicates(conn, [(1, 0.3), (2, 0.2)], threshold=0.92)
    assert [k.passage_id for k in kept] == [1, 2]


def test_passages_without_vectors_are_never_folded():
    conn = make_db([(1, 1, [1.0, 0.0])])
    kept = collapse_duplicates(conn, [(1, 0.3), (99, 0.2)], threshold=0.92)
    assert [k.passage_id for k in kept] == [1, 99]
