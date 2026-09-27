"""Hybrid search: FTS5 + text vectors + thesis vectors, fused, filtered, and de-duplicated."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import sqlite_vec

from carlinpedia.ingest.embed import Embedder, check_embedding_space
from carlinpedia.library import passage_dict
from carlinpedia.search.dedup import collapse_duplicates
from carlinpedia.search.fusion import reciprocal_rank_fusion
from carlinpedia.vocab import VocabError, descendant_theme_ids, load_vocab, resolve_terms

RETRIEVERS = ("fts", "text", "thesis")
MAX_LIMIT = 25
KINDS = ("special", "album", "book", "interview")


class SearchError(ValueError):
    """The query or its filters can't be run as given."""


@dataclass(frozen=True)
class SearchFilters:
    themes: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    year_from: int | None = None
    year_to: int | None = None
    kinds: tuple[str, ...] = ()
    favorites_only: bool = False

    def is_empty(self) -> bool:
        return self == SearchFilters()


def fts_query(text: str) -> str | None:
    """Turn free text into an FTS5 OR-query of quoted tokens, so user punctuation can't break the syntax."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens)) or None


def _filter_subquery(conn: sqlite3.Connection, filters: SearchFilters) -> tuple[str, list]:
    """SQL selecting the passage ids that satisfy the filters."""
    clauses, params = [], []
    vocab = load_vocab(conn)
    try:
        themes = resolve_terms(vocab.themes, list(filters.themes), "theme")
        targets = resolve_terms(vocab.targets, list(filters.targets), "target")
    except VocabError as exc:
        raise SearchError(str(exc)) from exc
    if themes:
        ids = descendant_theme_ids(conn, [t.id for t in themes])
        clauses.append(f"p.id IN (SELECT passage_id FROM passage_theme WHERE theme_id IN ({','.join('?' * len(ids))}))")
        params += ids
    if targets:
        clauses.append(f"p.id IN (SELECT passage_id FROM passage_target WHERE target_id IN ({','.join('?' * len(targets))}))")
        params += [t.id for t in targets]
    year = "CAST(substr(COALESCE(w.recorded_on, w.released_on), 1, 4) AS INTEGER)"
    if filters.year_from is not None:
        clauses.append(f"{year} >= ?")
        params.append(filters.year_from)
    if filters.year_to is not None:
        clauses.append(f"{year} <= ?")
        params.append(filters.year_to)
    if filters.kinds:
        bad = [k for k in filters.kinds if k not in KINDS]
        if bad:
            raise SearchError(f"Unknown kind(s) {bad}. Valid kinds: {', '.join(KINDS)}")
        clauses.append(f"w.kind IN ({','.join('?' * len(filters.kinds))})")
        params += list(filters.kinds)
    if filters.favorites_only:
        clauses.append("p.favorite = 1")
    where = " AND ".join(clauses) or "1"
    return (f"SELECT p.id FROM passage p JOIN bit b ON b.id = p.bit_id JOIN work w ON w.id = b.work_id WHERE {where}",
            params)


def _fts_ranking(conn, query: str, subquery: str, params: list, depth: int) -> list[int]:
    match = fts_query(query)
    if match is None:
        return []
    rows = conn.execute(
        f"SELECT rowid FROM passage_fts WHERE passage_fts MATCH ? AND rowid IN ({subquery}) "
        f"ORDER BY bm25(passage_fts, 1.0, 0.6, 0.4) LIMIT ?",
        [match, *params, depth],
    )
    return [r[0] for r in rows]


def _vector_ranking(conn, table: str, vector: list[float], subquery: str, params: list, depth: int, filtered: bool) -> list[int]:
    blob = sqlite_vec.serialize_float32(vector)
    if not filtered:
        rows = conn.execute(
            f"SELECT passage_id FROM {table} WHERE embedding MATCH ? AND k = ? ORDER BY distance", (blob, depth)
        )
    else:
        rows = conn.execute(
            f"SELECT passage_id FROM {table} WHERE passage_id IN ({subquery}) "
            f"ORDER BY vec_distance_cosine(embedding, ?) LIMIT ?",
            [*params, blob, depth],
        )
    return [r[0] for r in rows]


def search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    filters: SearchFilters = SearchFilters(),
    limit: int = 10,
    *,
    rrf_k: int = 60,
    depth: int = 50,
    dedup_threshold: float = 0.92,
    retrievers: tuple[str, ...] = RETRIEVERS,
) -> list[dict]:
    query = query.strip()
    if not query:
        raise SearchError("The query is empty.")
    unknown = [r for r in retrievers if r not in RETRIEVERS]
    if unknown:
        raise SearchError(f"Unknown retriever(s) {unknown}")
    limit = max(1, min(limit, MAX_LIMIT))
    check_embedding_space(conn, embedder)
    subquery, params = _filter_subquery(conn, filters)
    filtered = not filters.is_empty()

    rankings = []
    if "fts" in retrievers:
        rankings.append(_fts_ranking(conn, query, subquery, params, depth))
    if "text" in retrievers or "thesis" in retrievers:
        vector = embedder.embed_query(query)
        if "text" in retrievers:
            rankings.append(_vector_ranking(conn, "vec_passage_text", vector, subquery, params, depth, filtered))
        if "thesis" in retrievers:
            rankings.append(_vector_ranking(conn, "vec_passage_thesis", vector, subquery, params, depth, filtered))

    fused = reciprocal_rank_fusion(rankings, k=rrf_k)
    results = []
    for item in collapse_duplicates(conn, fused, dedup_threshold)[:limit]:
        result = passage_dict(conn, item.passage_id)
        result["score"] = round(item.score, 5)
        result["also_in"] = [
            {"passage_id": pid, "work": d["work"]["title"], "year": d["work"]["year"]}
            for pid in item.also_in
            for d in [passage_dict(conn, pid)]
        ]
        results.append(result)
    return results
