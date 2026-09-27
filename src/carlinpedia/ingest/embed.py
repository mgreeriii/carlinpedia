"""Embedders and the per-work index refresh (FTS rows plus text and thesis vectors)."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from typing import Protocol

import sqlite_vec

from carlinpedia.db.schema import delete_index_rows, drop_vector_tables, ensure_vector_tables, get_meta, set_meta


class EmbeddingMismatch(Exception):
    """The database was indexed with a different embedding model than the one in use."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Local embeddings. BGE models want an instruction prefix on queries but not on documents."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, dim: int):
        from sentence_transformers import SentenceTransformer  # heavy import, deferred

        self.name = model_name
        self.dim = dim
        self._model = SentenceTransformer(model_name)
        actual = self._model.get_embedding_dimension()
        if actual != dim:
            raise EmbeddingMismatch(
                f"{model_name} produces {actual}-dimensional vectors, but carlinpedia.toml says dim = {dim}. "
                f"Set [embedding] dim = {actual}."
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([self.QUERY_PREFIX + text], normalize_embeddings=True)[0].tolist()


class HashEmbedder:
    """Deterministic bag-of-words vectors for tests and offline development. Shared words mean similarity."""

    def __init__(self, dim: int = 64):
        self.name = "hash"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9']+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def embedder_key(embedder: Embedder) -> str:
    return f"{embedder.name}:{embedder.dim}"


def prepare_embedding_space(conn: sqlite3.Connection, embedder: Embedder) -> bool:
    """Make the vector tables match the embedder. Returns True if existing vectors were discarded."""
    key = embedder_key(embedder)
    current = get_meta(conn, "embedding_model")
    if current == key:
        return False
    drop_vector_tables(conn)
    ensure_vector_tables(conn, embedder.dim)
    set_meta(conn, "embedding_model", key)
    conn.commit()
    return current is not None


def check_embedding_space(conn: sqlite3.Connection, embedder: Embedder) -> None:
    current = get_meta(conn, "embedding_model")
    if current is not None and current != embedder_key(embedder):
        raise EmbeddingMismatch(
            f"The index was built with {current} but the configured embedder is {embedder_key(embedder)}. "
            "Run `carlin ingest --from embed` to re-index."
        )


def index_work(conn: sqlite3.Connection, embedder: Embedder, work_id: int) -> int:
    """Rewrite FTS rows and both vectors for every passage in the work. The caller owns the transaction."""
    rows = conn.execute(
        "SELECT p.id, p.text, p.thesis, b.title FROM passage p JOIN bit b ON b.id = p.bit_id "
        "WHERE b.work_id = ? ORDER BY p.id",
        (work_id,),
    ).fetchall()
    ids = [r["id"] for r in rows]
    delete_index_rows(conn, ids)
    conn.executemany(
        "INSERT INTO passage_fts (rowid, text, thesis, bit_title) VALUES (?, ?, ?, ?)",
        [(r["id"], r["text"], r["thesis"] or "", r["title"]) for r in rows],
    )
    if rows:
        text_vectors = embedder.embed_documents([r["text"] for r in rows])
        conn.executemany(
            "INSERT INTO vec_passage_text (passage_id, embedding) VALUES (?, ?)",
            [(pid, sqlite_vec.serialize_float32(v)) for pid, v in zip(ids, text_vectors)],
        )
    with_thesis = [r for r in rows if r["thesis"]]
    if with_thesis:
        thesis_vectors = embedder.embed_documents([r["thesis"] for r in with_thesis])
        conn.executemany(
            "INSERT INTO vec_passage_thesis (passage_id, embedding) VALUES (?, ?)",
            [(r["id"], sqlite_vec.serialize_float32(v)) for r, v in zip(with_thesis, thesis_vectors)],
        )
    return len(rows)
