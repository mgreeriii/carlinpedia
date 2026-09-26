"""Fold near-identical passages from different works into one result."""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass, field


@dataclass
class Collapsed:
    passage_id: int
    score: float
    also_in: list[int] = field(default_factory=list)


def _vectors(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[float, ...]]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    out = {}
    for row in conn.execute(f"SELECT passage_id, embedding FROM vec_passage_text WHERE passage_id IN ({marks})", ids):
        blob = row["embedding"]
        out[row["passage_id"]] = struct.unpack(f"{len(blob) // 4}f", blob)
    return out


def _work_ids(conn: sqlite3.Connection, ids: list[int]) -> dict[int, int]:
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    return {
        r["id"]: r["work_id"]
        for r in conn.execute(f"SELECT p.id, b.work_id FROM passage p JOIN bit b ON b.id = p.bit_id WHERE p.id IN ({marks})", ids)
    }


def collapse_duplicates(conn: sqlite3.Connection, ranked: list[tuple[int, float]], threshold: float) -> list[Collapsed]:
    """Walk the ranking; a passage whose text vector is within `threshold` cosine of a kept passage
    from a different work is folded into that result's also_in list. Vectors are unit length, so cosine is a dot product."""
    ids = [pid for pid, _ in ranked]
    vectors, works = _vectors(conn, ids), _work_ids(conn, ids)
    kept: list[Collapsed] = []
    for pid, score in ranked:
        vec = vectors.get(pid)
        home = None
        if vec is not None:
            for candidate in kept:
                other = vectors.get(candidate.passage_id)
                if other is None or works.get(candidate.passage_id) == works.get(pid):
                    continue
                if sum(a * b for a, b in zip(vec, other)) >= threshold:
                    home = candidate
                    break
        if home is None:
            kept.append(Collapsed(pid, score))
        else:
            home.also_in.append(pid)
    return kept
