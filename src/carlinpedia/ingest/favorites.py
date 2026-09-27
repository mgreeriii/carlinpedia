"""Keep user favorites across re-segmentation, and export them."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class FavoriteSnapshot:
    content_hash: str
    char_start: int
    char_end: int
    text: str


def snapshot_favorites(conn: sqlite3.Connection, work_id: int) -> list[FavoriteSnapshot]:
    return [
        FavoriteSnapshot(r["content_hash"], r["char_start"], r["char_end"], r["text"])
        for r in conn.execute(
            "SELECT p.content_hash, p.char_start, p.char_end, p.text FROM passage p JOIN bit b ON b.id = p.bit_id "
            "WHERE b.work_id = ? AND p.favorite = 1 ORDER BY p.char_start",
            (work_id,),
        )
    ]


def restore_favorites(conn: sqlite3.Connection, work_id: int, snapshots: list[FavoriteSnapshot]) -> list[FavoriteSnapshot]:
    """Re-mark favorites by identical text, then by at least 50% character overlap. Returns and records orphans."""
    passages = conn.execute(
        "SELECT p.id, p.content_hash, p.char_start, p.char_end FROM passage p JOIN bit b ON b.id = p.bit_id "
        "WHERE b.work_id = ?",
        (work_id,),
    ).fetchall()
    unmatched = []
    for snap in snapshots:
        match = next((p["id"] for p in passages if p["content_hash"] == snap.content_hash), None)
        if match is None:
            best_id, best_overlap = None, 0
            for p in passages:
                overlap = min(p["char_end"], snap.char_end) - max(p["char_start"], snap.char_start)
                if overlap > best_overlap:
                    best_id, best_overlap = p["id"], overlap
            if best_id is not None and best_overlap * 2 >= snap.char_end - snap.char_start:
                match = best_id
        if match is None:
            unmatched.append(snap)
            conn.execute(
                "INSERT INTO orphaned_favorite (work_id, content_hash, text) VALUES (?, ?, ?)",
                (work_id, snap.content_hash, snap.text),
            )
        else:
            conn.execute("UPDATE passage SET favorite = 1 WHERE id = ?", (match,))
    return unmatched


def list_orphaned(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT w.slug AS work, o.text, o.created_at FROM orphaned_favorite o JOIN work w ON w.id = o.work_id "
        "ORDER BY o.created_at, o.id"
    )]


def export_favorites(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT w.slug AS work, b.title AS bit, p.content_hash, p.text FROM passage p "
        "JOIN bit b ON b.id = p.bit_id JOIN work w ON w.id = b.work_id WHERE p.favorite = 1 "
        "ORDER BY w.slug, b.ordinal, p.ordinal"
    )]
