"""Read-side lookups shared by search, the CLI, and the MCP tools. Everything returns plain dicts."""

from __future__ import annotations

import sqlite3


class NotFound(LookupError):
    """No row with that id."""


_PASSAGE_SQL = (
    "SELECT p.*, b.id AS bit_id, b.title AS bit_title, b.summary AS bit_summary, "
    "w.slug AS work_slug, w.title AS work_title, w.kind AS work_kind, "
    "CAST(substr(COALESCE(w.recorded_on, w.released_on), 1, 4) AS INTEGER) AS work_year "
    "FROM passage p JOIN bit b ON b.id = p.bit_id JOIN work w ON w.id = b.work_id WHERE p.id = ?"
)


def _approved_terms(conn: sqlite3.Connection, passage_id: int) -> tuple[list[str], list[str]]:
    themes = [r["name"] for r in conn.execute(
        "SELECT t.name FROM passage_theme pt JOIN theme t ON t.id = pt.theme_id "
        "WHERE pt.passage_id = ? AND t.status = 'approved' ORDER BY pt.weight DESC, t.name", (passage_id,))]
    targets = [r["name"] for r in conn.execute(
        "SELECT t.name FROM passage_target pt JOIN target t ON t.id = pt.target_id "
        "WHERE pt.passage_id = ? AND t.status = 'approved' ORDER BY t.name", (passage_id,))]
    return themes, targets


def passage_dict(conn: sqlite3.Connection, passage_id: int) -> dict:
    row = conn.execute(_PASSAGE_SQL, (passage_id,)).fetchone()
    if row is None:
        raise NotFound(f"No passage with id {passage_id}")
    themes, targets = _approved_terms(conn, passage_id)
    return {
        "passage_id": row["id"],
        "text": row["text"],
        "thesis": row["thesis"],
        "work": {"slug": row["work_slug"], "title": row["work_title"], "year": row["work_year"], "kind": row["work_kind"]},
        "bit": {"id": row["bit_id"], "title": row["bit_title"], "ordinal_in_bit": row["ordinal"]},
        "locator": {"start_ts": row["start_ts"], "end_ts": row["end_ts"], "chapter": row["chapter"], "page": row["page"]},
        "themes": themes,
        "targets": targets,
        "tone": row["tone"],
        "profanity": bool(row["profanity"]) if row["profanity"] is not None else None,
        "favorite": bool(row["favorite"]),
    }


def get_passage(conn: sqlite3.Connection, passage_id: int, context: int = 1) -> dict:
    """The passage, up to `context` neighbors on each side within its bit, and the bit summary."""
    result = passage_dict(conn, passage_id)
    context = max(0, min(context, 5))
    ordinal = result["bit"]["ordinal_in_bit"]
    neighbors = conn.execute(
        "SELECT id, ordinal, text FROM passage WHERE bit_id = ? AND ordinal BETWEEN ? AND ? AND id != ? ORDER BY ordinal",
        (result["bit"]["id"], ordinal - context, ordinal + context, passage_id),
    ).fetchall()
    result["before"] = [{"passage_id": n["id"], "text": n["text"]} for n in neighbors if n["ordinal"] < ordinal]
    result["after"] = [{"passage_id": n["id"], "text": n["text"]} for n in neighbors if n["ordinal"] > ordinal]
    result["bit"]["summary"] = conn.execute("SELECT summary FROM bit WHERE id = ?", (result["bit"]["id"],)).fetchone()[0]
    return result


def get_bit(conn: sqlite3.Connection, bit_id: int) -> dict:
    row = conn.execute(
        "SELECT b.id, b.title, b.summary, w.title AS work_title, "
        "CAST(substr(COALESCE(w.recorded_on, w.released_on), 1, 4) AS INTEGER) AS work_year "
        "FROM bit b JOIN work w ON w.id = b.work_id WHERE b.id = ?", (bit_id,)
    ).fetchone()
    if row is None:
        raise NotFound(f"No bit with id {bit_id}")
    passages = [
        {"passage_id": p["id"], "text": p["text"], "thesis": p["thesis"], "favorite": bool(p["favorite"])}
        for p in conn.execute("SELECT id, text, thesis, favorite FROM passage WHERE bit_id = ? ORDER BY ordinal", (bit_id,))
    ]
    return {"bit_id": row["id"], "title": row["title"], "summary": row["summary"],
            "work": {"title": row["work_title"], "year": row["work_year"]}, "passages": passages}


def list_works(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT w.slug, w.title, w.kind, CAST(substr(COALESCE(w.recorded_on, w.released_on), 1, 4) AS INTEGER) AS year, "
        "(SELECT count(*) FROM bit b WHERE b.work_id = w.id) AS bits, "
        "(SELECT count(*) FROM passage p JOIN bit b ON b.id = p.bit_id WHERE b.work_id = w.id) AS passages "
        "FROM work w ORDER BY year, w.title"
    )]


def list_themes(conn: sqlite3.Connection) -> dict:
    themes = [dict(r) for r in conn.execute(
        "SELECT t.slug, t.name, t.description, p.slug AS parent FROM theme t LEFT JOIN theme p ON p.id = t.parent_id "
        "WHERE t.status = 'approved' ORDER BY COALESCE(p.slug, t.slug), t.parent_id IS NOT NULL, t.slug"
    )]
    targets = [dict(r) for r in conn.execute(
        "SELECT slug, name, description FROM target WHERE status = 'approved' ORDER BY slug"
    )]
    return {"themes": themes, "targets": targets}


def set_favorite(conn: sqlite3.Connection, passage_id: int, favorite: bool) -> dict:
    cur = conn.execute("UPDATE passage SET favorite = ? WHERE id = ?", (int(favorite), passage_id))
    if cur.rowcount == 0:
        raise NotFound(f"No passage with id {passage_id}")
    conn.commit()
    return {"passage_id": passage_id, "favorite": favorite}
