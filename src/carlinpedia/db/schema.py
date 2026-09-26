"""SQLite connection, migrations, and the sqlite-vec vector tables."""

from __future__ import annotations

import re
import sqlite3
from importlib import resources
from pathlib import Path

import sqlite_vec

VECTOR_TABLES = ("vec_passage_text", "vec_passage_thesis")
_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def connect(db_path: Path | str, embedding_dim: int = 768) -> sqlite3.Connection:
    """Open the database, load sqlite-vec, apply migrations, and make sure vector tables exist."""
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # MCP runs sync tools on worker threads, so the connection must not be pinned to one thread.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    migrate(conn)
    ensure_vector_tables(conn, embedding_dim)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _migrations() -> list[tuple[int, str]]:
    found = []
    for entry in resources.files("carlinpedia.db").joinpath("migrations").iterdir():
        match = _MIGRATION_NAME.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.read_text(encoding="utf-8")))
    return sorted(found)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply migrations newer than meta.schema_version. Returns the resulting version."""
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    current = int(get_meta(conn, "schema_version") or 0)
    for version, sql in _migrations():
        if version <= current:
            continue
        conn.executescript(sql)
        set_meta(conn, "schema_version", str(version))
        conn.commit()
        current = version
    return current


def ensure_vector_tables(conn: sqlite3.Connection, dim: int) -> None:
    for table in VECTOR_TABLES:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"passage_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
        )
    conn.commit()


def drop_vector_tables(conn: sqlite3.Connection) -> None:
    for table in VECTOR_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def delete_index_rows(conn: sqlite3.Connection, passage_ids: list[int]) -> None:
    """Remove passages from FTS and vector tables, which foreign keys don't cascade into."""
    if not passage_ids:
        return
    marks = ",".join("?" * len(passage_ids))
    conn.execute(f"DELETE FROM passage_fts WHERE rowid IN ({marks})", passage_ids)
    for table in VECTOR_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE passage_id IN ({marks})", passage_ids)
