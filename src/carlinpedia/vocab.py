"""Theme and target vocabularies: YAML seeds, approval status, and lookups."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

TABLES = ("theme", "target")


class VocabError(Exception):
    """A vocabulary file is malformed or a term is unknown."""


@dataclass(frozen=True)
class Term:
    id: int
    slug: str
    name: str
    description: str
    parent_slug: str | None = None


@dataclass(frozen=True)
class Vocab:
    themes: dict[str, Term]
    targets: dict[str, Term]

    def fingerprint(self) -> str:
        """A stable string that changes whenever the approved vocabulary changes."""
        return "|".join(sorted(self.themes)) + "#" + "|".join(sorted(self.targets))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "term"


def _read_yaml_list(path: Path) -> list[dict]:
    if not path.is_file():
        raise VocabError(f"Missing vocabulary file {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(data, list):
        raise VocabError(f"{path} must contain a YAML list")
    return data


def _upsert_approved(conn, table: str, entry: dict, parent_id: int | None, path: Path) -> int:
    try:
        slug, name = entry["slug"], entry["name"]
    except (KeyError, TypeError) as exc:
        raise VocabError(f"{path}: every entry needs a slug and a name, got {entry!r}") from exc
    description = entry.get("description", "")
    if table == "theme":
        conn.execute(
            "INSERT INTO theme (slug, name, parent_id, description, status) VALUES (?, ?, ?, ?, 'approved') "
            "ON CONFLICT(slug) DO UPDATE SET name = excluded.name, parent_id = excluded.parent_id, "
            "description = excluded.description, status = 'approved'",
            (slug, name, parent_id, description),
        )
    else:
        conn.execute(
            "INSERT INTO target (slug, name, description, status) VALUES (?, ?, ?, 'approved') "
            "ON CONFLICT(slug) DO UPDATE SET name = excluded.name, description = excluded.description, "
            "status = 'approved'",
            (slug, name, description),
        )
    return conn.execute(f"SELECT id FROM {table} WHERE slug = ?", (slug,)).fetchone()["id"]


def sync_vocab(conn: sqlite3.Connection, vocab_dir: Path) -> int:
    """Upsert every YAML term as approved. Terms missing from YAML are left alone. Returns the term count."""
    count = 0
    themes_path = vocab_dir / "themes.yaml"
    for top in _read_yaml_list(themes_path):
        parent_id = _upsert_approved(conn, "theme", top, None, themes_path)
        count += 1
        for child in top.get("children", []) or []:
            _upsert_approved(conn, "theme", child, parent_id, themes_path)
            count += 1
    targets_path = vocab_dir / "targets.yaml"
    for entry in _read_yaml_list(targets_path):
        _upsert_approved(conn, "target", entry, None, targets_path)
        count += 1
    conn.commit()
    return count


def load_vocab(conn: sqlite3.Connection) -> Vocab:
    """Approved terms only. Proposed and rejected terms never reach prompts or filters."""
    themes = {
        r["slug"]: Term(r["id"], r["slug"], r["name"], r["description"], r["parent_slug"])
        for r in conn.execute(
            "SELECT t.id, t.slug, t.name, t.description, p.slug AS parent_slug FROM theme t "
            "LEFT JOIN theme p ON p.id = t.parent_id WHERE t.status = 'approved' ORDER BY t.slug"
        )
    }
    targets = {
        r["slug"]: Term(r["id"], r["slug"], r["name"], r["description"])
        for r in conn.execute("SELECT id, slug, name, description FROM target WHERE status = 'approved' ORDER BY slug")
    }
    return Vocab(themes=themes, targets=targets)


def upsert_proposed(conn: sqlite3.Connection, table: str, name: str, description: str) -> int | None:
    """Record a term the model proposed. Returns its id, or None if the user already rejected it."""
    if table not in TABLES:
        raise ValueError(f"unknown vocabulary table {table!r}")
    slug = f"proposed.{slugify(name)}"
    row = conn.execute(f"SELECT id, status FROM {table} WHERE slug = ?", (slug,)).fetchone()
    if row is not None:
        return None if row["status"] == "rejected" else row["id"]
    cur = conn.execute(
        f"INSERT INTO {table} (slug, name, description, status) VALUES (?, ?, ?, 'proposed')",
        (slug, name.strip(), description.strip()),
    )
    return cur.lastrowid


def set_status(conn: sqlite3.Connection, slug: str, status: str) -> str:
    """Approve or reject a term by slug in whichever table holds it. Returns the table name."""
    if status not in ("approved", "rejected"):
        raise ValueError("status must be 'approved' or 'rejected'")
    for table in TABLES:
        cur = conn.execute(f"UPDATE {table} SET status = ? WHERE slug = ?", (status, slug))
        if cur.rowcount:
            conn.commit()
            return table
    raise VocabError(f"No theme or target with slug {slug!r}")


def list_proposed(conn: sqlite3.Connection) -> list[dict]:
    rows = []
    for table in TABLES:
        for r in conn.execute(
            f"SELECT t.slug, t.name, t.description, "
            f"(SELECT count(*) FROM passage_{table} pt WHERE pt.{table}_id = t.id) AS uses "
            f"FROM {table} t WHERE t.status = 'proposed' ORDER BY uses DESC, t.slug"
        ):
            rows.append({"kind": table, **dict(r)})
    return rows


def resolve_terms(vocab_terms: dict[str, Term], requested: list[str], kind: str) -> list[Term]:
    """Match slugs or display names (case-insensitive) to approved terms, failing loudly on unknowns."""
    by_key = {}
    for term in vocab_terms.values():
        by_key[term.slug.lower()] = term
        by_key[term.name.lower()] = term
    resolved, unknown = [], []
    for item in requested:
        term = by_key.get(item.strip().lower())
        (resolved if term else unknown).append(term or item)
    if unknown:
        raise VocabError(
            f"Unknown {kind}(s): {', '.join(repr(u) for u in unknown)}. Call list_themes to see valid values."
        )
    return resolved


def descendant_theme_ids(conn: sqlite3.Connection, theme_ids: list[int]) -> list[int]:
    """The given themes plus every approved descendant."""
    if not theme_ids:
        return []
    marks = ",".join("?" * len(theme_ids))
    rows = conn.execute(
        f"WITH RECURSIVE tree(id) AS (SELECT id FROM theme WHERE id IN ({marks}) "
        f"UNION SELECT t.id FROM theme t JOIN tree ON t.parent_id = tree.id WHERE t.status = 'approved') "
        f"SELECT id FROM tree",
        theme_ids,
    )
    return sorted(r["id"] for r in rows)
