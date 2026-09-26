"""Per-work manifest.yaml loading and Work/SourceDocument registration."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

MANIFEST_FILE = "manifest.yaml"


class ManifestError(Exception):
    """A manifest or its transcript is missing or invalid."""


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    provenance: str
    quality: Literal["verified", "unverified", "auto_transcribed"] = "unverified"
    has_timestamps: bool = False
    strip_patterns: list[str] = []


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    kind: Literal["special", "album", "book", "interview"]
    recorded_on: date | None = None
    released_on: date | None = None
    venue: str | None = None
    notes: str | None = None
    source: SourceSpec


@dataclass(frozen=True)
class Registration:
    work_id: int
    source_id: int
    source_path: Path
    manifest: Manifest
    checksum: str
    changed: bool


def discover_slugs(corpus_dir: Path) -> list[str]:
    if not corpus_dir.is_dir():
        return []
    return sorted(p.name for p in corpus_dir.iterdir() if (p / MANIFEST_FILE).is_file())


def load_manifest(work_dir: Path) -> Manifest:
    path = work_dir / MANIFEST_FILE
    if not path.is_file():
        raise ManifestError(f"No {MANIFEST_FILE} in {work_dir}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return Manifest.model_validate(data)
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc
    except ValidationError as exc:
        raise ManifestError(f"{path} is invalid:\n{exc}") from exc


def read_transcript(path: Path) -> str:
    """Decode a transcript. Web transcripts are often Windows-1252, so fall back to it."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def register_work(conn: sqlite3.Connection, corpus_dir: Path, slug: str) -> Registration:
    """Upsert the Work and its active SourceDocument. `changed` is True when the transcript is new or different."""
    work_dir = corpus_dir / slug
    manifest = load_manifest(work_dir)
    source_path = (work_dir / manifest.source.file).resolve()
    if not source_path.is_relative_to(work_dir.resolve()):
        raise ManifestError(f"{slug}: source.file must stay inside {work_dir}")
    if not source_path.is_file():
        raise ManifestError(f"{slug}: transcript {source_path} does not exist")
    checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()

    conn.execute(
        "INSERT INTO work (slug, title, kind, recorded_on, released_on, venue, notes) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(slug) DO UPDATE SET title = excluded.title, kind = excluded.kind, "
        "recorded_on = excluded.recorded_on, released_on = excluded.released_on, venue = excluded.venue, "
        "notes = excluded.notes",
        (slug, manifest.title, manifest.kind,
         manifest.recorded_on.isoformat() if manifest.recorded_on else None,
         manifest.released_on.isoformat() if manifest.released_on else None,
         manifest.venue, manifest.notes),
    )
    work_id = conn.execute("SELECT id FROM work WHERE slug = ?", (slug,)).fetchone()["id"]
    rel_path = str(source_path.relative_to(corpus_dir.resolve()))

    active = conn.execute(
        "SELECT id, checksum, path FROM source_document WHERE work_id = ? AND is_active = 1", (work_id,)
    ).fetchone()
    if active is not None and active["checksum"] == checksum and active["path"] == rel_path:
        conn.execute(
            "UPDATE source_document SET provenance = ?, quality = ?, has_timestamps = ? WHERE id = ?",
            (manifest.source.provenance, manifest.source.quality, int(manifest.source.has_timestamps), active["id"]),
        )
        conn.commit()
        return Registration(work_id, active["id"], source_path, manifest, checksum, changed=False)

    conn.execute("UPDATE source_document SET is_active = 0 WHERE work_id = ?", (work_id,))
    cur = conn.execute(
        "INSERT INTO source_document (work_id, path, provenance, quality, has_timestamps, checksum) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (work_id, rel_path, manifest.source.provenance, manifest.source.quality,
         int(manifest.source.has_timestamps), checksum),
    )
    conn.commit()
    return Registration(work_id, cur.lastrowid, source_path, manifest, checksum, changed=True)
