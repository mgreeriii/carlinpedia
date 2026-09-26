from pathlib import Path

import pytest

from carlinpedia.db.schema import connect
from carlinpedia.ingest.manifest import (
    ManifestError, discover_slugs, load_manifest, read_transcript, register_work,
)

MANIFEST = """\
title: "Test Special"
kind: special
recorded_on: 1990-02-17
source:
  file: transcript.txt
  provenance: "Written for tests"
"""


def make_work(corpus: Path, slug: str, manifest: str = MANIFEST, transcript: str = "Hello there.") -> Path:
    work_dir = corpus / slug
    work_dir.mkdir(parents=True)
    (work_dir / "manifest.yaml").write_text(manifest)
    (work_dir / "transcript.txt").write_text(transcript)
    return work_dir


@pytest.fixture
def conn():
    return connect(":memory:", embedding_dim=8)


def test_discover_slugs_only_lists_dirs_with_manifests(tmp_path):
    make_work(tmp_path, "b-work")
    make_work(tmp_path, "a-work")
    (tmp_path / "no-manifest").mkdir()
    assert discover_slugs(tmp_path) == ["a-work", "b-work"]
    assert discover_slugs(tmp_path / "missing") == []


def test_load_manifest_rejects_unknown_keys_and_bad_kind(tmp_path):
    make_work(tmp_path, "w", manifest=MANIFEST.replace("kind: special", "kind: podcast"))
    with pytest.raises(ManifestError, match="kind"):
        load_manifest(tmp_path / "w")
    make_work(tmp_path, "x", manifest=MANIFEST + "colour: red\n")
    with pytest.raises(ManifestError, match="colour"):
        load_manifest(tmp_path / "x")


def test_register_is_unchanged_on_second_run(conn, tmp_path):
    make_work(tmp_path, "w")
    first = register_work(conn, tmp_path, "w")
    second = register_work(conn, tmp_path, "w")
    assert first.changed is True
    assert second.changed is False
    assert second.source_id == first.source_id
    assert conn.execute("SELECT title, recorded_on FROM work").fetchone()[:] == ("Test Special", "1990-02-17")


def test_register_new_transcript_deactivates_old_source(conn, tmp_path):
    work_dir = make_work(tmp_path, "w")
    first = register_work(conn, tmp_path, "w")
    (work_dir / "transcript.txt").write_text("A better transcript.")
    second = register_work(conn, tmp_path, "w")
    assert second.changed is True
    rows = conn.execute("SELECT id, is_active FROM source_document ORDER BY id").fetchall()
    assert [tuple(r) for r in rows] == [(first.source_id, 0), (second.source_id, 1)]


def test_register_missing_transcript_is_a_clear_error(conn, tmp_path):
    work_dir = make_work(tmp_path, "w")
    (work_dir / "transcript.txt").unlink()
    with pytest.raises(ManifestError, match="does not exist"):
        register_work(conn, tmp_path, "w")


def test_register_rejects_paths_outside_the_work_dir(conn, tmp_path):
    make_work(tmp_path, "w", manifest=MANIFEST.replace("transcript.txt", "../secret.txt"))
    (tmp_path / "secret.txt").write_text("nope")
    with pytest.raises(ManifestError, match="inside"):
        register_work(conn, tmp_path, "w")


def test_read_transcript_falls_back_to_cp1252(tmp_path):
    path = tmp_path / "t.txt"
    path.write_bytes("It\x92s \x93soft\x94 language.".encode("latin-1"))
    assert read_transcript(path) == "It\u2019s \u201csoft\u201d language."
    path.write_bytes("\ufeffUTF-8 with BOM".encode("utf-8"))
    assert read_transcript(path) == "UTF-8 with BOM"


def test_read_transcript_decodes_utf16_and_rejects_binary(tmp_path):
    path = tmp_path / "t.txt"
    path.write_bytes("It\u2019s soft language.".encode("utf-16"))  # Notepad "Unicode", with BOM
    assert read_transcript(path) == "It\u2019s soft language."
    path.write_bytes(b"\xfe\xff" + "Big endian.".encode("utf-16-be"))
    assert read_transcript(path) == "Big endian."
    path.write_bytes(b"\x00\x01binary\x00junk")
    with pytest.raises(ManifestError, match="not a text transcript"):
        read_transcript(path)
