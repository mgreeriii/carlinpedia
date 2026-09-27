"""Run register -> normalize -> segment -> enrich -> embed for each work, skipping steps whose inputs haven't changed."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field

from carlinpedia.config import Config
from carlinpedia.ingest.embed import Embedder, embedder_key, index_work, prepare_embedding_space
from carlinpedia.ingest.enrich import ENRICH_PROMPT, apply_enrichment, enrich_bit, load_bits
from carlinpedia.ingest.favorites import restore_favorites, snapshot_favorites
from carlinpedia.ingest.manifest import discover_slugs, read_transcript, register_work
from carlinpedia.ingest.normalize import normalize
from carlinpedia.ingest.segment import SEGMENT_PROMPT, SegmentationFailed, segment_document, store_segmentation
from carlinpedia.llm.client import LLM
from carlinpedia.vocab import load_vocab, sync_vocab

STEPS = ("normalize", "segment", "enrich", "embed")


@dataclass
class WorkReport:
    slug: str
    status: str = "skipped"  # skipped | ok | failed | needs_review
    steps_run: list[str] = field(default_factory=list)
    error: str | None = None
    unmatched_favorites: int = 0
    proposed: list[str] = field(default_factory=list)


@dataclass
class IngestReport:
    works: list[WorkReport]
    vectors_reset: bool = False


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _last_ok_hash(conn: sqlite3.Connection, work_id: int, step: str) -> str | None:
    row = conn.execute(
        "SELECT input_hash FROM pipeline_step WHERE work_id = ? AND step = ? AND status = 'ok'", (work_id, step)
    ).fetchone()
    return None if row is None else row["input_hash"]


def _record(conn: sqlite3.Connection, work_id: int, step: str, input_hash: str, status: str, error: str | None = None):
    conn.execute(
        "INSERT INTO pipeline_step (work_id, step, input_hash, status, error, finished_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now')) ON CONFLICT(work_id, step) DO UPDATE SET "
        "input_hash = excluded.input_hash, status = excluded.status, error = excluded.error, "
        "finished_at = excluded.finished_at",
        (work_id, step, input_hash, status, error),
    )
    conn.commit()


def run_ingest(
    conn: sqlite3.Connection,
    cfg: Config,
    llm: LLM,
    embedder: Embedder,
    slugs: list[str] | None = None,
    force_from: str | None = None,
) -> IngestReport:
    """Ingest the given works (all works when slugs is None). force_from re-runs that step and every later step."""
    if force_from is not None and force_from not in STEPS:
        raise ValueError(f"force_from must be one of {STEPS}")
    forced = set(STEPS[STEPS.index(force_from):]) if force_from else set()

    sync_vocab(conn, cfg.vocab_dir)
    vocab = load_vocab(conn)
    vectors_reset = prepare_embedding_space(conn, embedder)
    reports = []
    for slug in slugs if slugs is not None else discover_slugs(cfg.corpus_dir):
        reports.append(_ingest_one(conn, cfg, llm, embedder, vocab, slug, forced))
    return IngestReport(works=reports, vectors_reset=vectors_reset)


def _ingest_one(conn, cfg, llm, embedder, vocab, slug: str, forced: set[str]) -> WorkReport:
    report = WorkReport(slug=slug)
    step, work_id, input_hash = "register", None, ""

    def should_run(name: str, h: str) -> bool:
        return name in forced or _last_ok_hash(conn, work_id, name) != h

    try:
        reg = register_work(conn, cfg.corpus_dir, slug)
        work_id = reg.work_id

        step = "normalize"
        input_hash = reg.checksum
        doc = normalize(read_transcript(reg.source_path), reg.manifest.source.has_timestamps,
                        reg.manifest.source.strip_patterns)
        if should_run(step, input_hash):
            _record(conn, work_id, step, input_hash, "ok")
            report.steps_run.append(step)

        step = "segment"
        segment_hash = input_hash = _hash(doc.content_hash, SEGMENT_PROMPT, llm.model, str(cfg.max_passage_units))
        if should_run(step, input_hash):
            result = segment_document(llm, doc, work_title=reg.manifest.title, key=slug,
                                      max_passage_units=cfg.max_passage_units)
            with conn:
                favorites = snapshot_favorites(conn, work_id)
                store_segmentation(conn, work_id, doc, result)
                report.unmatched_favorites = len(restore_favorites(conn, work_id, favorites))
            _record(conn, work_id, step, input_hash, "ok")
            report.steps_run.append(step)

        step = "enrich"
        enrich_hash = input_hash = _hash(segment_hash, ENRICH_PROMPT, llm.model, vocab.fingerprint())
        if should_run(step, input_hash):
            for bit in load_bits(conn, work_id):
                enrichment = enrich_bit(llm, bit, vocab, key=f"{slug}:{bit.ordinal}")
                with conn:
                    report.proposed += apply_enrichment(conn, bit, enrichment, vocab, model=llm.model).proposed
            _record(conn, work_id, step, input_hash, "ok")
            report.steps_run.append(step)

        step = "embed"
        input_hash = _hash(enrich_hash, embedder_key(embedder))
        if should_run(step, input_hash):
            with conn:
                index_work(conn, embedder, work_id)
            _record(conn, work_id, step, input_hash, "ok")
            report.steps_run.append(step)

        report.status = "ok" if report.steps_run else "skipped"
    except SegmentationFailed as exc:
        report.status, report.error = "needs_review", str(exc)
        _record(conn, work_id, step, input_hash, "needs_review", str(exc))
    except Exception as exc:  # one bad work must not stop the rest
        report.status, report.error = "failed", f"{step}: {exc}"
        if work_id is not None and step in STEPS:
            _record(conn, work_id, step, input_hash, "failed", str(exc))
    return report
