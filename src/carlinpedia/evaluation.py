"""Measure retrieval quality against a hand-written golden set."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from carlinpedia.config import Config
from carlinpedia.ingest.embed import Embedder
from carlinpedia.search.hybrid import search

MODES: dict[str, tuple[str, ...]] = {
    "fts": ("fts",),
    "text": ("text",),
    "thesis": ("thesis",),
    "fused": ("fts", "text", "thesis"),
}


class GoldenItem(BaseModel):
    news: str
    queries: list[str]
    expect_bits: list[str]


@dataclass(frozen=True)
class EvalRow:
    news: str
    query: str
    mode: str
    recall: float
    missing: tuple[str, ...]


def load_golden(path: Path) -> list[GoldenItem]:
    if not path.is_file():
        raise FileNotFoundError(f"No golden set at {path}")
    try:
        return [GoldenItem.model_validate(item) for item in yaml.safe_load(path.read_text(encoding="utf-8")) or []]
    except ValidationError as exc:
        raise ValueError(f"{path} is invalid:\n{exc}") from exc


def _found_titles(results: list[dict], conn: sqlite3.Connection) -> list[str]:
    titles = [r["bit"]["title"] for r in results]
    for r in results:
        for dup in r["also_in"]:
            row = conn.execute("SELECT b.title FROM passage p JOIN bit b ON b.id = p.bit_id WHERE p.id = ?",
                               (dup["passage_id"],)).fetchone()
            titles.append(row["title"])
    return [t.lower() for t in titles]


def run_eval(conn: sqlite3.Connection, embedder: Embedder, items: list[GoldenItem], cfg: Config, k: int = 10) -> list[EvalRow]:
    """For each query and retrieval mode, the share of expected bits found in the top k.
    An expected bit matches any result whose bit title contains it, ignoring case."""
    rows = []
    for item in items:
        expected = [e.lower() for e in item.expect_bits]
        for query in item.queries:
            for mode, retrievers in MODES.items():
                results = search(conn, embedder, query, limit=k, rrf_k=cfg.rrf_k, depth=cfg.retriever_depth,
                                 dedup_threshold=cfg.dedup_threshold, retrievers=retrievers)
                titles = _found_titles(results, conn)
                missing = tuple(e for e in expected if not any(e in t for t in titles))
                recall = 1.0 - len(missing) / len(expected) if expected else 1.0
                rows.append(EvalRow(item.news, query, mode, recall, missing))
    return rows


def summarize(rows: list[EvalRow]) -> dict[str, float]:
    by_mode: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_mode[row.mode].append(row.recall)
    return {mode: sum(v) / len(v) for mode, v in by_mode.items()}
