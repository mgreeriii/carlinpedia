"""Load carlinpedia.toml into a Config."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILE = "carlinpedia.toml"
ROOT_ENV = "CARLINPEDIA_ROOT"


class ConfigError(Exception):
    """The config file is missing or malformed."""


@dataclass(frozen=True)
class Config:
    root: Path
    db_path: Path
    corpus_dir: Path
    vocab_dir: Path
    ingest_model: str = "claude-sonnet-5"
    max_passage_units: int = 40
    price_in_per_mtok: float = 2.0
    price_out_per_mtok: float = 10.0
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    rrf_k: int = 60
    retriever_depth: int = 50
    dedup_threshold: float = 0.92


def resolve_root(explicit: Path | None = None) -> Path:
    """Pick the project root: an explicit path, then $CARLINPEDIA_ROOT, then the cwd."""
    if explicit is not None:
        return explicit.resolve()
    env = os.environ.get(ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def load_config(root: Path) -> Config:
    path = root / CONFIG_FILE
    if not path.is_file():
        raise ConfigError(f"No {CONFIG_FILE} in {root}. Run carlin from the project root or set {ROOT_ENV}.")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc

    paths = data.get("paths", {})
    ingest = data.get("ingest", {})
    embedding = data.get("embedding", {})
    search = data.get("search", {})
    defaults = Config(root=root, db_path=root, corpus_dir=root, vocab_dir=root)
    return Config(
        root=root,
        db_path=root / paths.get("db", "data/carlin.db"),
        corpus_dir=root / paths.get("corpus", "corpus"),
        vocab_dir=root / paths.get("vocab", "vocab"),
        ingest_model=ingest.get("model", defaults.ingest_model),
        max_passage_units=int(ingest.get("max_passage_units", defaults.max_passage_units)),
        price_in_per_mtok=float(ingest.get("price_in_per_mtok", defaults.price_in_per_mtok)),
        price_out_per_mtok=float(ingest.get("price_out_per_mtok", defaults.price_out_per_mtok)),
        embedding_model=embedding.get("model", defaults.embedding_model),
        embedding_dim=int(embedding.get("dim", defaults.embedding_dim)),
        rrf_k=int(search.get("rrf_k", defaults.rrf_k)),
        retriever_depth=int(search.get("retriever_depth", defaults.retriever_depth)),
        dedup_threshold=float(search.get("dedup_threshold", defaults.dedup_threshold)),
    )
