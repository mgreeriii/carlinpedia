from pathlib import Path

import pytest

from carlinpedia.config import ConfigError, load_config, resolve_root


def test_load_config_reads_values_and_resolves_paths(tmp_path: Path):
    (tmp_path / "carlinpedia.toml").write_text(
        '[paths]\ndb = "x/test.db"\n[ingest]\nmodel = "claude-opus-5"\n[search]\nrrf_k = 30\n'
    )
    cfg = load_config(tmp_path)
    assert cfg.db_path == tmp_path / "x" / "test.db"
    assert cfg.corpus_dir == tmp_path / "corpus"
    assert cfg.ingest_model == "claude-opus-5"
    assert cfg.rrf_k == 30
    assert cfg.embedding_dim == 768


def test_load_config_missing_file_names_the_fix(tmp_path: Path):
    with pytest.raises(ConfigError, match="CARLINPEDIA_ROOT"):
        load_config(tmp_path)


def test_load_config_rejects_bad_toml(tmp_path: Path):
    (tmp_path / "carlinpedia.toml").write_text("[paths\n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(tmp_path)


def test_resolve_root_prefers_explicit_then_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CARLINPEDIA_ROOT", str(tmp_path / "env"))
    assert resolve_root(tmp_path) == tmp_path.resolve()
    assert resolve_root() == (tmp_path / "env").resolve()
