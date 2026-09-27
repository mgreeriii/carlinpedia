import shutil
from pathlib import Path

import pytest

from carlinpedia.config import load_config
from carlinpedia.db.schema import connect
from carlinpedia.ingest.embed import HashEmbedder
from carlinpedia.llm.client import FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    """A throwaway project root with the fixture corpus, the repo's vocab, and a config file."""
    shutil.copytree(FIXTURES / "corpus", tmp_path / "corpus")
    shutil.copytree(REPO_ROOT / "vocab", tmp_path / "vocab")
    (tmp_path / "carlinpedia.toml").write_text('[embedding]\nmodel = "hash"\ndim = 64\n')
    return load_config(tmp_path)


@pytest.fixture
def fake_llm():
    return FakeLLM.from_file(FIXTURES / "llm_responses.json")


@pytest.fixture
def embedder():
    return HashEmbedder(dim=64)


@pytest.fixture
def conn(project):
    connection = connect(project.db_path, embedding_dim=project.embedding_dim)
    yield connection
    connection.close()


@pytest.fixture
def ingested(conn, project, fake_llm, embedder):
    """A database with both fixture works fully ingested."""
    from carlinpedia.ingest.pipeline import run_ingest

    report = run_ingest(conn, project, fake_llm, embedder)
    assert [w.status for w in report.works] == ["ok", "ok"], report
    return conn
