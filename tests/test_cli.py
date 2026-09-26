import json

import pytest
from typer.testing import CliRunner

from carlinpedia import cli

runner = CliRunner()


@pytest.fixture
def wired(project, fake_llm, embedder, monkeypatch):
    monkeypatch.setattr(cli, "make_llm", lambda cfg: fake_llm)
    monkeypatch.setattr(cli, "make_embedder", lambda cfg: embedder)
    return ["--root", str(project.root)]


def run(args):
    result = runner.invoke(cli.app, args)
    return result.exit_code, result.output


def test_ingest_then_search(wired):
    code, out = run(wired + ["ingest"])
    assert code == 0, out
    assert "test-special-1990: ok (normalize, segment, enrich, embed)" in out
    assert "1 vocabulary proposal(s)" in out
    code, out = run(wired + ["search", "storage unit stuff", "--limit", "1"])
    assert code == 0 and "Test Special (1990) · Stuff" in out


def test_ingest_rejects_unknown_step(wired):
    code, out = run(wired + ["ingest", "--from", "nope"])
    assert code == 2 and "--from must be one of" in out


def test_ingest_failure_sets_exit_code(wired, fake_llm):
    fake_llm.responses["segment_v1:test-album-1992"] = [{"bits": []}]
    code, out = run(wired + ["ingest"])
    assert code == 1
    assert "test-album-1992: needs_review" in out


def test_review_commands(wired):
    run(wired + ["ingest"])
    code, out = run(wired + ["review", "works"])
    assert code == 0 and "test-album-1992" in out and "embed" in out
    code, out = run(wired + ["review", "works", "test-special-1990"])
    assert "[1] Euphemisms" in out and "Society renames poverty" in out
    code, out = run(wired + ["review", "vocab"])
    assert "proposed.security-theater" in out
    code, out = run(wired + ["review", "vocab", "--approve", "proposed.security-theater"])
    assert code == 0 and "approved theme" in out
    code, out = run(wired + ["review", "vocab", "--reject", "no.such.slug"])
    assert code == 1
    code, out = run(wired + ["review", "favorites"])
    assert "No orphaned favorites." in out


def test_export_favorites(wired, project):
    run(wired + ["ingest"])
    code, out = run(wired + ["export-favorites"])
    assert code == 0 and "Wrote 0 favorite(s)" in out
    assert json.loads((project.root / "data" / "favorites.json").read_text()) == []


def test_eval_command(wired, project):
    run(wired + ["ingest"])
    (project.root / "eval").mkdir()
    (project.root / "eval" / "golden.yaml").write_text(
        '- news: "Storage"\n  queries: ["storage unit stuff"]\n  expect_bits: ["Stuff"]\n')
    code, out = run(wired + ["eval"])
    assert code == 0 and "fused   recall@10 = 1.00" in out


def test_missing_config_is_a_clear_error(tmp_path):
    code, out = run(["--root", str(tmp_path), "review", "works"])
    assert code == 2 and "carlinpedia.toml" in out


def test_dry_run_estimates_without_calling_the_model(wired, monkeypatch):
    class Counter:
        model = "counter"

        def count_tokens(self, *, system, user):
            return 1000

    monkeypatch.setattr(cli, "make_llm", lambda cfg: Counter())
    code, out = run(wired + ["ingest", "--dry-run"])
    assert code == 0, out
    assert "test-album-1992: 1,000 transcript tokens" in out
    assert "Estimated total: 34,000 input and 10,000 output tokens, about $0.17" in out
