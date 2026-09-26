import pytest

from carlinpedia.evaluation import GoldenItem, load_golden, run_eval, summarize


def test_load_golden_validates(tmp_path):
    path = tmp_path / "golden.yaml"
    path.write_text('- news: "Fee story"\n  queries: ["fees"]\n  expect_bits: ["Euphemisms"]\n')
    assert load_golden(path) == [GoldenItem(news="Fee story", queries=["fees"], expect_bits=["Euphemisms"])]
    path.write_text('- news: "No queries"\n')
    with pytest.raises(ValueError, match="queries"):
        load_golden(path)
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "missing.yaml")


def test_run_eval_reports_recall_per_mode(ingested, embedder, project):
    items = [GoldenItem(news="Storage story", queries=["storage unit stuff"], expect_bits=["stuff", "Nonexistent Bit"])]
    rows = run_eval(ingested, embedder, items, project, k=10)
    assert {r.mode for r in rows} == {"fts", "text", "thesis", "fused"}
    fused = next(r for r in rows if r.mode == "fused")
    assert fused.recall == 0.5
    assert fused.missing == ("nonexistent bit",)
    assert summarize(rows)["fused"] == 0.5


def test_duplicates_count_toward_recall(ingested, embedder, project):
    items = [GoldenItem(news="Car story", queries=["used car pre-owned vehicle"], expect_bits=["Pre-Owned", "Euphemisms"])]
    fused = next(r for r in run_eval(ingested, embedder, items, project) if r.mode == "fused")
    assert fused.recall == 1.0
