import pytest

from carlinpedia.ingest.embed import EmbeddingMismatch, HashEmbedder
from carlinpedia.search.hybrid import SearchError, SearchFilters, fts_query, search


def bit_titles(results):
    return [r["bit"]["title"] for r in results]


def test_fts_query_quotes_tokens_and_survives_punctuation():
    assert fts_query('What "about" AND/OR (pre-owned)?') == '"what" OR "about" OR "and" OR "or" OR "pre" OR "owned"'
    assert fts_query("?!") is None


def test_keyword_query_finds_the_bit(ingested, embedder):
    results = search(ingested, embedder, "storage unit for my stuff")
    assert bit_titles(results)[0] == "Stuff"


def test_thesis_vectors_match_idea_level_queries(ingested, embedder):
    results = search(ingested, embedder, "renames poverty instead of fixing it", retrievers=("thesis",))
    assert results[0]["text"].startswith("We don't have poor people")


def test_duplicate_across_works_is_collapsed(ingested, embedder):
    results = search(ingested, embedder, "used car pre-owned vehicle")
    top = results[0]
    assert "pre-owned" in top["text"]
    assert len(top["also_in"]) == 1
    all_ids = [r["passage_id"] for r in results] + [a["passage_id"] for r in results for a in r["also_in"]]
    assert len(all_ids) == len(set(all_ids))


def test_theme_filter_includes_descendants_and_accepts_names(ingested, embedder):
    by_parent = search(ingested, embedder, "things", SearchFilters(themes=("language",)), limit=25)
    by_name = search(ingested, embedder, "things", SearchFilters(themes=("Euphemism",)), limit=25)
    assert by_parent and all("Euphemism" in r["themes"] for r in by_name)
    assert {r["passage_id"] for r in by_name} <= {r["passage_id"] for r in by_parent}


def test_year_kind_and_target_filters(ingested, embedder):
    album_only = search(ingested, embedder, "vote complain", SearchFilters(kinds=("album",)), limit=25)
    assert {r["work"]["title"] for r in album_only} == {"Test Album"}
    early = search(ingested, embedder, "stuff", SearchFilters(year_to=1990), limit=25)
    assert {r["work"]["year"] for r in early} == {1990}
    politicians = search(ingested, embedder, "vote", SearchFilters(targets=("politicians",)), limit=25)
    assert bit_titles(politicians) and set(bit_titles(politicians)) == {"Voting"}


def test_favorites_only(ingested, embedder):
    assert search(ingested, embedder, "stuff", SearchFilters(favorites_only=True)) == []
    ingested.execute("UPDATE passage SET favorite = 1 WHERE text LIKE 'Ever notice%'")
    results = search(ingested, embedder, "stuff", SearchFilters(favorites_only=True))
    assert [r["favorite"] for r in results] == [True]


def test_proposed_themes_are_not_filterable_or_shown(ingested, embedder):
    with pytest.raises(SearchError, match="list_themes"):
        search(ingested, embedder, "security", SearchFilters(themes=("Security theater",)))
    results = search(ingested, embedder, "joke about security")
    assert all("Security theater" not in r["themes"] for r in results)


def test_bad_inputs_raise_search_error(ingested, embedder):
    with pytest.raises(SearchError, match="empty"):
        search(ingested, embedder, "   ")
    with pytest.raises(SearchError, match="kind"):
        search(ingested, embedder, "x", SearchFilters(kinds=("podcast",)))


def test_limit_is_clamped(ingested, embedder):
    assert len(search(ingested, embedder, "the", limit=0)) == 1
    assert len(search(ingested, embedder, "the you a", limit=500)) <= 25


def test_punctuation_only_query_still_uses_vectors(ingested, embedder):
    assert isinstance(search(ingested, embedder, "?!"), list)


def test_mismatched_embedder_is_reported(ingested):
    with pytest.raises(EmbeddingMismatch, match="carlin ingest --from embed"):
        search(ingested, HashEmbedder(dim=32), "stuff")
