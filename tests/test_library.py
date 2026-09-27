import pytest

from carlinpedia.library import NotFound, get_bit, get_passage, list_themes, list_works, set_favorite


def passage_id(conn, like):
    return conn.execute("SELECT id FROM passage WHERE text LIKE ?", (like,)).fetchone()[0]


def test_get_passage_returns_citation_and_neighbors(ingested):
    pid = passage_id(ingested, "The dump became%")
    result = get_passage(ingested, pid, context=1)
    assert result["work"] == {"slug": "test-special-1990", "title": "Test Special", "year": 1990, "kind": "special"}
    assert result["bit"]["title"] == "Euphemisms"
    assert result["bit"]["summary"] == "Soft names hide ugly realities."
    assert [n["text"][:9] for n in result["before"]] == ["Somewhere"]
    assert [n["text"][:9] for n in result["after"]] == ["We don't "]
    assert result["themes"][0] == "Euphemism"


def test_get_passage_context_zero_and_bounds(ingested):
    pid = passage_id(ingested, "Somewhere along%")
    result = get_passage(ingested, pid, context=0)
    assert result["before"] == [] and result["after"] == []


def test_get_bit_lists_passages_in_order(ingested):
    bit_id = ingested.execute("SELECT id FROM bit WHERE title = 'Stuff'").fetchone()[0]
    bit = get_bit(ingested, bit_id)
    assert [p["text"][:10] for p in bit["passages"]] == ["Ever notic", "You buy a "]
    assert bit["work"] == {"title": "Test Special", "year": 1990}


def test_list_works_counts(ingested):
    assert list_works(ingested) == [
        {"slug": "test-special-1990", "title": "Test Special", "kind": "special", "year": 1990, "bits": 5, "passages": 9},
        {"slug": "test-album-1992", "title": "Test Album", "kind": "album", "year": 1992, "bits": 3, "passages": 4},
    ]


def test_list_themes_only_approved_with_parents(ingested):
    vocab = list_themes(ingested)
    slugs = [t["slug"] for t in vocab["themes"]]
    assert "proposed.security-theater" not in slugs
    euphemism = next(t for t in vocab["themes"] if t["slug"] == "language.euphemism")
    assert euphemism["parent"] == "language"
    assert slugs.index("language") < slugs.index("language.euphemism")


def test_set_favorite_and_unknown_ids(ingested):
    pid = passage_id(ingested, "Ever notice%")
    assert set_favorite(ingested, pid, True) == {"passage_id": pid, "favorite": True}
    assert get_passage(ingested, pid)["favorite"] is True
    with pytest.raises(NotFound):
        set_favorite(ingested, 99999, True)
    with pytest.raises(NotFound):
        get_passage(ingested, 99999)
    with pytest.raises(NotFound):
        get_bit(ingested, 99999)
