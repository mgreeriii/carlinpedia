from pathlib import Path

import pytest

from carlinpedia.db.schema import connect
from carlinpedia.vocab import (
    VocabError, descendant_theme_ids, list_proposed, load_vocab, resolve_terms, set_status, slugify,
    sync_vocab, upsert_proposed,
)

REPO_VOCAB = Path(__file__).resolve().parents[1] / "vocab"


@pytest.fixture
def conn():
    return connect(":memory:", embedding_dim=8)


def test_repo_seed_vocab_loads(conn):
    count = sync_vocab(conn, REPO_VOCAB)
    vocab = load_vocab(conn)
    assert count == len(vocab.themes) + len(vocab.targets)
    assert "language.euphemism" in vocab.themes
    assert vocab.themes["language.euphemism"].parent_slug == "language"
    assert "politicians" in vocab.targets


def test_sync_is_idempotent_and_reapproves(conn):
    sync_vocab(conn, REPO_VOCAB)
    set_status(conn, "language.euphemism", "rejected")
    sync_vocab(conn, REPO_VOCAB)
    assert "language.euphemism" in load_vocab(conn).themes


def test_malformed_yaml_entry_names_the_file(conn, tmp_path):
    (tmp_path / "themes.yaml").write_text("- {name: Missing slug}\n")
    (tmp_path / "targets.yaml").write_text("[]\n")
    with pytest.raises(VocabError, match="themes.yaml"):
        sync_vocab(conn, tmp_path)


def test_proposed_terms_are_hidden_until_approved(conn):
    sync_vocab(conn, REPO_VOCAB)
    term_id = upsert_proposed(conn, "theme", "Airline fees", "Charging for everything")
    assert term_id is not None
    assert "proposed.airline-fees" not in load_vocab(conn).themes
    assert [p["slug"] for p in list_proposed(conn)] == ["proposed.airline-fees"]
    set_status(conn, "proposed.airline-fees", "approved")
    assert "proposed.airline-fees" in load_vocab(conn).themes


def test_rejected_proposal_is_not_recreated(conn):
    upsert_proposed(conn, "target", "Airlines", "")
    set_status(conn, "proposed.airlines", "rejected")
    assert upsert_proposed(conn, "target", "Airlines", "") is None


def test_resolve_terms_accepts_slug_or_name_and_rejects_unknowns(conn):
    sync_vocab(conn, REPO_VOCAB)
    vocab = load_vocab(conn)
    terms = resolve_terms(vocab.themes, ["Euphemism", "language.jargon"], "theme")
    assert [t.slug for t in terms] == ["language.euphemism", "language.jargon"]
    with pytest.raises(VocabError, match="list_themes"):
        resolve_terms(vocab.themes, ["greed-ish"], "theme")


def test_descendant_theme_ids_includes_children(conn):
    sync_vocab(conn, REPO_VOCAB)
    vocab = load_vocab(conn)
    ids = descendant_theme_ids(conn, [vocab.themes["language"].id])
    assert vocab.themes["language.euphemism"].id in ids
    assert vocab.themes["power.politicians"].id not in ids


def test_slugify():
    assert slugify("  Corporate Greed & Lies! ") == "corporate-greed-lies"
    assert slugify("!!!") == "term"
