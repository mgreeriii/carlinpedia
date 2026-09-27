from pathlib import Path

import pytest

from carlinpedia.db.schema import connect
from carlinpedia.ingest.enrich import BitEnrichment, apply_enrichment, enrich_bit, load_bits, render_bit
from carlinpedia.llm.client import FakeLLM, LLMError
from carlinpedia.vocab import load_vocab, sync_vocab

REPO_VOCAB = Path(__file__).resolve().parents[1] / "vocab"


def entry(ordinal, themes=(("language.euphemism", 0.9),), targets=("advertisers-marketers",), **extra):
    return {
        "ordinal": ordinal, "thesis": f"Thesis {ordinal}.",
        "themes": [{"slug": s, "weight": w} for s, w in themes],
        "proposed_themes": extra.get("proposed_themes", []), "targets": list(targets),
        "proposed_targets": extra.get("proposed_targets", []), "tone": "rant", "profanity": False,
    }


@pytest.fixture
def db():
    conn = connect(":memory:", embedding_dim=8)
    sync_vocab(conn, REPO_VOCAB)
    work_id = conn.execute("INSERT INTO work (slug, title, kind) VALUES ('w', 'W', 'special')").lastrowid
    bit_id = conn.execute("INSERT INTO bit (work_id, ordinal, title, summary) VALUES (?, 0, 'Euphemisms', 'Soft words')",
                          (work_id,)).lastrowid
    for i, text in enumerate(["Toilet paper became bathroom tissue.", "Used cars became pre-owned."]):
        conn.execute("INSERT INTO passage (bit_id, ordinal, text, content_hash, unit_start, unit_end, char_start, "
                     "char_end, favorite) VALUES (?, ?, ?, ?, ?, ?, 0, 1, ?)", (bit_id, i, text, f"h{i}", i, i, i))
    return conn, work_id


def test_load_and_render_bit(db):
    conn, work_id = db
    [bit] = load_bits(conn, work_id)
    assert [p.ordinal for p in bit.passages] == [0, 1]
    rendered = render_bit(bit)
    assert '<passage ordinal="1">\nUsed cars became pre-owned.\n</passage>' in rendered


def test_enrich_bit_puts_vocab_in_system_prompt_and_retries_bad_ordinals(db):
    conn, work_id = db
    [bit] = load_bits(conn, work_id)
    fake = FakeLLM({"enrich_v1:w:0": [{"passages": [entry(0)]}, {"passages": [entry(0), entry(1)]}]})
    result = enrich_bit(fake, bit, load_vocab(conn), key="w:0")
    assert [p.ordinal for p in result.passages] == [0, 1]
    assert "language.euphemism" in fake.calls[0]["system"]
    assert "expected exactly one entry" in fake.calls[1]["user"]


def test_enrich_bit_raises_after_second_bad_answer(db):
    conn, work_id = db
    [bit] = load_bits(conn, work_id)
    fake = FakeLLM({"enrich_v1:w:0": [{"passages": [entry(0), entry(0)]}]})
    with pytest.raises(LLMError, match="expected exactly one entry"):
        enrich_bit(fake, bit, load_vocab(conn), key="w:0")


def test_apply_enrichment_writes_fields_links_and_keeps_favorite(db):
    conn, work_id = db
    [bit] = load_bits(conn, work_id)
    vocab = load_vocab(conn)
    result = BitEnrichment.model_validate({"passages": [entry(0), entry(1, themes=(("language.euphemism", 3.0),))]})
    apply_enrichment(conn, bit, result, vocab, model="claude-sonnet-5")
    row = conn.execute("SELECT thesis, tone, profanity, favorite, enriched_model, enriched_prompt_version "
                       "FROM passage WHERE ordinal = 1").fetchone()
    assert tuple(row) == ("Thesis 1.", "rant", 0, 1, "claude-sonnet-5", "enrich_v1")
    weights = [r[0] for r in conn.execute("SELECT weight FROM passage_theme ORDER BY passage_id")]
    assert weights == [0.9, 1.0]
    assert conn.execute("SELECT count(*) FROM passage_target").fetchone()[0] == 2


def test_apply_enrichment_turns_unknown_slugs_into_proposals(db):
    conn, work_id = db
    [bit] = load_bits(conn, work_id)
    result = BitEnrichment.model_validate({"passages": [
        entry(0, themes=(("made.up", 0.7),), targets=("airlines",)),
        entry(1, proposed_themes=[{"name": "Hidden fees", "description": "Charging for everything"}]),
    ]})
    report = apply_enrichment(conn, bit, result, load_vocab(conn), model="m")
    assert sorted(report.proposed) == ["target: airlines", "theme: Hidden fees", "theme: made.up"]
    statuses = {r["slug"]: r["status"] for r in conn.execute("SELECT slug, status FROM theme WHERE slug LIKE 'proposed.%'")}
    assert statuses == {"proposed.made-up": "proposed", "proposed.hidden-fees": "proposed"}


def test_reapplying_replaces_links(db):
    conn, work_id = db
    [bit] = load_bits(conn, work_id)
    vocab = load_vocab(conn)
    apply_enrichment(conn, bit, BitEnrichment.model_validate({"passages": [entry(0), entry(1)]}), vocab, model="m")
    apply_enrichment(conn, bit, BitEnrichment.model_validate(
        {"passages": [entry(0, themes=(("language.jargon", 0.5),)), entry(1)]}), vocab, model="m")
    assert conn.execute("SELECT count(*) FROM passage_theme").fetchone()[0] == 2
