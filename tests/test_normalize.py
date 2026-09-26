from carlinpedia.ingest.normalize import normalize


def texts(doc):
    return [doc.unit_text(i) for i in range(len(doc.units))]


def test_splits_sentences_and_keeps_closing_quotes():
    doc = normalize('He said "stop." Then he left! Why? Nobody knows.')
    assert texts(doc) == ['He said "stop."', "Then he left!", "Why?", "Nobody knows."]


def test_unit_offsets_slice_the_normalized_text():
    doc = normalize("First line.\nSecond   line here.\n\n\nNew paragraph")
    assert doc.text == "First line. Second line here.\n\nNew paragraph"
    for unit in doc.units:
        assert doc.text[unit.char_start:unit.char_end] == doc.unit_text(unit.index)
    assert texts(doc) == ["First line.", "Second line here.", "New paragraph"]


def test_paragraph_break_ends_a_unit_without_punctuation():
    doc = normalize("shit piss fuck\n\ncunt cocksucker")
    assert texts(doc) == ["shit piss fuck", "cunt cocksucker"]


def test_audience_cues_are_removed_and_attached_to_the_preceding_unit():
    doc = normalize("They call it bathroom tissue. [laughter] Isn't that nice? (audience applauds) Yes.")
    assert texts(doc) == ["They call it bathroom tissue.", "Isn't that nice?", "Yes."]
    assert [(c.kind, c.unit_index) for c in doc.cues] == [("laughter", 0), ("applause", 1)]


def test_other_stage_directions_and_speaker_labels_are_removed():
    doc = normalize("GEORGE CARLIN: Hello. [music playing] Goodbye.\nAnnouncer: Ladies and gentlemen.")
    assert texts(doc) == ["Hello.", "Goodbye.", "Ladies and gentlemen."]


def test_timestamps_parsed_only_when_enabled():
    raw = "[00:01:05] First bit starts.\n[1:10] Second thought."
    doc = normalize(raw, has_timestamps=True)
    assert texts(doc) == ["First bit starts.", "Second thought."]
    assert [u.ts for u in doc.units] == [65.0, 70.0]
    plain = normalize("Meet me at 7:30 tonight.")
    assert texts(plain) == ["Meet me at 7:30 tonight."]
    assert plain.units[0].ts is None


def test_long_unpunctuated_run_is_split_at_word_boundaries():
    raw = " ".join(["word"] * 300)
    doc = normalize(raw, max_unit_chars=100)
    assert len(doc.units) > 1
    assert all(u.char_end - u.char_start <= 100 for u in doc.units)
    assert " ".join(texts(doc)) == doc.text


def test_strip_patterns_remove_boilerplate_lines():
    doc = normalize("Transcript courtesy of Example.com\nReal words.", strip_patterns=[r"^Transcript courtesy.*$"])
    assert texts(doc) == ["Real words."]


def test_empty_and_whitespace_input_yield_no_units():
    assert normalize("   \n\n ").units == ()
    assert normalize("[applause]").cues == ()


def test_content_hash_changes_with_text():
    assert normalize("a.").content_hash != normalize("b.").content_hash
    assert normalize("a.").content_hash == normalize("a.").content_hash
