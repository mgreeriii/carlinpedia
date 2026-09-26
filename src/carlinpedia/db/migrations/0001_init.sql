CREATE TABLE work (
    id          INTEGER PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('special', 'album', 'book', 'interview')),
    recorded_on TEXT,
    released_on TEXT,
    venue       TEXT,
    notes       TEXT
);

CREATE TABLE source_document (
    id             INTEGER PRIMARY KEY,
    work_id        INTEGER NOT NULL REFERENCES work(id) ON DELETE CASCADE,
    path           TEXT NOT NULL,
    provenance     TEXT NOT NULL,
    quality        TEXT NOT NULL CHECK (quality IN ('verified', 'unverified', 'auto_transcribed')),
    has_timestamps INTEGER NOT NULL DEFAULT 0,
    checksum       TEXT NOT NULL,
    is_active      INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX source_document_one_active ON source_document(work_id) WHERE is_active = 1;

CREATE TABLE bit (
    id         INTEGER PRIMARY KEY,
    work_id    INTEGER NOT NULL REFERENCES work(id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    title      TEXT NOT NULL,
    summary    TEXT NOT NULL,
    routine_id INTEGER,
    UNIQUE (work_id, ordinal)
);

CREATE TABLE passage (
    id                      INTEGER PRIMARY KEY,
    bit_id                  INTEGER NOT NULL REFERENCES bit(id) ON DELETE CASCADE,
    ordinal                 INTEGER NOT NULL,
    text                    TEXT NOT NULL,
    content_hash            TEXT NOT NULL,
    unit_start              INTEGER NOT NULL,
    unit_end                INTEGER NOT NULL,
    char_start              INTEGER NOT NULL,
    char_end                INTEGER NOT NULL,
    start_ts                REAL,
    end_ts                  REAL,
    chapter                 TEXT,
    page                    INTEGER,
    thesis                  TEXT,
    tone                    TEXT CHECK (tone IS NULL OR tone IN
                                ('rant', 'observational', 'absurdist', 'wordplay', 'dark', 'reflective')),
    profanity               INTEGER,
    favorite                INTEGER NOT NULL DEFAULT 0,
    enriched_model          TEXT,
    enriched_prompt_version TEXT,
    enriched_at             TEXT,
    UNIQUE (bit_id, ordinal)
);
CREATE INDEX passage_content_hash ON passage(content_hash);

CREATE TABLE theme (
    id          INTEGER PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    parent_id   INTEGER REFERENCES theme(id),
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL CHECK (status IN ('approved', 'proposed', 'rejected'))
);

CREATE TABLE target (
    id          INTEGER PRIMARY KEY,
    slug        TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL CHECK (status IN ('approved', 'proposed', 'rejected'))
);

CREATE TABLE passage_theme (
    passage_id INTEGER NOT NULL REFERENCES passage(id) ON DELETE CASCADE,
    theme_id   INTEGER NOT NULL REFERENCES theme(id) ON DELETE CASCADE,
    weight     REAL NOT NULL,
    PRIMARY KEY (passage_id, theme_id)
);

CREATE TABLE passage_target (
    passage_id INTEGER NOT NULL REFERENCES passage(id) ON DELETE CASCADE,
    target_id  INTEGER NOT NULL REFERENCES target(id) ON DELETE CASCADE,
    PRIMARY KEY (passage_id, target_id)
);

CREATE TABLE pipeline_step (
    work_id     INTEGER NOT NULL REFERENCES work(id) ON DELETE CASCADE,
    step        TEXT NOT NULL CHECK (step IN ('normalize', 'segment', 'enrich', 'embed')),
    input_hash  TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'failed', 'needs_review')),
    error       TEXT,
    finished_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (work_id, step)
);

CREATE TABLE orphaned_favorite (
    id           INTEGER PRIMARY KEY,
    work_id      INTEGER NOT NULL REFERENCES work(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    text         TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE passage_fts USING fts5(text, thesis, bit_title, tokenize = 'porter');
