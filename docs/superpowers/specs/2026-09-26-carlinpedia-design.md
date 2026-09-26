# Carlinpedia: Design Spec

**Date:** 2026-09-26
**Status:** Draft for review

## 1. Purpose

Carlinpedia is a private, personal library of George Carlin's work. You give it a current event and it retrieves the Carlin passages that speak to it. The premise is that Carlin aimed at durable features of human nature and institutions. A passage written decades ago can map onto today's news through its underlying idea, even when the surface details differ.

### Success criteria

1. From inside Claude Desktop or Claude Code, the user pastes a headline or story and gets back a small set of relevant, verbatim Carlin passages. Each one is cited to work, year, bit, and location.
2. Matching works at the level of ideas. A story about a renamed layoff program finds the "Euphemisms" material even though no words overlap.
3. Quotes are always exact spans of the source transcript, never paraphrases.
4. When the same routine appears in several specials, the results show it once.
5. To add a new work, the user drops in a transcript and a short manifest and runs one command.

### Constraints

1. Personal use only. Transcripts and the database contain copyrighted material and are never published. `corpus/` and `data/` are gitignored.
2. Runs locally on macOS. The only hosted service is the Anthropic API, used during ingest.
3. Python.

## 2. Scope and stages

Stage 1 is enriched bits (Approach 2). It ingests transcripts of HBO specials and comedy albums, segments them into bits and passages, and enriches each passage with a thesis, themes, targets, and tone. It adds hybrid search and a local MCP server. This stage delivers the core use case end to end.

Stage 2 is cross-work linking (Approach 3, the end goal). It adds `Routine` (the same bit across performances) and `Claim` (a distinct truism, linked to every passage that expresses it). That enables "every time he made this point" and shows how an idea changed across his career.

Later work, not designed here:

- Books. The model already supports them through `Work.kind = book` and a chapter/page locator. The normalizer needs a book-specific front end.
- Remote MCP hosting.
- Audio transcription with Whisper.

Non-goals: a web UI, public sharing, automatic daily news digests, audio or video playback.

## 3. Architecture

```
corpus/<work>/            ingest pipeline (CLI)                 data/carlin.db
  manifest.yaml   ──►  normalize ─► segment ─► enrich ─► embed ──►  SQLite
  transcript.txt                   (Claude)   (Claude)  (local)     + FTS5
                                                                    + sqlite-vec
                                                                        │
Claude Desktop / Claude Code ◄── stdio ── MCP server (FastMCP) ─────────┘
```

The two entry points share one library:

1. The `carlin` CLI handles ingest, review, debug search, and evaluation.
2. The MCP server gives Claude read-mostly tools, plus `set_favorite`.

### Project layout

```
carlinpedia/
  pyproject.toml              # uv-managed
  carlinpedia.toml            # runtime config
  corpus/                     # gitignored: transcripts + manifests
    doin-it-again-1990/
      manifest.yaml
      transcript.txt
  data/carlin.db              # gitignored
  vocab/
    themes.yaml               # controlled theme hierarchy
    targets.yaml              # controlled target list
  eval/golden.yaml            # retrieval golden set
  src/carlinpedia/
    models.py                 # Pydantic models for entities + LLM schemas
    db/
      schema.py               # connection, migrations runner
      migrations/0001_init.sql
    ingest/
      manifest.py             # load + validate manifest.yaml
      normalize.py            # cleanup, audience cues, sentence units
      segment.py              # LLM segmentation + validator
      enrich.py               # LLM enrichment
      embed.py                # sentence-transformers wrapper
      favorites.py            # favorite carry-over on re-segment
      pipeline.py             # step orchestration, idempotency
    llm/
      client.py               # Anthropic client, batching, retries
      prompts/                # versioned prompt files (segment_v1.md, enrich_v1.md)
    search/
      hybrid.py               # FTS + vector queries, filters
      fusion.py               # reciprocal rank fusion
      dedup.py                # near-duplicate collapse
    mcp_server.py
    cli.py
  tests/
    fixtures/                 # synthetic mini-corpus + recorded LLM responses
```

## 4. Domain model

All entities live in SQLite. IDs are integer surrogate keys. Slugs are human-readable and unique where noted.

### Work

One release.

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `slug` | text, unique | e.g. `doin-it-again-1990` |
| `title` | text | *Doin' It Again* |
| `kind` | enum | `special` · `album` · `book` · `interview` |
| `recorded_on` | date, nullable | performance or taping date |
| `released_on` | date, nullable | release, air, or publication date |
| `venue` | text, nullable | |
| `notes` | text, nullable | |

A work's effective year is `recorded_on` if present, otherwise `released_on`. Passages inherit it.

### SourceDocument

The raw text a Work was built from.

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `work_id` | fk Work | |
| `path` | text | relative to `corpus/` |
| `provenance` | text | where the transcript came from |
| `quality` | enum | `verified` · `unverified` · `auto_transcribed` |
| `checksum` | text | sha256 of the raw file; drives re-ingest |
| `is_active` | bool | exactly one active per Work |

### Bit

One routine as performed within one Work.

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `work_id` | fk Work | |
| `ordinal` | int | position within the work |
| `title` | text | LLM-proposed, e.g. "Euphemisms" |
| `summary` | text | 1–2 sentences |
| `routine_id` | fk Routine, nullable | Stage 2 |

### Passage

One beat within a bit. This is the unit of search and quotation.

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `bit_id` | fk Bit | |
| `ordinal` | int | position within the bit |
| `text` | text | verbatim span of the normalized transcript |
| `content_hash` | text | sha256 of `text`; used for favorite carry-over |
| `unit_start`, `unit_end` | int | sentence-unit range (see 6.2) |
| `char_start`, `char_end` | int | offsets into the normalized text |
| `start_ts`, `end_ts` | real, nullable | seconds; only when the source has timestamps |
| `chapter`, `page` | text / int, nullable | books (later) |
| `thesis` | text | plain-language statement of the idea |
| `tone` | enum | `rant` · `observational` · `absurdist` · `wordplay` · `dark` · `reflective` |
| `profanity` | bool | |
| `favorite` | bool, default false | user-owned; the pipeline writes it only during carry-over |
| `enriched_model` | text | e.g. `claude-sonnet-5` |
| `enriched_prompt_version` | text | e.g. `enrich_v1` |
| `enriched_at` | timestamp | |

The Locator from the design discussion is the group of `char_*`, `*_ts`, `chapter`, and `page` fields. `char_start` and `char_end` are always present. The others depend on the source type and on what the transcript contains. Many web transcripts have no timestamps, so timestamps are optional.

### Theme

The idea. A curated hierarchy loaded from `vocab/themes.yaml`.

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `slug` | text, unique | `language.euphemism` |
| `name` | text | Euphemism |
| `parent_id` | fk Theme, nullable | |
| `description` | text | one line, shown to the enrichment prompt and to Claude via `list_themes` |
| `status` | enum | `approved` · `proposed` · `rejected` |

### Target

The object of the attack. A flat list loaded from `vocab/targets.yaml`, with the same `status` workflow as Theme.

| Field | Type |
|---|---|
| `id`, `slug`, `name`, `description`, `status` | as Theme, no hierarchy |

### Join tables

| Table | Columns | Notes |
|---|---|---|
| `passage_theme` | `passage_id`, `theme_id`, `weight` | weight 0–1 from enrichment; the highest weight is the primary theme |
| `passage_target` | `passage_id`, `target_id` | |

### Pipeline bookkeeping

`pipeline_step(work_id, step, input_hash, status, error, finished_at)` records each run. `step` is one of `normalize`, `segment`, `enrich`, `embed`. `status` is one of `ok`, `failed`, `needs_review`. The pipeline skips a step when its `input_hash` matches the last `ok` run.

### Stage 2 entities (added by a later migration)

| Entity | Fields | Notes |
|---|---|---|
| Routine | `id`, `slug`, `title`, `first_year` | `Bit.routine_id` links performances |
| Claim | `id`, `statement`, `status`, `created_model`, `created_prompt_version` | `statement` is the canonical truism in one sentence; `status` is `approved`, `proposed`, or `rejected` |
| PassageClaim | `passage_id`, `claim_id`, `relation`, `confidence` | `relation` is `states`, `elaborates`, or `revises` |
| claim_theme | `claim_id`, `theme_id` | |

## 5. Vocabularies

The repo ships a seed `vocab/themes.yaml` (about 45 leaf themes) and `vocab/targets.yaml` (about 15 targets). The user edits both before the first ingest. Starter themes, grouped by parent:

| Parent | Leaf themes |
|---|---|
| Language | euphemism, soft language, jargon and bureaucratese, political correctness, clichés and empty phrases, advertising language |
| Power and politics | politicians, the owners and class, voting and democracy, patriotism and flag-waving, war and militarism, government control |
| Money and consumption | consumerism, stuff and possessions, corporate greed, advertising and marketing, the American Dream |
| Belief | organized religion, God, superstition and magical thinking, prayer |
| Society | conformity, rights and freedoms, law and order, education and schooling, children and parenting, family, sports, celebrity, media and news |
| Human nature | hypocrisy, self-deception, stupidity, fear and safety obsession, selfishness, violence, human arrogance |
| Planet and mortality | the environment ("the planet is fine"), death, disease and the body |
| Everyday absurdity | travel and airlines, driving, food, pets and animals, modern life and technology, health and the self-help industry |

Starter targets: politicians, corporations and "the owners", advertisers and marketers, organized religion, the news media, the military, police and courts, lawyers, parents, the self-help industry, Americans, the human species, "people" in general, conservatives, liberals, himself.

Enrichment may propose new themes or targets. The pipeline stores them with `status = proposed`. Search filters ignore them until the user approves them with `carlin review vocab`.

## 6. Ingest pipeline

The command is `carlin ingest [<slug>...]`, and it runs every work when no slug is given. Steps run in order per work. Each step is idempotent and is skipped when its input hash has not changed. A failure in one work does not stop the others.

### 6.1 Register

The pipeline reads `corpus/<slug>/manifest.yaml`:

```yaml
title: "Doin' It Again"
kind: special
recorded_on: 1990-02-17
released_on: 1990-06-23
venue: "Park West, Chicago"   # example value
source:
  file: transcript.txt
  provenance: "Transcript from <site>, cleaned by hand"
  quality: unverified
  has_timestamps: false
```

Pydantic validates it, then the pipeline upserts the Work and SourceDocument. When the checksum changes, the pipeline deactivates the old SourceDocument and invalidates every later step.

### 6.2 Normalize

This step is deterministic and uses no LLM.

1. Strip site boilerplate, speaker labels, and bracketed stage directions other than audience cues.
2. Move audience cues (`[applause]`, `[laughter]`, `(audience laughs)`) into an annotation list with their positions, and remove them from the text.
3. If `has_timestamps` is true, parse timestamp markers into a unit-to-seconds map and remove them from the text.
4. Split the text into sentence units numbered `0..N`, each with `char_start` and `char_end`.

The output is the normalized text, the units, and the cues. Sentence units carry the verbatim guarantee: every later step refers to text only by unit ranges.

### 6.3 Segment (Claude)

The pipeline sends one request per work. The input is the numbered units (`[17] Somewhere along the line...`), with audience cues shown inline as `⟨laugh⟩` hints. The output is structured JSON, parsed with `client.messages.parse` against a Pydantic schema:

```json
{"bits": [{"title": "Euphemisms", "summary": "...", "unit_start": 120, "unit_end": 212,
           "passages": [{"unit_start": 120, "unit_end": 131}, ...]}]}
```

The model returns only unit ranges and labels, never quoted text, so it cannot paraphrase.

The validator checks these rules:

1. Bits are contiguous, ordered, non-overlapping, and cover units `0..N`. Preamble and outro can be bits titled "Intro" and "Outro".
2. Passages tile each bit the same way.
3. Each passage is 1 to 40 units long.

If validation fails, the pipeline retries once with the validator's error list added to the prompt. If the retry also fails, it marks the step `needs_review` and moves to the next work. `carlin review works` lists those works.

The pipeline then slices each passage's text from the normalized text, from `char_start` of its first unit to `char_end` of its last.

### 6.4 Enrich (Claude)

The pipeline sends one request per bit rather than per passage, so the model sees the whole bit as context and returns one entry per passage. The input is the bit title, the bit text with passage boundaries marked, and the approved theme and target vocabularies with their descriptions. The output for each passage looks like this:

```json
{"ordinal": 3, "thesis": "People soften language to avoid facing unpleasant truths.",
 "themes": [{"slug": "language.euphemism", "weight": 0.9}],
 "proposed_themes": [], "targets": ["advertisers-marketers"], "proposed_targets": [],
 "tone": "rant", "profanity": false}
```

The prompt sets these rules:

1. The thesis states Carlin's underlying claim about people or institutions in neutral, general language, without the joke's specifics.
2. Each passage gets one to four themes.
3. The schema restricts `themes` and `targets` to approved slugs. New ideas go in `proposed_themes` and `proposed_targets`.

Re-enrichment (`carlin ingest --only enrich --force`) overwrites only the LLM fields and the `enriched_*` provenance. It never touches `favorite`.

### 6.5 Embed (local)

The pipeline uses `sentence-transformers` with `BAAI/bge-base-en-v1.5` (768 dimensions, normalized). Each passage gets two vectors, one for `text` and one for `thesis`. Queries use the BGE query instruction prefix. The model name is stored in a `meta` table, and changing it triggers a full re-embed.

### 6.6 Favorite carry-over on re-segment

Before replacing a work's passages, the pipeline snapshots each favorited passage's `content_hash`, `char_start`, `char_end`, and `text`. After re-segmenting, it restores favorites in this order:

1. Match by identical `content_hash`.
2. Otherwise, match the new passage with the largest character-range overlap, if that overlap covers at least 50% of the old passage.
3. Report anything still unmatched on stdout and in `carlin review favorites`. Nothing is dropped silently.

### 6.7 Claude API usage

| Concern | Choice |
|---|---|
| SDK | `anthropic` (Python) |
| Model | `claude-sonnet-5` for segment and enrich, configurable in `carlinpedia.toml`; adaptive thinking on |
| Structured output | `client.messages.parse` with a Pydantic model (`output_config.format` under the hood) |
| Prompt caching | on the system prompt and vocabularies, the stable prefix shared by every enrich call |
| Bulk runs | `--batch` sends requests through the Message Batches API at 50% cost, asynchronously; single-work ingest calls the API directly |
| Cost | about $10–30 for ~25 works (~400k transcript tokens) at $2 / $10 per million tokens in/out, roughly half with `--batch`; `carlin ingest --dry-run` prints a `count_tokens` estimate before spending |
| Errors | the SDK retries 429 and 5xx responses; a `refusal` or a schema failure marks the step `failed` with the reason, and the pipeline moves on |

## 7. Index and retrieval

### 7.1 Physical index

1. An FTS5 virtual table `passage_fts(text, thesis, bit_title)`, with the porter tokenizer and `bm25` ranking. Column weights are `text` 1.0, `thesis` 0.6, `bit_title` 0.4.
2. Two sqlite-vec `vec0` tables: `vec_passage_text(passage_id, embedding float[768])` and `vec_passage_thesis`, with the same shape.

### 7.2 `search(query, filters, limit)`

1. Run three retrievers, each returning its top 50 passage IDs: FTS (bm25), text vectors (cosine), and thesis vectors (cosine).
2. Merge them with reciprocal rank fusion: `score = Σ 1 / (60 + rank_r)` over each retriever `r` that returned the passage.
3. Apply filters as SQL constraints on the candidates: `themes` (the theme or any descendant), `targets`, `year_from` and `year_to`, `kinds`, and `favorites_only`. Only approved vocabulary can be used as a filter.
4. Collapse near-duplicates. Walk the fused list, and when a passage's text vector has cosine ≥ 0.92 with an already-kept passage from a different work, fold it into that result's `also_in` list (work, year, passage_id). Stage 2 also folds passages that share a `routine_id`.
5. Return the top `limit` results (default 10, max 25).

### 7.3 Result shape

```json
{"passage_id": 812, "text": "...", "thesis": "...",
 "work": {"title": "Doin' It Again", "year": 1990, "kind": "special"},
 "bit": {"id": 44, "title": "Euphemisms"},
 "locator": {"start_ts": null, "chapter": null},
 "themes": ["Euphemism", "Advertising language"], "targets": ["advertisers & marketers"],
 "tone": "rant", "profanity": true, "favorite": false,
 "also_in": [{"work": "Jammin' in New York", "year": 1992, "passage_id": 1377}],
 "score": 0.041}
```

## 8. MCP server

The server lives in `src/carlinpedia/mcp_server.py`. It uses the official `mcp` Python SDK (FastMCP) over the stdio transport. The user registers it in Claude Desktop's config and with `claude mcp add` for Claude Code. It opens `data/carlin.db` read-write, and only `set_favorite` writes.

### Tools

| Tool | Purpose |
|---|---|
| `search_passages(query, themes?, targets?, year_from?, year_to?, kinds?, favorites_only?, limit?)` | the search in 7.2 |
| `get_passage(passage_id, context=1)` | the passage plus `context` neighbors on each side within the bit, the bit summary, and the full citation |
| `get_bit(bit_id)` | the whole bit, passage by passage |
| `list_works()` | works with kind, year, and bit and passage counts |
| `list_themes()` | the approved theme hierarchy and targets with descriptions, so Claude can choose filters |
| `set_favorite(passage_id, favorite)` | lets the user say "favorite that one" in chat |

Stage 2 adds three tools: `search_claims(query, limit?)`, `get_claim(claim_id)` (the statement plus linked passages in chronological order), and `get_routine(routine_id)` (all performances).

### Prompt

An MCP prompt, `carlin_on(news)`, gives Claude the intended workflow:

1. Identify the human behaviors and institutional patterns in the story, stated generally.
2. Run 3–5 `search_passages` queries. Phrase some as general theses and some in concrete, Carlin-like language. Use theme and target filters where they clearly apply.
3. Call `get_passage` for context on promising hits.
4. Present the best 2–4 passages verbatim with citations, each with one line on why it fits. Never alter quoted text, and say so when nothing fits well.

The server's `instructions` field carries a short version of the same guidance.

## 9. Stage 2: Routines and claims

Stage 2 starts after Stage 1 is in daily use. It adds migration `0002_stage2.sql`.

### 9.1 Routines

1. Find candidates: pairs of bits from different works whose titles are similar after normalization, or whose mean passage-text-vector cosine is at least 0.85.
2. Confirm: Claude judges each candidate pair as the same routine, related, or different, and returns structured JSON.
3. Cluster: connected components of "same routine" edges become Routines, titled after the earliest performance.
4. Review: `carlin review routines` shows each cluster for approve, split, or merge before the pipeline writes `Bit.routine_id`.

### 9.2 Claims

1. Extract: for each bit, Claude proposes 0–2 atomic claims per passage. A claim is a general, standalone truism, such as "Consumer culture teaches people to find meaning in accumulating things." Candidates are stored with a link to their passage.
2. Cluster: embed the candidates and group them with agglomerative clustering at cosine ≥ 0.82.
3. Canonicalize: for each cluster, Claude writes one canonical `statement`, gives each member passage a `relation` (states, elaborates, or revises) and a `confidence`, and splits clusters that mix distinct ideas.
4. Review: `carlin review claims` supports approve, edit, and merge. MCP tools serve only approved claims.
5. Index: FTS5 and a vector table on `Claim.statement`. `search_passages` gains an optional claim-expansion step, where the top-matching claims contribute their linked passages as a fourth RRF retriever.

## 10. Evaluation and testing

### Unit tests (pytest)

| Module | What the tests cover |
|---|---|
| `normalize` | cue extraction, timestamp parsing, sentence-unit offsets that round-trip to the original text |
| `segment` validator | gaps, overlaps, out-of-order ranges, out-of-range units, over-long passages |
| `fusion` | RRF ordering on hand-built rankings |
| `dedup` | folds duplicates across works but not within one work |
| `favorites` | hash match, overlap match, unmatched reporting |
| filters | theme descendant expansion; proposed vocabulary excluded |

### Integration tests

`tests/fixtures/` holds a synthetic mini-corpus and recorded Claude responses. The corpus is original Carlin-style text written for the tests, not real transcripts. With it, the full pipeline and the MCP tools run offline and give the same result every time. A tiny embedding stub keeps the tests fast. One opt-in test (`-m slow`) uses the real BGE model.

### Retrieval evaluation

`carlin eval` runs a golden set that the user writes in `eval/golden.yaml`:

```yaml
- news: "Airline introduces 'comfort fee' for standard legroom"
  queries: ["companies disguise price increases with pleasant names", "airline fees"]
  expect_bits: ["Euphemisms", "Airline Announcements"]
```

It reports recall@10 for each query and overall, both per retriever and fused. That way, changes to prompts, embeddings, or fusion weights are measured instead of guessed. The Stage 1 target is about 20 golden items with fused recall@10 of at least 0.8.

### Ingest review

After each ingest, `carlin review works <slug>` prints the bit list with titles, passage counts, and theses for a quick human spot-check.

## 11. Configuration and operations

1. `carlinpedia.toml` holds model IDs, the embedding model, the DB path, the RRF k, the dedup threshold, and retriever depth.
2. Ingest needs `ANTHROPIC_API_KEY` or an `ant auth login` profile. The MCP server needs no API key.
3. Migrations are numbered SQL files applied on startup, and `meta.schema_version` tracks which have run.
4. `data/carlin.db` is a single file. Favorites are the only data that can't be rebuilt from the corpus, so `carlin export-favorites` writes them to `data/favorites.json`, keyed by content hash and work slug.

## 12. Decisions made during spec writing

These defaults were not discussed explicitly. Override any of them during review.

| Decision | Choice | Why |
|---|---|---|
| Boundary mechanism | sentence-unit ranges instead of raw character offsets | LLMs count characters poorly; unit IDs give the same verbatim guarantee reliably |
| Enrichment granularity | one request per bit | each passage gets context, with fewer calls and lower cost |
| Embedding model | `BAAI/bge-base-en-v1.5`, local | free, offline, strong for its size, and swappable |
| Timestamps | optional | most available transcripts lack them |
| Package manager | `uv` | fast, has a lockfile, simple scripts |
| Ingest model | `claude-sonnet-5`, configurable | approved in discussion; the corpus is small |
