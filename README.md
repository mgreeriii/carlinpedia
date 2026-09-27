# Carlinpedia

A personal, private library of George Carlin's work, built to find the Carlin passages that speak to a current event.

You paste a headline into Claude Desktop or Claude Code. Claude searches a local index of transcripts from Carlin's specials and albums through an MCP server and returns verbatim passages, cited to the work, year, and bit.

Matching works on ideas rather than keywords. Each passage is enriched with a plain-language thesis, themes, and targets, so a story about a renamed layoff program can find his material on euphemisms even when no words overlap.

The design is in `docs/superpowers/specs/2026-09-26-carlinpedia-design.md`.

## Privacy

Transcripts and the database contain copyrighted material and are for personal use only. `corpus/` and `data/` are gitignored and must never be committed.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an Anthropic API key (or an `ant auth login` profile) for ingest.

```bash
uv sync
uv run pytest
```

The first real ingest or search downloads the `BAAI/bge-base-en-v1.5` embedding model, about 400 MB.

## Add a work

Create a folder under `corpus/` with the transcript and a manifest:

```
corpus/doin-it-again-1990/
  manifest.yaml
  transcript.txt
```

```yaml
title: "Doin' It Again"
kind: special            # special, album, book, or interview
recorded_on: 1990-02-17
released_on: 1990-06-23
source:
  file: transcript.txt
  provenance: "Where the transcript came from"
  quality: unverified    # verified, unverified, or auto_transcribed
  has_timestamps: false
  strip_patterns: []     # regexes for site boilerplate lines to remove
```

## Ingest

```bash
uv run carlin ingest --dry-run      # estimate tokens and cost first
uv run carlin ingest                # every work; unchanged steps are skipped
uv run carlin ingest doin-it-again-1990 --from enrich   # redo enrichment and indexing for one work
uv run carlin review works doin-it-again-1990           # spot-check bits and theses
uv run carlin review vocab                              # approve or reject proposed themes
```

Before the first ingest, edit `vocab/themes.yaml` and `vocab/targets.yaml` to taste.

## Use it from Claude

Claude Desktop: add this to `~/Library/Application Support/Claude/claude_desktop_config.json`, with your own path, and restart the app.

```json
{
  "mcpServers": {
    "carlinpedia": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/you/projects/personal/carlinpedia", "carlin", "serve"]
    }
  }
}
```

Claude Code:

```bash
claude mcp add carlinpedia -- uv run --directory /Users/you/projects/personal/carlinpedia carlin serve
```

Then ask Claude what Carlin would say about a story, or use the `carlin_on` prompt.

## Measure retrieval

Write about 20 news items and the bits you'd expect back in `eval/golden.yaml`, then run:

```bash
uv run carlin eval
```

## Keep your favorites

Favorites are the only data that ingest can't rebuild. Back them up with:

```bash
uv run carlin export-favorites
```
