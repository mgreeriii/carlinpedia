# Carlinpedia

A personal, private library of George Carlin's work, built to find the Carlin passages that speak to a current event.

You paste a headline into Claude Desktop or Claude Code. Claude searches a local index of transcripts from Carlin's specials and albums through an MCP server and returns verbatim passages, cited to the work, year, and bit.

Matching works on ideas rather than keywords. Each passage is enriched with a plain-language thesis, themes, and targets, so a story about a renamed layoff program can find his material on euphemisms even when no words overlap.

## Status

Design stage. The design spec will live in `docs/superpowers/specs/`.

## Privacy

Transcripts and the database contain copyrighted material and are for personal use only. `corpus/` and `data/` are gitignored and must never be committed.
