"""The MCP server Claude Desktop and Claude Code talk to over stdio."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from carlinpedia import library
from carlinpedia.config import Config, load_config, resolve_root
from carlinpedia.db.schema import connect
from carlinpedia.ingest.embed import Embedder, EmbeddingMismatch, SentenceTransformerEmbedder
from carlinpedia.search.hybrid import SearchError, SearchFilters, search

INSTRUCTIONS = """\
Carlinpedia is a private library of George Carlin's stand-up. Use it to find Carlin passages that speak to a news story or situation.
Carlin's words rarely match a news story's words, so search for the underlying idea: phrase some queries as general claims about people or institutions, and some in concrete, Carlin-like language.
Quote passages exactly as returned, with their work, year, and bit. Never paraphrase a passage as if it were a quote. If nothing fits well, say so.
"""

CARLIN_ON = """\
Find what George Carlin would say about this, using the Carlinpedia tools.

{news}

1. Name the human behaviors and institutional patterns at work here, stated generally (for example: "companies rename price increases to hide them").
2. Run 3 to 5 search_passages queries. Phrase some as general theses and some in concrete, Carlin-like words. Call list_themes first if a theme or target filter would clearly help.
3. Call get_passage for context on the most promising results.
4. Present the best 2 to 4 passages verbatim, each cited as work, year, and bit, with one sentence on why it fits. Never alter quoted text. If nothing fits well, say so plainly.
"""


def build_server(conn: sqlite3.Connection, embedder: Embedder, cfg: Config) -> MCPServer:
    server = MCPServer("carlinpedia", instructions=INSTRUCTIONS)

    @server.tool()
    def search_passages(
        query: str,
        themes: list[str] | None = None,
        targets: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        kinds: list[str] | None = None,
        favorites_only: bool = False,
        limit: int = 10,
    ) -> dict:
        """Search Carlin passages by idea and wording. Returns up to `limit` (max 25) verbatim passages with
        citations, thesis, themes, and targets. Filters: theme or target slugs or names from list_themes (a parent
        theme includes its children), a year range, kinds (special, album, book, interview), favorites_only."""
        filters = SearchFilters(
            themes=tuple(themes or ()), targets=tuple(targets or ()), year_from=year_from, year_to=year_to,
            kinds=tuple(kinds or ()), favorites_only=favorites_only,
        )
        try:
            results = search(conn, embedder, query, filters, limit, rrf_k=cfg.rrf_k, depth=cfg.retriever_depth,
                             dedup_threshold=cfg.dedup_threshold)
        except (SearchError, EmbeddingMismatch) as exc:
            raise ToolError(str(exc)) from exc
        return {"count": len(results), "results": results}

    @server.tool()
    def get_passage(passage_id: int, context: int = 1) -> dict:
        """One passage with its full citation, the bit summary, and up to `context` (max 5) neighboring
        passages on each side within the same bit."""
        try:
            return library.get_passage(conn, passage_id, context)
        except library.NotFound as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    def get_bit(bit_id: int) -> dict:
        """A whole bit (routine), passage by passage."""
        try:
            return library.get_bit(conn, bit_id)
        except library.NotFound as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    def list_works() -> dict:
        """Every work in the library with its kind, year, and bit and passage counts."""
        return {"works": library.list_works(conn)}

    @server.tool()
    def list_themes() -> dict:
        """The approved theme hierarchy and target list, with descriptions, for use as search filters."""
        return library.list_themes(conn)

    @server.tool()
    def set_favorite(passage_id: int, favorite: bool = True) -> dict:
        """Mark or unmark a passage as one of the user's favorites."""
        try:
            return library.set_favorite(conn, passage_id, favorite)
        except library.NotFound as exc:
            raise ToolError(str(exc)) from exc

    @server.prompt()
    def carlin_on(news: str) -> str:
        """Find Carlin passages that speak to a news story or situation."""
        return CARLIN_ON.format(news=news)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Carlinpedia MCP server over stdio.")
    parser.add_argument("--root", type=Path, default=None, help="project root containing carlinpedia.toml")
    args = parser.parse_args(argv)
    cfg = load_config(resolve_root(args.root))
    conn = connect(cfg.db_path, embedding_dim=cfg.embedding_dim)
    embedder = SentenceTransformerEmbedder(cfg.embedding_model, cfg.embedding_dim)
    build_server(conn, embedder, cfg).run("stdio")


if __name__ == "__main__":
    main()
