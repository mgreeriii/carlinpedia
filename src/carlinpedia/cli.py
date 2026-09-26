"""The `carlin` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from carlinpedia import library
from carlinpedia.config import Config, ConfigError, load_config, resolve_root
from carlinpedia.db.schema import connect
from carlinpedia.ingest.embed import Embedder, EmbeddingMismatch
from carlinpedia.ingest.favorites import export_favorites, list_orphaned
from carlinpedia.ingest.manifest import ManifestError, discover_slugs, read_transcript, register_work
from carlinpedia.ingest.normalize import normalize
from carlinpedia.ingest.pipeline import STEPS, run_ingest
from carlinpedia.ingest.segment import SEGMENT_PROMPT, render_units
from carlinpedia.llm.client import LLM, load_prompt
from carlinpedia.search.hybrid import SearchError, SearchFilters, search
from carlinpedia.vocab import VocabError, list_proposed, set_status

app = typer.Typer(help="Carlinpedia: a personal George Carlin library.", no_args_is_help=True)
review_app = typer.Typer(help="Inspect ingest results and approve vocabulary.", no_args_is_help=True)
app.add_typer(review_app, name="review")

_state: dict = {"root": None}


def make_llm(cfg: Config) -> LLM:
    from carlinpedia.llm.client import AnthropicLLM

    return AnthropicLLM(cfg.ingest_model)


def make_embedder(cfg: Config) -> Embedder:
    from carlinpedia.ingest.embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(cfg.embedding_model, cfg.embedding_dim)


def _config() -> Config:
    try:
        return load_config(resolve_root(_state["root"]))
    except ConfigError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(2) from exc


def _connect(cfg: Config):
    return connect(cfg.db_path, embedding_dim=cfg.embedding_dim)


@app.callback()
def main(root: Optional[Path] = typer.Option(None, help="Project root containing carlinpedia.toml.")):
    _state["root"] = root


@app.command()
def ingest(
    slugs: Optional[list[str]] = typer.Argument(None, help="Works to ingest (default: every work in corpus/)."),
    force: bool = typer.Option(False, "--force", help="Re-run every step, even if inputs are unchanged."),
    from_step: Optional[str] = typer.Option(None, "--from", help=f"Re-run this step and all later ones: {', '.join(STEPS)}."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate token use and cost without calling the model."),
):
    """Ingest transcripts into the library."""
    cfg = _config()
    if from_step is not None and from_step not in STEPS:
        typer.secho(f"--from must be one of: {', '.join(STEPS)}", fg="red", err=True)
        raise typer.Exit(2)
    conn = _connect(cfg)
    targets = slugs or discover_slugs(cfg.corpus_dir)
    if not targets:
        typer.echo(f"No works found in {cfg.corpus_dir}. Each work needs a folder with manifest.yaml.")
        raise typer.Exit(1)
    if dry_run:
        _dry_run(cfg, conn, targets)
        return
    report = run_ingest(conn, cfg, make_llm(cfg), make_embedder(cfg), targets, "normalize" if force else from_step)
    if report.vectors_reset:
        typer.secho("Embedding model changed: all vectors were reset. Run `carlin ingest` for every work.", fg="yellow")
    failed = False
    for work in report.works:
        color = {"ok": "green", "skipped": None, "failed": "red", "needs_review": "yellow"}[work.status]
        detail = ", ".join(work.steps_run) if work.steps_run else "nothing to do"
        typer.secho(f"{work.slug}: {work.status} ({detail})", fg=color)
        if work.error:
            typer.echo(f"  {work.error}")
            failed = True
        if work.unmatched_favorites:
            typer.secho(f"  {work.unmatched_favorites} favorite(s) could not be carried over; see `carlin review favorites`",
                        fg="yellow")
        if work.proposed:
            typer.echo(f"  {len(work.proposed)} vocabulary proposal(s); see `carlin review vocab`")
    raise typer.Exit(1 if failed else 0)


def _dry_run(cfg: Config, conn, targets: list[str]) -> None:
    llm = make_llm(cfg)
    system = load_prompt(SEGMENT_PROMPT)
    total_in = total_out = 0
    for slug in targets:
        try:
            reg = register_work(conn, cfg.corpus_dir, slug)
        except ManifestError as exc:
            typer.secho(f"{slug}: {exc}", fg="red")
            continue
        doc = normalize(read_transcript(reg.source_path), reg.manifest.source.has_timestamps,
                        reg.manifest.source.strip_patterns)
        tokens = llm.count_tokens(system=system, user=render_units(doc))
        # Rough model: segmentation reads the transcript once, enrichment reads it again plus a cached
        # vocabulary prompt per bit; output is about one transcript's worth including thinking.
        est_in, est_out = 2 * tokens + 15_000, tokens + 4_000
        total_in, total_out = total_in + est_in, total_out + est_out
        typer.echo(f"{slug}: {tokens:,} transcript tokens, about {est_in:,} in / {est_out:,} out")
    cost = total_in / 1e6 * cfg.price_in_per_mtok + total_out / 1e6 * cfg.price_out_per_mtok
    typer.echo(f"Estimated total: {total_in:,} input and {total_out:,} output tokens, about ${cost:.2f}")


@review_app.command("works")
def review_works(slug: Optional[str] = typer.Argument(None)):
    """Show each work's pipeline status, or one work's bits and theses."""
    cfg = _config()
    conn = _connect(cfg)
    if slug is None:
        rows = conn.execute(
            "SELECT w.slug, s.step, s.status, s.error FROM work w LEFT JOIN pipeline_step s ON s.work_id = w.id "
            "ORDER BY w.slug, CASE s.step WHEN 'normalize' THEN 0 WHEN 'segment' THEN 1 WHEN 'enrich' THEN 2 ELSE 3 END"
        ).fetchall()
        for r in rows:
            typer.echo(f"{r['slug']:32} {r['step'] or '-':10} {r['status'] or '-'}")
            if r["error"]:
                typer.echo(f"    {r['error'].splitlines()[0]}")
        return
    work = conn.execute("SELECT id FROM work WHERE slug = ?", (slug,)).fetchone()
    if work is None:
        typer.secho(f"No work named {slug}", fg="red", err=True)
        raise typer.Exit(1)
    for bit in conn.execute("SELECT id, ordinal, title FROM bit WHERE work_id = ? ORDER BY ordinal", (work["id"],)):
        typer.secho(f"[{bit['ordinal']}] {bit['title']}", bold=True)
        for p in conn.execute("SELECT id, thesis, favorite FROM passage WHERE bit_id = ? ORDER BY ordinal", (bit["id"],)):
            star = "*" if p["favorite"] else " "
            typer.echo(f"  {star} {p['id']:>6}  {p['thesis'] or '(not enriched)'}")


@review_app.command("vocab")
def review_vocab(
    approve: list[str] = typer.Option([], "--approve", help="Slug of a proposed term to approve."),
    reject: list[str] = typer.Option([], "--reject", help="Slug of a proposed term to reject."),
):
    """List proposed themes and targets, or approve and reject them."""
    cfg = _config()
    conn = _connect(cfg)
    try:
        for slug in approve:
            typer.echo(f"approved {set_status(conn, slug, 'approved')} {slug}")
        for slug in reject:
            typer.echo(f"rejected {set_status(conn, slug, 'rejected')} {slug}")
    except VocabError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    if approve or reject:
        typer.echo("Run `carlin ingest --from enrich` so enrichment can use the updated vocabulary.")
        return
    proposals = list_proposed(conn)
    if not proposals:
        typer.echo("No proposals waiting.")
    for p in proposals:
        typer.echo(f"{p['kind']:6} {p['slug']:40} used {p['uses']}x  {p['description']}")


@review_app.command("favorites")
def review_favorites():
    """List favorites that could not be carried over after re-segmentation."""
    conn = _connect(_config())
    orphans = list_orphaned(conn)
    if not orphans:
        typer.echo("No orphaned favorites.")
    for o in orphans:
        typer.echo(f"{o['work']} ({o['created_at']}): {o['text'][:120]}")


@app.command("search")
def search_cmd(
    query: str,
    theme: list[str] = typer.Option([], "--theme"),
    target: list[str] = typer.Option([], "--target"),
    limit: int = typer.Option(5, "--limit"),
):
    """Search from the terminal (for debugging retrieval)."""
    cfg = _config()
    conn = _connect(cfg)
    try:
        results = search(conn, make_embedder(cfg), query, SearchFilters(themes=tuple(theme), targets=tuple(target)),
                         limit, rrf_k=cfg.rrf_k, depth=cfg.retriever_depth, dedup_threshold=cfg.dedup_threshold)
    except (SearchError, EmbeddingMismatch) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    for r in results:
        typer.secho(f"{r['work']['title']} ({r['work']['year']}) · {r['bit']['title']} · #{r['passage_id']}", bold=True)
        typer.echo(f"  {r['text']}")
        typer.echo(f"  thesis: {r['thesis']}")
        if r["also_in"]:
            typer.echo("  also in: " + ", ".join(f"{a['work']} ({a['year']})" for a in r["also_in"]))


@app.command("eval")
def eval_cmd(golden: Path = typer.Option(Path("eval/golden.yaml"), help="Golden set, relative to the project root.")):
    """Report recall@10 per retriever against the golden set."""
    from carlinpedia.evaluation import load_golden, run_eval, summarize

    cfg = _config()
    conn = _connect(cfg)
    path = golden if golden.is_absolute() else cfg.root / golden
    try:
        items = load_golden(path)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    rows = run_eval(conn, make_embedder(cfg), items, cfg)
    for row in rows:
        if row.mode == "fused" and row.missing:
            typer.echo(f"missed {', '.join(row.missing)} for {row.query!r}")
    for mode, recall in summarize(rows).items():
        typer.echo(f"{mode:7} recall@10 = {recall:.2f}")


@app.command("export-favorites")
def export_favorites_cmd(out: Path = typer.Option(Path("data/favorites.json"), help="Output path, relative to the project root.")):
    """Write favorites to JSON, since they are the one thing ingest can't rebuild."""
    cfg = _config()
    path = out if out.is_absolute() else cfg.root / out
    favorites = export_favorites(_connect(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(favorites, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {len(favorites)} favorite(s) to {path}")


@app.command()
def serve():
    """Run the MCP server over stdio (what Claude Desktop and Claude Code launch)."""
    from carlinpedia.mcp_server import build_server

    cfg = _config()
    build_server(_connect(cfg), make_embedder(cfg), cfg).run("stdio")
