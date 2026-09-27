"""Annotate each passage with a thesis, themes, targets, tone, and profanity, one bit per Claude call."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from carlinpedia.llm.client import LLM, LLMError, load_prompt
from carlinpedia.vocab import Vocab, upsert_proposed

ENRICH_PROMPT = "enrich_v1"
PROPOSED_WEIGHT = 0.5

Tone = Literal["rant", "observational", "absurdist", "wordplay", "dark", "reflective"]


class ThemeWeight(BaseModel):
    slug: str
    weight: float


class ProposedTerm(BaseModel):
    name: str
    description: str


class PassageEnrichment(BaseModel):
    ordinal: int
    thesis: str
    themes: list[ThemeWeight]
    proposed_themes: list[ProposedTerm]
    targets: list[str]
    proposed_targets: list[ProposedTerm]
    tone: Tone
    profanity: bool


class BitEnrichment(BaseModel):
    passages: list[PassageEnrichment]


@dataclass(frozen=True)
class PassageRef:
    id: int
    ordinal: int
    text: str


@dataclass(frozen=True)
class BitForEnrichment:
    id: int
    ordinal: int
    title: str
    summary: str
    passages: tuple[PassageRef, ...]


@dataclass
class ApplyReport:
    proposed: list[str] = field(default_factory=list)


def load_bits(conn: sqlite3.Connection, work_id: int) -> list[BitForEnrichment]:
    bits = []
    for b in conn.execute("SELECT id, ordinal, title, summary FROM bit WHERE work_id = ? ORDER BY ordinal", (work_id,)):
        passages = tuple(
            PassageRef(r["id"], r["ordinal"], r["text"])
            for r in conn.execute("SELECT id, ordinal, text FROM passage WHERE bit_id = ? ORDER BY ordinal", (b["id"],))
        )
        bits.append(BitForEnrichment(b["id"], b["ordinal"], b["title"], b["summary"], passages))
    return bits


def render_vocab(vocab: Vocab) -> str:
    theme_lines = [f"- {t.slug}: {t.name}. {t.description}" for t in vocab.themes.values()]
    target_lines = [f"- {t.slug}: {t.name}. {t.description}" for t in vocab.targets.values()]
    return "Approved themes:\n" + "\n".join(theme_lines) + "\n\nApproved targets:\n" + "\n".join(target_lines)


def render_bit(bit: BitForEnrichment) -> str:
    parts = [f"Bit: {bit.title}", f"Summary: {bit.summary}", ""]
    for p in bit.passages:
        parts.append(f'<passage ordinal="{p.ordinal}">\n{p.text}\n</passage>')
    return "\n".join(parts)


def _ordinal_problems(bit: BitForEnrichment, result: BitEnrichment) -> list[str]:
    expected = [p.ordinal for p in bit.passages]
    got = [p.ordinal for p in result.passages]
    if sorted(got) == expected:
        return []
    return [f"expected exactly one entry for each passage ordinal {expected}, got {got}"]


def enrich_bit(llm: LLM, bit: BitForEnrichment, vocab: Vocab, *, key: str) -> BitEnrichment:
    # The vocabulary goes in the system prompt so it is cached across every bit.
    system = load_prompt(ENRICH_PROMPT) + "\n\n" + render_vocab(vocab)
    user = render_bit(bit)
    result = llm.parse(prompt_name=ENRICH_PROMPT, key=key, system=system, user=user, output_model=BitEnrichment)
    problems = _ordinal_problems(bit, result)
    if problems:
        retry = f"{user}\n\nYour previous answer had this problem: {problems[0]}. Return one entry per passage."
        result = llm.parse(prompt_name=ENRICH_PROMPT, key=key, system=system, user=retry, output_model=BitEnrichment)
        problems = _ordinal_problems(bit, result)
        if problems:
            raise LLMError(f"{ENRICH_PROMPT} [{key}]: {problems[0]}")
    return result


def apply_enrichment(
    conn: sqlite3.Connection, bit: BitForEnrichment, result: BitEnrichment, vocab: Vocab, *,
    model: str, prompt_version: str = ENRICH_PROMPT,
) -> ApplyReport:
    """Write LLM fields and links. Never touches `favorite`. Unknown slugs become proposals."""
    report = ApplyReport()
    by_ordinal = {p.ordinal: p for p in bit.passages}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for entry in result.passages:
        passage = by_ordinal[entry.ordinal]
        conn.execute(
            "UPDATE passage SET thesis = ?, tone = ?, profanity = ?, enriched_model = ?, "
            "enriched_prompt_version = ?, enriched_at = ? WHERE id = ?",
            (entry.thesis.strip(), entry.tone, int(entry.profanity), model, prompt_version, now, passage.id),
        )
        conn.execute("DELETE FROM passage_theme WHERE passage_id = ?", (passage.id,))
        conn.execute("DELETE FROM passage_target WHERE passage_id = ?", (passage.id,))

        theme_links: dict[int, float] = {}
        for tw in entry.themes:
            term = vocab.themes.get(tw.slug)
            if term is not None:
                theme_links[term.id] = max(0.0, min(1.0, tw.weight))
            else:
                entry.proposed_themes.append(ProposedTerm(name=tw.slug, description=""))
        for proposal in entry.proposed_themes:
            term_id = upsert_proposed(conn, "theme", proposal.name, proposal.description)
            if term_id is not None:
                theme_links.setdefault(term_id, PROPOSED_WEIGHT)
                report.proposed.append(f"theme: {proposal.name}")
        conn.executemany(
            "INSERT INTO passage_theme (passage_id, theme_id, weight) VALUES (?, ?, ?)",
            [(passage.id, tid, w) for tid, w in theme_links.items()],
        )

        target_ids: set[int] = set()
        for slug in entry.targets:
            term = vocab.targets.get(slug)
            if term is not None:
                target_ids.add(term.id)
            else:
                entry.proposed_targets.append(ProposedTerm(name=slug, description=""))
        for proposal in entry.proposed_targets:
            term_id = upsert_proposed(conn, "target", proposal.name, proposal.description)
            if term_id is not None:
                target_ids.add(term_id)
                report.proposed.append(f"target: {proposal.name}")
        conn.executemany(
            "INSERT INTO passage_target (passage_id, target_id) VALUES (?, ?)",
            [(passage.id, tid) for tid in sorted(target_ids)],
        )
    return report
