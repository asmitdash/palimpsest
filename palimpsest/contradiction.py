"""Contradiction detection engine + memory verifier + resolver.

Hot path on Memory.write():
  1. Embed new atom + extract subject (already done by caller)
  2. Retrieve top-K active atoms with same subject + similar embedding
  3. For each candidate above similarity floor, run the VERIFIER (cheap LLM call)
  4. If any verdict says contradicts=True, run the RESOLVER (one more LLM call)
  5. Apply the resolver's action: lineage update + status changes + log row

The whole engine is one entry point — `check_and_resolve(...)` — that returns a
ContradictionReport. Memory.write() consumes the report to decide what to write.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from palimpsest.prompts import (
    RESOLVER_SCHEMA, RESOLVER_SYSTEM,
    VERIFIER_SCHEMA, VERIFIER_SYSTEM,
)
from palimpsest.providers import LLMProvider
from palimpsest.schemas import Atom, AtomKind, Resolution, Verdict


if TYPE_CHECKING:
    from palimpsest.store import Store


# Distance floor: sqlite-vec returns L2 distance for normalised vectors. The
# stub embedder is L2-normalised (cosine ~ dot product), so distance < ~1.4
# corresponds to non-orthogonal candidates. Real embedders are also normalised.
# We pre-filter cheaply on distance, then the LLM verifier flips precision high.
_SIMILARITY_DISTANCE_MAX = 1.0
_TOP_K_CANDIDATES = 6


@dataclass
class CandidateMatch:
    atom: Atom
    distance: float
    verdict: Verdict | None = None


@dataclass
class ContradictionReport:
    """What the engine decided. Memory.write() acts on this."""

    candidates_inspected: int = 0
    contradictions_found: list[CandidateMatch] = field(default_factory=list)
    resolution: Resolution | None = None
    chosen_prior: Atom | None = None  # the prior that we resolved against
    merged_atom: Atom | None = None   # only set when action='merge'

    @property
    def has_contradiction(self) -> bool:
        return bool(self.contradictions_found)


def check_and_resolve(
    *,
    new_content: str,
    new_kind: AtomKind,
    new_subject: str,
    new_embedding: list[float],
    store: "Store",
    llm: LLMProvider,
) -> ContradictionReport:
    """Run the full verifier + resolver loop for a candidate write.

    Doesn't mutate state — purely diagnostic. The caller (Memory.write) reads
    the report and decides what to insert / supersede / log.
    """
    report = ContradictionReport()

    # Step 1: retrieve same-subject candidates
    hits = store.search(
        new_embedding,
        k=_TOP_K_CANDIDATES,
        subject=new_subject,
        kind=None,            # cross-kind contradictions are valid (semantic vs episodic)
        status="active",
    )
    report.candidates_inspected = len(hits)

    if not hits:
        return report

    # Step 2: verifier loop. Only run on candidates within the similarity floor.
    for atom, distance in hits:
        if distance > _SIMILARITY_DISTANCE_MAX:
            continue
        verdict = _run_verifier(
            llm,
            new_content=new_content, new_kind=new_kind,
            prior=atom,
        )
        if verdict.contradicts and verdict.severity != "low":
            report.contradictions_found.append(
                CandidateMatch(atom=atom, distance=distance, verdict=verdict)
            )

    if not report.has_contradiction:
        return report

    # Step 3: resolver — pick the most-confident prior to resolve against.
    # (If multiple priors contradict the new atom, the strongest one is the
    # natural anchor; we then handle the others by chaining the same action.)
    primary = max(
        report.contradictions_found,
        key=lambda m: (m.atom.confidence, m.atom.reinforcement_count, -m.distance),
    )
    report.chosen_prior = primary.atom
    report.resolution = _run_resolver(
        llm,
        prior=primary.atom,
        new_content=new_content,
        new_kind=new_kind,
        verdict=primary.verdict,  # type: ignore[arg-type]
    )
    return report


def _run_verifier(
    llm: LLMProvider, *, new_content: str, new_kind: AtomKind, prior: Atom,
) -> Verdict:
    user = (
        f"PRIOR atom (kind={prior.kind}, subject={prior.subject}, "
        f"reinforced={prior.reinforcement_count}x, confidence={prior.confidence:.2f}):\n"
        f"  {prior.content}\n\n"
        f"NEW atom (kind={new_kind}, subject={prior.subject}):\n"
        f"  {new_content}\n\n"
        "Does the NEW atom contradict the PRIOR atom about this subject?"
    )
    res = llm.call(system=VERIFIER_SYSTEM, user=user, schema=VERIFIER_SCHEMA)
    return Verdict.model_validate(res.payload)


def _run_resolver(
    llm: LLMProvider, *, prior: Atom, new_content: str, new_kind: AtomKind, verdict: Verdict,
) -> Resolution:
    user = (
        f"PRIOR atom:\n"
        f"  content: {prior.content}\n"
        f"  created: {prior.created_at.isoformat()}\n"
        f"  reinforced: {prior.reinforcement_count}x\n"
        f"  confidence: {prior.confidence:.2f}\n\n"
        f"NEW atom:\n"
        f"  content: {new_content}\n"
        f"  kind: {new_kind}\n\n"
        f"Verifier severity: {verdict.severity}\n"
        f"Verifier rationale: {verdict.rationale}\n\n"
        "Pick the resolution action."
    )
    res = llm.call(system=RESOLVER_SYSTEM, user=user, schema=RESOLVER_SCHEMA)
    return Resolution.model_validate(res.payload)


# ----- Confidence dynamics -----------------------------------------------

# All confidence math lives here so policy changes are one diff.

def confidence_decay(
    base_confidence: float,
    *,
    age_days: float,
    half_life_days: float = 90.0,
) -> float:
    """Time-based decay. Half-life-of-90-days is the default — tunable per
    instance later. Atom that hasn't been reinforced in 90d falls to half its
    confidence; in 180d to a quarter. Episodic atoms get a shorter half-life
    (set by the caller).
    """
    if age_days <= 0:
        return base_confidence
    return base_confidence * math.pow(0.5, age_days / max(half_life_days, 1.0))


def contradiction_pressure_penalty(
    confidence: float,
    *,
    severity: str,
) -> float:
    """When an atom is contradicted but the resolver kept it (old_wins or
    keep_both), it still loses some confidence — the new evidence existed."""
    sev_to_drop = {"low": 0.02, "medium": 0.07, "high": 0.15}
    drop = sev_to_drop.get(severity, 0.05)
    return max(0.0, confidence - drop)


def reinforcement_gain(
    confidence: float,
    *,
    incoming_confidence: float,
    independent_source: bool,
) -> float:
    """Independent re-assertion adds more weight than a same-source repeat."""
    base = 0.05 * incoming_confidence
    bonus = 0.05 if independent_source else 0.0
    return min(1.0, confidence + base + bonus)
