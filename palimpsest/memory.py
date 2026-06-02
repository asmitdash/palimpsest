"""Public Memory API.

Day 2+ surface (contradiction-aware):

  mem.write(content, kind=..., subject=..., source=...,
            check_contradictions=True)        # default True

  mem.read(query, k=..., subject=..., kind=..., as_of=...)
  mem.episodic_window(subject=, start=, end=, limit=)
  mem.episodic_chain(subject=)                 # group by source

  mem.reinforce(atom_id, source=, confidence=, independent_source=True)
  mem.retract(atom_id)

  mem.consolidate(subject=None)                # semantic-store merge pass
  mem.decay(now=None, forget_threshold=0.05)   # confidence decay -> retract
  mem.prune(confidence_threshold=, max_kept_per_subject=)

  mem.get(atom_id) / mem.lineage(atom_id) / mem.list_subject(subject)
  mem.contradictions(limit=)
  mem.stats()

Every contradiction-related call returns a `WriteOutcome` so callers know what
the engine decided. The hot-path stays simple for the no-contradiction case
(zero LLM calls beyond the optional subject-extract).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from palimpsest.contradiction import (
    ContradictionReport,
    check_and_resolve,
    contradiction_pressure_penalty,
    reinforcement_gain,
)
from palimpsest.episodic import chain_by_source, window as episodic_window_fn
from palimpsest.forgetting import DecaySummary, PruneSummary, decay_pass, prune_pass
from palimpsest.providers import (
    EmbeddingProvider, LLMProvider,
    get_embedding_provider, get_llm_provider,
)
from palimpsest.schemas import Atom, AtomKind, AtomStatus, Resolution
from palimpsest.semantic import consolidate_subject
from palimpsest.store import Store
from palimpsest.subject import extract_subject


WriteAction = Literal[
    "inserted",
    "reinforced",
    "rejected_old_wins",
    "superseded_prior",
    "merged",
    "kept_both",
]


@dataclass
class WriteOutcome:
    """Returned by Memory.write so callers can react to contradiction outcomes."""

    action: WriteAction
    atom_id: UUID | None              # the id the caller should reference (new or merged)
    superseded_ids: list[UUID]        # any prior atoms now in 'superseded' status
    contradiction_report: ContradictionReport | None = None


class Memory:
    """Agent memory store. One Memory per namespace (db file).

    A namespace usually maps to one user / one agent / one tenant. Cross-namespace
    contradictions are intentionally NOT detected — different agents can hold
    different beliefs. If you want a shared belief layer, run a higher-level
    arbiter on top.
    """

    def __init__(
        self,
        store: Store,
        *,
        llm: LLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.store = store
        self.llm = llm or get_llm_provider()
        self.embedder = embedder or get_embedding_provider()
        if self.embedder.dimensions != self.store.vec_dim:
            raise ValueError(
                f"embedder dim {self.embedder.dimensions} != store dim {self.store.vec_dim}"
            )

    # ---------- constructors ----------

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        llm: LLMProvider | None = None,
        embedder: EmbeddingProvider | None = None,
        vec_dim: int | None = None,
    ) -> "Memory":
        emb = embedder or get_embedding_provider()
        dim = vec_dim or emb.dimensions
        store = Store(path, vec_dim=dim)
        return cls(store, llm=llm, embedder=emb)

    # ---------- writes ----------

    def write(
        self,
        content: str,
        *,
        kind: AtomKind = "semantic",
        subject: str | None = None,
        source: str | None = None,
        confidence: float = 1.0,
        check_contradictions: bool = True,
    ) -> WriteOutcome:
        """Insert a new atom. By default runs the contradiction engine.

        Outcomes:
          - new content has no near-subject conflict        -> inserted
          - new content matches an existing atom textually  -> reinforced (no insert)
          - prior contradicts AND resolver picks new wins   -> superseded_prior + new inserted
          - prior contradicts AND resolver picks old wins   -> rejected_old_wins (no insert,
                                                               but prior still loses some confidence)
          - prior contradicts AND resolver picks merge      -> merged (third atom written,
                                                               both old + new superseded)
          - prior contradicts AND resolver picks keep_both  -> kept_both (new still inserted)
        """
        if not content.strip():
            raise ValueError("empty content")

        subj = (subject or extract_subject(content, provider=self.llm)).lower().strip()
        embedding = self.embedder.embed([content])[0]

        # Hot-path shortcut: exact-text reinforcement. Saves an LLM call when an
        # agent re-asserts the same fact via a different code path.
        if check_contradictions:
            for existing in self.store.list_by_subject(subj, status="active"):
                if existing.content.strip().lower() == content.strip().lower():
                    new_conf = reinforcement_gain(
                        existing.confidence,
                        incoming_confidence=confidence,
                        independent_source=(existing.source or None) != (source or None),
                    )
                    self.store.record_reinforcement(
                        existing.id, source=source, confidence=confidence,
                        new_confidence=new_conf,
                    )
                    return WriteOutcome(
                        action="reinforced", atom_id=existing.id,
                        superseded_ids=[],
                    )

        # Contradiction engine
        report: ContradictionReport | None = None
        if check_contradictions:
            report = check_and_resolve(
                new_content=content, new_kind=kind, new_subject=subj,
                new_embedding=embedding, store=self.store, llm=self.llm,
            )

        if report is None or not report.has_contradiction or report.resolution is None:
            # Plain insert.
            atom = Atom(
                content=content.strip(), kind=kind, subject=subj,
                source=source, confidence=confidence,
            )
            self.store.insert_atom(atom, embedding)
            return WriteOutcome(action="inserted", atom_id=atom.id, superseded_ids=[],
                                contradiction_report=report)

        # Contradiction confirmed — apply the resolver action.
        return self._apply_resolution(
            report=report,
            content=content, kind=kind, subject=subj,
            source=source, confidence=confidence, embedding=embedding,
        )

    def _apply_resolution(
        self,
        *,
        report: ContradictionReport,
        content: str,
        kind: AtomKind,
        subject: str,
        source: str | None,
        confidence: float,
        embedding: list[float],
    ) -> WriteOutcome:
        resolution: Resolution = report.resolution  # type: ignore[assignment]
        prior = report.chosen_prior  # type: ignore[assignment]
        assert prior is not None  # invariant when has_contradiction

        # Always log the contradiction event (audit trail).
        all_priors = report.contradictions_found

        if resolution.action == "old_wins":
            # New atom is REJECTED. Mark prior as 'contradicted' on each
            # additional candidate? No — we keep the chosen prior 'active' but
            # apply a confidence pressure penalty. Other contradicting priors
            # also get the penalty.
            for m in all_priors:
                penalised = contradiction_pressure_penalty(
                    m.atom.confidence, severity=m.verdict.severity if m.verdict else "low",
                )
                self.store.set_confidence(m.atom.id, penalised)
            for m in all_priors:
                self.store.record_contradiction(
                    new_atom_id=prior.id,        # placeholder — there is no new atom
                    prior_atom_id=m.atom.id,
                    severity=m.verdict.severity if m.verdict else "low",
                    verifier_rationale=m.verdict.rationale if m.verdict else "",
                    resolution_action="old_wins",
                    resolution_rationale=resolution.rationale,
                )
            return WriteOutcome(
                action="rejected_old_wins", atom_id=prior.id, superseded_ids=[],
                contradiction_report=report,
            )

        if resolution.action == "keep_both":
            # Verifier was wrong, in the resolver's view. Insert the new atom
            # without superseding anyone, but log it.
            atom = Atom(
                content=content.strip(), kind=kind, subject=subject,
                source=source, confidence=confidence,
            )
            self.store.insert_atom(atom, embedding)
            for m in all_priors:
                self.store.record_contradiction(
                    new_atom_id=atom.id, prior_atom_id=m.atom.id,
                    severity=m.verdict.severity if m.verdict else "low",
                    verifier_rationale=m.verdict.rationale if m.verdict else "",
                    resolution_action="keep_both",
                    resolution_rationale=resolution.rationale,
                )
            return WriteOutcome(
                action="kept_both", atom_id=atom.id, superseded_ids=[],
                contradiction_report=report,
            )

        if resolution.action == "merge":
            merged_text = (resolution.merged_content or "").strip()
            if not merged_text:
                # Resolver didn't produce merged_content — fall back to new_supersedes.
                resolution = Resolution(
                    action="new_supersedes",
                    rationale=resolution.rationale + " [merge fallback: empty merged_content]",
                )
            else:
                merged_atom = Atom(
                    content=merged_text, kind=kind, subject=subject,
                    source=source, confidence=min(1.0, confidence + 0.05),
                )
                merged_emb = self.embedder.embed([merged_text])[0]
                self.store.insert_atom(merged_atom, merged_emb)

                superseded: list[UUID] = []
                # supersede every contradicting prior onto the merged atom
                for m in all_priors:
                    self.store.mark_superseded(m.atom.id, merged_atom.id)
                    superseded.append(m.atom.id)
                    self.store.record_contradiction(
                        new_atom_id=merged_atom.id, prior_atom_id=m.atom.id,
                        severity=m.verdict.severity if m.verdict else "low",
                        verifier_rationale=m.verdict.rationale if m.verdict else "",
                        resolution_action="merge",
                        resolution_rationale=resolution.rationale,
                        merged_atom_id=merged_atom.id,
                    )
                report.merged_atom = merged_atom
                return WriteOutcome(
                    action="merged", atom_id=merged_atom.id,
                    superseded_ids=superseded, contradiction_report=report,
                )

        # new_supersedes (default)
        atom = Atom(
            content=content.strip(), kind=kind, subject=subject,
            source=source, confidence=confidence,
        )
        self.store.insert_atom(atom, embedding)
        superseded: list[UUID] = []
        for m in all_priors:
            self.store.mark_superseded(m.atom.id, atom.id)
            superseded.append(m.atom.id)
            self.store.record_contradiction(
                new_atom_id=atom.id, prior_atom_id=m.atom.id,
                severity=m.verdict.severity if m.verdict else "low",
                verifier_rationale=m.verdict.rationale if m.verdict else "",
                resolution_action="new_supersedes",
                resolution_rationale=resolution.rationale,
            )
        return WriteOutcome(
            action="superseded_prior", atom_id=atom.id,
            superseded_ids=superseded, contradiction_report=report,
        )

    def reinforce(
        self,
        atom_id: UUID,
        *,
        source: str | None = None,
        confidence: float = 1.0,
        independent_source: bool = True,
    ) -> None:
        """Mark an existing atom as re-asserted. Multi-source independence
        bumps confidence more than a same-source repeat."""
        existing = self.store.get_atom(atom_id)
        if existing is None:
            return
        new_conf = reinforcement_gain(
            existing.confidence,
            incoming_confidence=confidence,
            independent_source=independent_source,
        )
        self.store.record_reinforcement(
            atom_id, source=source, confidence=confidence, new_confidence=new_conf,
        )

    def retract(self, atom_id: UUID) -> None:
        self.store.retract(atom_id)

    # ---------- reads ----------

    def read(
        self,
        query: str,
        *,
        k: int = 8,
        subject: str | None = None,
        kind: AtomKind | None = None,
        include_superseded: bool = False,
        as_of: datetime | None = None,
    ) -> list[tuple[Atom, float]]:
        if not query.strip():
            return []
        q_emb = self.embedder.embed([query], input_type="query")[0]
        statuses: AtomStatus | tuple[AtomStatus, ...]
        if include_superseded:
            statuses = ("active", "superseded")
        else:
            statuses = "active"
        return self.store.search(
            q_emb, k=k, subject=subject, kind=kind, status=statuses, as_of=as_of,
        )

    def episodic_window(
        self,
        *,
        subject: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
    ) -> list[Atom]:
        return episodic_window_fn(self.store, subject=subject, start=start, end=end, limit=limit)

    def episodic_chain(self, *, subject: str | None = None) -> dict[str, list[Atom]]:
        return chain_by_source(self.store, subject=subject)

    def get(self, atom_id: UUID) -> Atom | None:
        return self.store.get_atom(atom_id)

    def lineage(self, atom_id: UUID) -> list[Atom]:
        return self.store.lineage_chain(atom_id)

    def list_subject(self, subject: str, *, status: AtomStatus | None = "active") -> list[Atom]:
        return self.store.list_by_subject(subject, status=status)

    def contradictions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.store.conn.execute(
            """
            SELECT id, new_atom_id, prior_atom_id, severity, verifier_rationale,
                   resolution_action, resolution_rationale, merged_atom_id, created_at
            FROM contradictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- maintenance ----------

    def consolidate(self, *, subject: str | None = None) -> list[UUID]:
        """Run the semantic-store consolidation pass. If `subject` is None,
        runs across every active subject. Returns ids of newly merged atoms."""
        merged: list[UUID] = []
        subjects = [subject] if subject else self.store.all_active_subjects()
        for s in subjects:
            merged.extend(
                consolidate_subject(
                    self.store, subject=s, llm=self.llm, embedder=self.embedder,
                )
            )
        return merged

    def decay(
        self, *, now: datetime | None = None, forget_threshold: float = 0.05,
    ) -> DecaySummary:
        return decay_pass(self.store, now=now, forget_threshold=forget_threshold)

    def prune(
        self,
        *,
        confidence_threshold: float = 0.10,
        max_kept_per_subject: int | None = None,
    ) -> PruneSummary:
        return prune_pass(
            self.store,
            confidence_threshold=confidence_threshold,
            max_kept_per_subject=max_kept_per_subject,
        )

    def stats(self) -> dict:
        return self.store.stats()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
