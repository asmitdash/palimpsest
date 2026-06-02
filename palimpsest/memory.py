"""Public Memory API.

This is what an agent imports:

    from palimpsest import Memory
    mem = Memory.open("agent.db")
    mem.write("User lives in Berlin", kind="semantic")
    hits = mem.read("where does the user live?", k=5)

Day 1 surface (no contradiction logic yet):
  * write(content, kind=..., subject=..., source=...)
  * read(query, k=, subject=, kind=, as_of=)
  * get(atom_id) / lineage(atom_id) / list_subject(subject)
  * stats()

Contradiction-aware writes land Day 2 in palimpsest.contradiction.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID

from palimpsest.providers import (
    EmbeddingProvider, LLMProvider,
    get_embedding_provider, get_llm_provider,
)
from palimpsest.schemas import Atom, AtomKind, AtomStatus
from palimpsest.store import Store
from palimpsest.subject import extract_subject


class Memory:
    """Agent memory store.

    One Memory per namespace (think: one per agent / one per user). The db
    file is the source of truth; opening the same path twice from the same
    process is fine but writes serialise on the SQLite connection.
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
        """Open or create a memory store at `path`.

        `vec_dim` defaults to the embedding provider's dimensions. Once a db
        file exists, its dim is fixed; pass a matching embedder.
        """
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
    ) -> UUID:
        """Insert a new atom. Day 1 = blind write (no contradiction handling)."""
        if not content.strip():
            raise ValueError("empty content")
        subj = (subject or extract_subject(content, provider=self.llm)).lower().strip()
        embedding = self.embedder.embed([content])[0]
        atom = Atom(
            content=content.strip(),
            kind=kind,
            subject=subj,
            source=source,
            confidence=confidence,
        )
        self.store.insert_atom(atom, embedding)
        return atom.id

    def reinforce(
        self, atom_id: UUID, *, source: str | None = None, confidence: float = 1.0,
    ) -> None:
        """Mark that an existing atom was re-asserted. Bumps confidence + count."""
        self.store.record_reinforcement(atom_id, source=source, confidence=confidence)

    def retract(self, atom_id: UUID) -> None:
        """Hard withdraw — atom is no longer believed and no longer retrieved."""
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
        """Vector retrieval over active atoms. Returns (atom, distance)."""
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

    def get(self, atom_id: UUID) -> Atom | None:
        return self.store.get_atom(atom_id)

    def lineage(self, atom_id: UUID) -> list[Atom]:
        """Walk the supersedes chain — oldest first, this atom last."""
        return self.store.lineage_chain(atom_id)

    def list_subject(self, subject: str, *, status: AtomStatus | None = "active") -> list[Atom]:
        return self.store.list_by_subject(subject, status=status)

    def stats(self) -> dict:
        return self.store.stats()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
