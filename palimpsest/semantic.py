"""Semantic-store-specific behaviour: consolidation.

Over time, multiple atoms drift toward saying the same thing in slightly
different words. ("User likes coffee", "the user prefers coffee", "User
enjoys drinking coffee.") The semantic store consolidation pass:

  1. Per subject, cluster active semantic atoms by embedding similarity
  2. For each cluster of size >= 2 with intra-cluster cosine sim >= threshold,
     ask the LLM to merge them
  3. If the LLM agrees they're duplicates, write a merged atom and
     supersede the inputs (lineage preserved)

Run periodically (call Memory.consolidate()), not on every write.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from uuid import UUID

from palimpsest.prompts import (
    CONSOLIDATOR_PROMPT_VERSION, CONSOLIDATOR_SCHEMA, CONSOLIDATOR_SYSTEM,
)
from palimpsest.providers import EmbeddingProvider, LLMProvider
from palimpsest.schemas import Atom


if TYPE_CHECKING:
    from palimpsest.store import Store


# Cosine similarity threshold — atoms must be this close to be candidates for
# merge. Conservative; we'd rather under-consolidate than collapse distinct facts.
_MERGE_COS_SIM_THRESHOLD = 0.85
_MIN_CLUSTER_SIZE = 2
_MAX_CLUSTER_SIZE = 6


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _cluster_by_similarity(
    atoms_with_embeddings: list[tuple[Atom, list[float]]],
    *,
    threshold: float,
) -> list[list[tuple[Atom, list[float]]]]:
    """Greedy single-link clustering. O(n^2) — fine for the scale a single
    agent's memory hits inside one subject."""
    clusters: list[list[tuple[Atom, list[float]]]] = []
    for atom, emb in atoms_with_embeddings:
        placed = False
        for c in clusters:
            # link if similar to ANY atom already in the cluster
            if any(_cosine(emb, e) >= threshold for _a, e in c):
                c.append((atom, emb))
                placed = True
                break
        if not placed:
            clusters.append([(atom, emb)])
    return clusters


def consolidate_subject(
    store: "Store",
    *,
    subject: str,
    llm: LLMProvider,
    embedder: EmbeddingProvider,
    threshold: float = _MERGE_COS_SIM_THRESHOLD,
) -> list[UUID]:
    """Run a single consolidation pass over all active SEMANTIC atoms for one
    subject. Returns the ids of newly-created merged atoms.

    Embeddings are recomputed here — the store doesn't expose them by atom_id
    (the vec0 virtual table indexes on rowid, not atom UUID for arbitrary
    extraction). Recompute is cheap on the scale a single subject hits.
    """
    atoms = [a for a in store.list_by_subject(subject) if a.kind == "semantic"]
    if len(atoms) < _MIN_CLUSTER_SIZE:
        return []

    embs = embedder.embed([a.content for a in atoms])
    pairs = list(zip(atoms, embs))

    clusters = _cluster_by_similarity(pairs, threshold=threshold)
    new_ids: list[UUID] = []

    for cluster in clusters:
        if len(cluster) < _MIN_CLUSTER_SIZE:
            continue
        if len(cluster) > _MAX_CLUSTER_SIZE:
            cluster = sorted(cluster, key=lambda p: -p[0].confidence)[:_MAX_CLUSTER_SIZE]

        cluster_atoms = [a for a, _e in cluster]
        prompt_user = "\n\n".join(
            f"Atom {i+1} (confidence {a.confidence:.2f}, reinforced {a.reinforcement_count}x):\n"
            f"  {a.content}"
            for i, a in enumerate(cluster_atoms)
        )
        prompt_user += f"\n\nSubject: {subject}\nKind: semantic"

        result = llm.call(
            system=CONSOLIDATOR_SYSTEM, user=prompt_user, schema=CONSOLIDATOR_SCHEMA,
        )
        payload = result.payload or {}
        if not payload.get("should_merge"):
            continue
        merged_text = (payload.get("merged_content") or "").strip()
        if not merged_text:
            continue

        # Strongest atom in the cluster is the "anchor" — we supersede the
        # weaker ones onto a new merged atom that supersedes the anchor too.
        anchor = max(cluster_atoms, key=lambda a: (a.confidence, a.reinforcement_count))

        merged_atom = Atom(
            content=merged_text,
            kind="semantic",
            subject=subject,
            confidence=min(1.0, max(a.confidence for a in cluster_atoms) + 0.05),
            source=f"consolidate:{CONSOLIDATOR_PROMPT_VERSION}",
        )
        merged_emb = embedder.embed([merged_text])[0]
        store.insert_atom(merged_atom, merged_emb)
        store.mark_superseded(anchor.id, merged_atom.id)
        for a in cluster_atoms:
            if a.id != anchor.id and a.status == "active":
                store.mark_superseded(a.id, merged_atom.id)
        new_ids.append(merged_atom.id)

    return new_ids
