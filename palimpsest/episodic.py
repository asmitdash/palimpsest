"""Episodic-store-specific behaviour.

Episodic atoms are 'a thing that happened'. They have time as a first-class
dimension. Two operations matter:

  * window(subject, start, end)  — return atoms ordered by created_at within
                                    a time bound. Critical for replay /
                                    "what happened during X session".
  * chain(source)                 — group atoms by `source` (turn id, session id,
                                    tool-call id) so the agent can replay the
                                    arc of one event.

Episodic atoms also decay faster than semantic ones. The default half-life is
14 days for episodic vs 90 days for semantic. The Memory.decay() pass reads
this distinction.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

from palimpsest.schemas import Atom


if TYPE_CHECKING:
    from palimpsest.store import Store


# Half-life for episodic confidence decay (days).
EPISODIC_HALF_LIFE_DAYS = 14.0
SEMANTIC_HALF_LIFE_DAYS = 90.0
PROCEDURAL_HALF_LIFE_DAYS = 180.0


def half_life_for(kind: str) -> float:
    if kind == "episodic":
        return EPISODIC_HALF_LIFE_DAYS
    if kind == "procedural":
        return PROCEDURAL_HALF_LIFE_DAYS
    return SEMANTIC_HALF_LIFE_DAYS


def window(
    store: "Store",
    *,
    subject: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500,
) -> list[Atom]:
    """Time-window retrieval over episodic atoms."""
    return store.episodic_window(subject, start=start, end=end, limit=limit)


def chain_by_source(
    store: "Store", *, subject: str | None = None,
) -> dict[str, list[Atom]]:
    """Group episodic atoms by their `source` field (turn id / session id /
    tool-call id). Useful for reconstructing "what happened in turn 14"."""
    atoms = store.episodic_window(subject)
    grouped: dict[str, list[Atom]] = defaultdict(list)
    for a in atoms:
        if a.source:
            grouped[a.source].append(a)
    return dict(grouped)
