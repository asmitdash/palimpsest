"""Core data shapes.

Atom is the unit. Verdict + Resolution are the two LLM-side outputs that
drive contradiction handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


AtomKind = Literal["episodic", "semantic", "procedural"]
"""
- episodic   : a thing that happened ("the user clicked submit at 12:04")
- semantic   : a fact the agent believes ("the user lives in Berlin")
- procedural : a learned how-to ("when the X tool returns 429, retry once with jitter")
"""

AtomStatus = Literal["active", "superseded", "contradicted", "retracted"]
"""
- active       : currently believed
- superseded   : replaced by a newer atom (forgotten, but lineage retained)
- contradicted : a contradiction was detected and is unresolved (NOT retrieved by default)
- retracted    : explicitly withdrawn
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Atom(BaseModel):
    """A single memory atom.

    `subject` is the entity this atom is about. Two atoms only contradict if
    their subjects refer to the same entity — this is the fix for the false
    positives that haunt subject-blind memory stores.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str
    kind: AtomKind
    subject: str = Field(
        description="Canonical name of the entity this atom is about. Lowercase.",
    )
    source: str | None = Field(
        default=None,
        description="Where this came from: tool call id, user turn id, doc id, etc.",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: AtomStatus = "active"
    supersedes_id: UUID | None = None
    superseded_by_id: UUID | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    last_reinforced_at: datetime = Field(default_factory=_utcnow)
    reinforcement_count: int = 1


class Verdict(BaseModel):
    """LLM verifier output for one (new, candidate) pair."""

    contradicts: bool
    severity: Literal["low", "medium", "high"] = "medium"
    rationale: str = ""


class Resolution(BaseModel):
    """LLM resolver output when a contradiction is confirmed.

    `action` decides what we do with the lineage:
      - new_supersedes : the new atom wins; old is marked superseded
      - old_wins       : the new atom is rejected (status='contradicted')
      - merge          : a third atom is written that combines them; both old + new are marked superseded
      - keep_both      : they're not really contradictory after all (verifier was wrong); both stay active
    """

    action: Literal["new_supersedes", "old_wins", "merge", "keep_both"]
    rationale: str = ""
    merged_content: str | None = Field(
        default=None,
        description="Required when action='merge'.",
    )
