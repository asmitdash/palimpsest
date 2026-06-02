"""Automatic forgetting + decay pass.

Two operations:

  decay(now)
    Walk every active atom. For each:
      new_confidence = confidence_decay(confidence, age_in_days, half_life)
    Where age = (now - last_reinforced_at) and half_life depends on kind
    (episodic: 14d, semantic: 90d, procedural: 180d).
    If new_confidence falls below `forget_threshold`, the atom is RETRACTED
    (not deleted). Lineage stays. `as_of` queries can still see it.

  prune(threshold, max_kept_per_subject=None)
    Hard pass: any active atom whose current confidence is below threshold
    gets retracted. Optional cap per subject — keep the top-K by confidence.

The agent runtime calls Memory.decay() periodically (every Nth write, or on
a schedule) and Memory.prune() rarely.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from palimpsest.contradiction import confidence_decay
from palimpsest.episodic import half_life_for


if TYPE_CHECKING:
    from palimpsest.store import Store


@dataclass
class DecaySummary:
    inspected: int = 0
    decayed: int = 0
    retracted: int = 0


@dataclass
class PruneSummary:
    inspected: int = 0
    retracted: int = 0


def decay_pass(
    store: "Store",
    *,
    now: datetime | None = None,
    forget_threshold: float = 0.05,
) -> DecaySummary:
    """Apply time-based decay to every active atom. Atoms below
    `forget_threshold` post-decay are retracted (lineage preserved)."""
    summary = DecaySummary()
    now = now or datetime.now(timezone.utc)
    for atom in store.list_active(limit=10_000):
        summary.inspected += 1
        age_days = max(0.0, (now - atom.last_reinforced_at).total_seconds() / 86400.0)
        if age_days < 0.5:
            continue  # too fresh, skip
        new_conf = confidence_decay(
            atom.confidence,
            age_days=age_days,
            half_life_days=half_life_for(atom.kind),
        )
        if new_conf < forget_threshold:
            store.set_confidence(atom.id, new_conf)
            store.retract(atom.id)
            summary.retracted += 1
        elif abs(new_conf - atom.confidence) > 0.001:
            store.set_confidence(atom.id, new_conf)
            summary.decayed += 1
    return summary


def prune_pass(
    store: "Store",
    *,
    confidence_threshold: float = 0.10,
    max_kept_per_subject: int | None = None,
) -> PruneSummary:
    """Forget atoms by confidence. Optional per-subject cap retains the top-K
    by current confidence."""
    summary = PruneSummary()

    # threshold pass
    for atom in store.list_active(limit=10_000):
        summary.inspected += 1
        if atom.confidence < confidence_threshold:
            store.retract(atom.id)
            summary.retracted += 1

    # per-subject cap pass
    if max_kept_per_subject is not None and max_kept_per_subject > 0:
        for subject in store.all_active_subjects():
            actives = store.list_by_subject(subject, status="active")
            actives_sorted = sorted(
                actives,
                key=lambda a: (-a.confidence, -a.reinforcement_count, a.last_reinforced_at),
            )
            for stale in actives_sorted[max_kept_per_subject:]:
                store.retract(stale.id)
                summary.retracted += 1

    return summary
