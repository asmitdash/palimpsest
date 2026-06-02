"""Confidence engine + forgetting layer."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from palimpsest import Memory


@pytest.fixture
def mem():
    with tempfile.TemporaryDirectory() as td:
        m = Memory.open(os.path.join(td, "p.db"))
        try:
            yield m
        finally:
            m.close()


def _backdate_reinforced(mem: Memory, atom_id, when: datetime) -> None:
    mem.store.conn.execute(
        "UPDATE atoms SET last_reinforced_at = ? WHERE id = ?",
        (when.isoformat(), str(atom_id)),
    )
    mem.store.conn.commit()


def test_decay_reduces_old_atom_confidence(mem):
    out = mem.write("User loves jazz", kind="semantic", subject="user", confidence=1.0)
    aid = out.atom_id
    _backdate_reinforced(mem, aid, datetime.now(timezone.utc) - timedelta(days=180))
    summary = mem.decay()
    assert summary.inspected >= 1
    assert summary.decayed >= 1 or summary.retracted >= 1
    fresh = mem.get(aid)
    # Either decayed and still alive, or retracted because below threshold
    assert fresh is not None
    if fresh.status == "active":
        assert fresh.confidence < 0.5  # 180d on a 90d half-life = 0.25


def test_decay_retracts_below_threshold(mem):
    out = mem.write("User likes jogging", kind="episodic", subject="user", confidence=0.2)
    aid = out.atom_id
    # episodic half-life is 14d; 100d is enough to crash confidence
    _backdate_reinforced(mem, aid, datetime.now(timezone.utc) - timedelta(days=100))
    summary = mem.decay(forget_threshold=0.05)
    assert summary.retracted >= 1
    fresh = mem.get(aid)
    assert fresh is not None and fresh.status == "retracted"


def test_reinforcement_independent_source_bumps_more(mem):
    out = mem.write("User uses Linux", subject="user", confidence=0.5)
    aid = out.atom_id
    mem.reinforce(aid, source="dup", independent_source=False)
    after_dup = mem.get(aid).confidence  # type: ignore[union-attr]
    mem.reinforce(aid, source="other", independent_source=True)
    after_indep = mem.get(aid).confidence  # type: ignore[union-attr]
    assert after_indep > after_dup


def test_old_wins_applies_pressure_penalty(mem):
    a1 = mem.write("User lives in Berlin", subject="user")  # confidence=1.0
    before = mem.get(a1.atom_id).confidence  # type: ignore[union-attr]
    mem.write("User lives in Munich [old-wins]", subject="user")
    after = mem.get(a1.atom_id).confidence  # type: ignore[union-attr]
    assert after < before


def test_prune_per_subject_cap(mem):
    for i in range(6):
        mem.write(f"Episodic event {i}", kind="episodic", subject="user", source=f"e{i}")
    summary = mem.prune(confidence_threshold=0.0, max_kept_per_subject=3)
    assert summary.retracted == 3
    actives = mem.list_subject("user")
    assert len(actives) == 3
