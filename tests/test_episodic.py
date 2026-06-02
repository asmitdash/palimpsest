"""Episodic store: time-window retrieval + source chaining."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from uuid import uuid4

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


def _backdate(mem: Memory, atom_id, when: datetime) -> None:
    mem.store.conn.execute(
        "UPDATE atoms SET created_at = ?, last_reinforced_at = ? WHERE id = ?",
        (when.isoformat(), when.isoformat(), str(atom_id)),
    )
    mem.store.conn.commit()


def test_episodic_window_filters_by_time(mem):
    now = datetime.now(timezone.utc)
    a_old = mem.write("User clicked submit (old)", kind="episodic", subject="user", source="t1").atom_id
    a_mid = mem.write("User opened settings",      kind="episodic", subject="user", source="t2").atom_id
    a_new = mem.write("User logged out",           kind="episodic", subject="user", source="t3").atom_id

    _backdate(mem, a_old, now - timedelta(days=10))
    _backdate(mem, a_mid, now - timedelta(days=3))
    # a_new stays at "now"

    last_week = mem.episodic_window(subject="user", start=now - timedelta(days=7), end=now + timedelta(seconds=1))
    ids = {a.id for a in last_week}
    assert a_old not in ids
    assert a_mid in ids
    assert a_new in ids


def test_episodic_chain_groups_by_source(mem):
    mem.write("Step 1", kind="episodic", subject="user", source="turn_1")
    mem.write("Step 2", kind="episodic", subject="user", source="turn_1")
    mem.write("Step 3", kind="episodic", subject="user", source="turn_2")
    chain = mem.episodic_chain(subject="user")
    assert "turn_1" in chain and len(chain["turn_1"]) == 2
    assert "turn_2" in chain and len(chain["turn_2"]) == 1


def test_episodic_kind_does_not_break_subject_contradiction(mem):
    """Episodic atoms about the same subject can still contradict if their
    content does. The engine ignores kind for contradiction detection."""
    mem.write("User clicked OK",     kind="episodic", subject="user")
    out = mem.write("User likes coffee", kind="episodic", subject="user")
    # No antonym pair so no contradiction expected.
    assert out.action == "inserted"
