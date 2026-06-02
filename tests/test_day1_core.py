"""Day-1 sanity tests, updated for the v0.0.2 WriteOutcome return shape.

These cover the SDK surface inherited from Day 1:
  * write -> read roundtrip
  * subject extraction (stub heuristic)
  * lineage chain walk (manual mark_superseded)
  * reinforce bumps confidence + count
  * retract removes from default reads
  * stats counts the right buckets
"""

from __future__ import annotations

import os
import tempfile

import pytest

from palimpsest import Memory


@pytest.fixture
def mem():
    with tempfile.TemporaryDirectory() as td:
        m = Memory.open(os.path.join(td, "palimpsest.db"))
        try:
            yield m
        finally:
            m.close()


def test_write_then_read_returns_the_atom(mem):
    out = mem.write("User likes coffee", kind="semantic", subject="user")
    hits = mem.read("what does the user drink", k=3, subject="user")
    assert any(a.id == out.atom_id for a, _d in hits), "newly written atom must be retrievable"


def test_subject_inferred_when_not_provided(mem):
    out = mem.write("The user lives in Berlin")
    atom = mem.get(out.atom_id)
    assert atom is not None
    assert atom.subject == "user"


def test_subject_filter_excludes_other_subjects(mem):
    mem.write("User likes coffee", subject="user")
    mem.write("Alice likes tea",   subject="alice")
    hits_user  = mem.read("preferences", k=10, subject="user")
    hits_alice = mem.read("preferences", k=10, subject="alice")
    assert all(a.subject == "user"  for a, _ in hits_user)
    assert all(a.subject == "alice" for a, _ in hits_alice)
    assert any("Alice" in a.content for a, _ in hits_alice)


def test_reinforce_bumps_confidence_and_count(mem):
    out = mem.write("User lives in Berlin", subject="user", confidence=0.7)
    mem.reinforce(out.atom_id, source="turn_2", confidence=1.0)
    a = mem.get(out.atom_id)
    assert a is not None
    assert a.reinforcement_count == 2
    assert a.confidence > 0.7  # bumped


def test_retract_excludes_from_default_reads(mem):
    out = mem.write("User lives in Berlin", subject="user")
    mem.retract(out.atom_id)
    hits = mem.read("where does the user live", k=5, subject="user")
    assert all(a.id != out.atom_id for a, _ in hits)
    # the row is still there for audit
    a = mem.get(out.atom_id)
    assert a is not None and a.status == "retracted"


def test_lineage_chain_walks_supersedes(mem):
    # Manual supersede path — auto-supersede is covered separately in test_contradiction.py.
    out1 = mem.write("User loves jazz", subject="user")
    out2 = mem.write("User now loves classical", subject="user", check_contradictions=False)
    mem.store.mark_superseded(out1.atom_id, out2.atom_id)
    chain = mem.lineage(out2.atom_id)
    assert [a.id for a in chain] == [out1.atom_id, out2.atom_id]
    older, newer = chain
    assert older.status == "superseded"
    assert older.superseded_by_id == out2.atom_id
    assert newer.supersedes_id == out1.atom_id


def test_stats_count_buckets(mem):
    out1 = mem.write("User likes coffee", subject="user")
    out2 = mem.write("User likes hiking",  subject="user")  # compatible -> no auto-supersede
    mem.retract(out2.atom_id)
    s = mem.stats()
    assert s["atoms_total"] == 2
    assert s["atoms_active"] == 1
    assert s["vec_dim"] >= 64
