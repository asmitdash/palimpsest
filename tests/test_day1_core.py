"""Day-1 sanity tests.

These cover the surface that exists before the contradiction layer lands:
  * write -> read roundtrip
  * subject extraction (stub heuristic)
  * lineage chain walk
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
    aid = mem.write("User likes coffee", kind="semantic", subject="user")
    hits = mem.read("what does the user drink", k=3, subject="user")
    assert any(a.id == aid for a, _d in hits), "newly written atom must be retrievable"


def test_subject_inferred_when_not_provided(mem):
    aid = mem.write("The user lives in Berlin")
    atom = mem.get(aid)
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
    aid = mem.write("User lives in Berlin", subject="user", confidence=0.7)
    mem.reinforce(aid, source="turn_2", confidence=1.0)
    a = mem.get(aid)
    assert a is not None
    assert a.reinforcement_count == 2
    assert a.confidence > 0.7  # bumped


def test_retract_excludes_from_default_reads(mem):
    aid = mem.write("User lives in Berlin", subject="user")
    mem.retract(aid)
    hits = mem.read("where does the user live", k=5, subject="user")
    assert all(a.id != aid for a, _ in hits)
    # but the row is still there for audit
    a = mem.get(aid)
    assert a is not None and a.status == "retracted"


def test_lineage_chain_walks_supersedes(mem):
    # Day-1 doesn't auto-supersede; test the manual lineage path through the store.
    a1 = mem.write("User lives in Berlin", subject="user")
    a2 = mem.write("User lives in Munich", subject="user")
    mem.store.mark_superseded(a1, a2)
    chain = mem.lineage(a2)
    assert [a.id for a in chain] == [a1, a2]
    older, newer = chain
    assert older.status == "superseded"
    assert older.superseded_by_id == a2
    assert newer.supersedes_id == a1


def test_stats_count_buckets(mem):
    a1 = mem.write("User likes coffee", subject="user")
    a2 = mem.write("User dislikes tea", subject="user")
    mem.retract(a2)
    s = mem.stats()
    assert s["atoms_total"] == 2
    assert s["atoms_active"] == 1
    assert s["vec_dim"] >= 64  # stub uses 768 by default
