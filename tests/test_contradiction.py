"""Contradiction engine + resolver tests."""

from __future__ import annotations

import os
import tempfile

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


def test_contradiction_triggers_supersede(mem):
    a1 = mem.write("User lives in Berlin", subject="user")
    out = mem.write("User lives in Munich", subject="user")
    assert out.action == "superseded_prior"
    assert out.atom_id is not None
    assert out.atom_id != a1.atom_id
    assert a1.atom_id in out.superseded_ids
    # the prior is now in superseded status
    a1_now = mem.get(a1.atom_id)  # type: ignore[arg-type]
    assert a1_now is not None and a1_now.status == "superseded"
    assert a1_now.superseded_by_id == out.atom_id


def test_no_contradiction_when_subjects_differ(mem):
    out_alice = mem.write("Alice likes coffee", subject="alice")
    out_bob   = mem.write("Bob dislikes coffee", subject="bob")
    # Even though stub_verify would fire on likes/dislikes, the engine never
    # gets there because subject filter excludes cross-subject candidates.
    assert out_alice.action == "inserted"
    assert out_bob.action   == "inserted"
    assert mem.get(out_alice.atom_id).status == "active"  # type: ignore[union-attr]
    assert mem.get(out_bob.atom_id).status   == "active"  # type: ignore[union-attr]


def test_compatible_facts_dont_supersede(mem):
    # Stub verifier only fires on specific antonym pairs (likes/dislikes,
    # berlin/munich). "Coffee" + "tea" are compatible.
    out_a = mem.write("User likes coffee", subject="user")
    out_b = mem.write("User likes tea",    subject="user")
    assert out_a.action == "inserted"
    assert out_b.action == "inserted"
    assert mem.get(out_a.atom_id).status == "active"  # type: ignore[union-attr]
    assert mem.get(out_b.atom_id).status == "active"  # type: ignore[union-attr]


def test_exact_text_reinforcement(mem):
    a1 = mem.write("User lives in Berlin", subject="user")
    out = mem.write("User lives in Berlin", subject="user", source="independent")
    assert out.action == "reinforced"
    assert out.atom_id == a1.atom_id
    fresh = mem.get(a1.atom_id)  # type: ignore[arg-type]
    assert fresh is not None
    assert fresh.reinforcement_count == 2
    assert fresh.confidence > 1.0 - 1e-9 or fresh.confidence > 0.99  # capped at 1


def test_resolver_merge_path_writes_third_atom(mem):
    a1 = mem.write("User lives in Berlin", subject="user")
    out = mem.write("User lives in Munich [merge]", subject="user")
    assert out.action == "merged"
    assert out.atom_id is not None
    merged = mem.get(out.atom_id)
    assert merged is not None
    assert merged.content.startswith("User has lived in both Berlin and Munich")
    # both prior and the new "input" should be linked via supersedes
    a1_now = mem.get(a1.atom_id)  # type: ignore[arg-type]
    assert a1_now is not None and a1_now.status == "superseded"
    assert a1_now.superseded_by_id == merged.id


def test_resolver_old_wins_path_rejects_new(mem):
    a1 = mem.write("User lives in Berlin", subject="user")
    out = mem.write("User lives in Munich [old-wins]", subject="user")
    assert out.action == "rejected_old_wins"
    # the prior atom is still active
    a1_now = mem.get(a1.atom_id)  # type: ignore[arg-type]
    assert a1_now is not None and a1_now.status == "active"
    # but its confidence took a pressure hit
    assert a1_now.confidence < 1.0
    # no new atom inserted
    assert mem.stats()["atoms_total"] == 1


def test_resolver_keep_both_inserts_new_without_supersede(mem):
    a1 = mem.write("User lives in Berlin", subject="user")
    out = mem.write("User lives in Munich [keep-both]", subject="user")
    assert out.action == "kept_both"
    a1_now = mem.get(a1.atom_id)  # type: ignore[arg-type]
    new_now = mem.get(out.atom_id)  # type: ignore[arg-type]
    assert a1_now is not None and a1_now.status == "active"
    assert new_now is not None and new_now.status == "active"


def test_contradiction_log_records_event(mem):
    mem.write("User likes coffee",   subject="user")
    mem.write("User dislikes coffee", subject="user")
    log = mem.contradictions(limit=10)
    assert len(log) == 1
    row = log[0]
    assert row["resolution_action"] in ("new_supersedes", "old_wins", "merge", "keep_both")
    assert row["severity"] in ("low", "medium", "high")
