"""Semantic store consolidation pass."""

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


def test_consolidate_merges_near_duplicates(mem):
    # Three near-duplicate atoms with overlapping tokens — the stub embedder is
    # bag-of-words hashed, so heavy overlap yields high cosine similarity.
    mem.write("User prefers coffee in the morning", subject="user", check_contradictions=False)
    mem.write("User prefers coffee morning",         subject="user", check_contradictions=False)
    mem.write("User prefers coffee in morning routine", subject="user", check_contradictions=False)

    new_ids = mem.consolidate(subject="user")
    assert len(new_ids) >= 1
    merged = mem.get(new_ids[0])
    assert merged is not None and merged.status == "active"
    # Old atoms should now be superseded
    actives = mem.list_subject("user", status="active")
    assert merged.id in {a.id for a in actives}
    superseded = mem.list_subject("user", status="superseded")
    assert len(superseded) >= 2


def test_consolidate_skips_when_no_cluster(mem):
    mem.write("Alice likes coffee", subject="alice", check_contradictions=False)
    mem.write("Bob likes tea",      subject="bob",   check_contradictions=False)
    new_ids = mem.consolidate()
    assert new_ids == []


def test_consolidate_no_merge_marker_keeps_inputs(mem):
    # Force LLM to refuse merging via the [no-merge] marker.
    mem.write("[no-merge] User prefers coffee morning", subject="user", check_contradictions=False)
    mem.write("[no-merge] User prefers coffee daily",    subject="user", check_contradictions=False)
    new_ids = mem.consolidate(subject="user")
    assert new_ids == []
    assert len(mem.list_subject("user", status="active")) == 2
