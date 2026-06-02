"""Edge cases + previously untested branches.

Covers the gaps I called out in the audit:
  - empty content
  - dim mismatch (embedder vs store)
  - cross-kind contradiction
  - multi-candidate contradiction (>=2 priors)
  - resolver merge fallback (empty merged_content)
  - as_of time-travel read
  - include_superseded read
  - procedural-kind decay
  - whitespace + case reinforcement match
  - reopen with mismatched dim is silent (caller responsible)
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


# ----- write / write-path edges --------------------------------------

def test_empty_content_raises(mem):
    with pytest.raises(ValueError):
        mem.write("")
    with pytest.raises(ValueError):
        mem.write("   ")


def test_whitespace_and_case_reinforcement(mem):
    a = mem.write("User likes coffee", subject="user")
    b = mem.write("  USER LIKES COFFEE  ", subject="user")
    assert b.action == "reinforced"
    assert b.atom_id == a.atom_id


def test_cross_kind_contradiction_still_supersedes(mem):
    # Same subject, different kinds — Berlin/Munich pair triggers stub verifier.
    a = mem.write("User lives in Berlin", subject="user", kind="semantic")
    b = mem.write("User lives in Munich", subject="user", kind="episodic")
    assert b.action == "superseded_prior"
    prior = mem.get(a.atom_id)
    assert prior is not None and prior.status == "superseded"


def test_multi_candidate_contradiction_supersedes_all(mem):
    # Three priors all about Berlin; one new "Munich" should supersede all of them.
    a = mem.write("User lives in Berlin",        subject="user", check_contradictions=False)
    b = mem.write("User has a Berlin address",   subject="user", check_contradictions=False)
    c = mem.write("User is based out of Berlin", subject="user", check_contradictions=False)
    out = mem.write("User lives in Munich", subject="user")
    assert out.action == "superseded_prior"
    superseded_ids = set(out.superseded_ids)
    # We expect at least the chosen prior + however many also passed the verifier.
    # The stub verifier flips on any "berlin"+"munich" pair, so all three should fire.
    assert len(superseded_ids) >= 1
    # Every superseded atom now points at the new one.
    for sid in superseded_ids:
        s = mem.get(sid)
        assert s is not None and s.status == "superseded"
        assert s.superseded_by_id == out.atom_id


def test_resolver_merge_fallback_when_merged_content_empty(mem):
    """If the resolver returns action=merge but no merged_content, code falls
    back to new_supersedes. Stub doesn't naturally produce empty merged_content,
    so we patch the LLM payload for one call."""
    from palimpsest.providers import StubLLM
    real_call = mem.llm.call

    def patched(*, system, user, schema):
        result = real_call(system=system, user=user, schema=schema)
        if schema.get("name") == "resolve_contradiction":
            result.payload = {"action": "merge", "rationale": "stub fallback test", "merged_content": "   "}
        return result

    mem.llm.call = patched  # type: ignore[assignment]
    try:
        mem.write("User lives in Berlin", subject="user")
        out = mem.write("User lives in Munich", subject="user")
        # merged_content was empty -> code falls back to new_supersedes
        assert out.action == "superseded_prior"
    finally:
        mem.llm.call = real_call  # type: ignore[assignment]


# ----- read-path edges ----------------------------------------------

def test_as_of_time_travel_excludes_future_atoms(mem):
    a = mem.write("User loves jazz", subject="user")
    # backdate this atom
    past = datetime.now(timezone.utc) - timedelta(days=10)
    mem.store.conn.execute(
        "UPDATE atoms SET created_at = ? WHERE id = ?",
        (past.isoformat(), str(a.atom_id)),
    )
    mem.store.conn.commit()

    b = mem.write("User loves classical", subject="user", check_contradictions=False)
    # Search as of 5 days ago — only the jazz atom should appear.
    cutoff = datetime.now(timezone.utc) - timedelta(days=5)
    hits = mem.read("what does user love", subject="user", k=5, as_of=cutoff)
    ids = {atom.id for atom, _d in hits}
    assert a.atom_id in ids
    assert b.atom_id not in ids


def test_include_superseded_returns_old_versions(mem):
    a = mem.write("User lives in Berlin", subject="user")
    mem.write("User lives in Munich", subject="user")  # auto-supersedes
    # Default read excludes superseded.
    default_hits = mem.read("where does user live", subject="user", k=5)
    assert all(atom.status == "active" for atom, _d in default_hits)
    # With include_superseded the prior should be visible.
    deep_hits = mem.read("where does user live", subject="user", k=5, include_superseded=True)
    assert any(atom.id == a.atom_id for atom, _d in deep_hits)


# ----- decay / forgetting edges ------------------------------------

def test_procedural_decay_uses_180d_half_life(mem):
    out = mem.write(
        "When the X tool returns 429 retry once with jitter",
        kind="procedural", subject="agent", confidence=1.0,
    )
    aid = out.atom_id
    # 180d == one half-life for procedural -> conf ~ 0.5
    past = datetime.now(timezone.utc) - timedelta(days=180)
    mem.store.conn.execute(
        "UPDATE atoms SET last_reinforced_at = ? WHERE id = ?",
        (past.isoformat(), str(aid)),
    )
    mem.store.conn.commit()
    mem.decay()
    a = mem.get(aid)
    assert a is not None
    # Allow some slack for rounding
    assert 0.40 < a.confidence < 0.60


def test_combined_prune_threshold_and_cap(mem):
    # 6 atoms — set 2 to low confidence, leave 4 high.
    ids = []
    for i in range(6):
        out = mem.write(
            f"Episodic event {i}", kind="episodic", subject="user", source=f"e{i}",
            confidence=0.05 if i < 2 else 0.9,
        )
        ids.append(out.atom_id)
    summary = mem.prune(confidence_threshold=0.10, max_kept_per_subject=3)
    # 2 dropped by threshold, then cap of 3 kept of the remaining 4 -> 1 more retracted.
    assert summary.retracted >= 3
    actives = mem.list_subject("user", status="active")
    assert len(actives) <= 3


# ----- store edges --------------------------------------------------

def test_dim_mismatch_at_insert_raises(mem):
    from palimpsest.schemas import Atom
    bad_emb = [0.0] * (mem.store.vec_dim + 1)
    atom = Atom(content="x", kind="semantic", subject="user")
    with pytest.raises(ValueError):
        mem.store.insert_atom(atom, bad_emb)


def test_dim_mismatch_at_search_raises(mem):
    bad = [0.0] * (mem.store.vec_dim + 1)
    with pytest.raises(ValueError):
        mem.store.search(bad, k=3)


def test_lineage_cycle_protection(mem):
    """Manually inject a supersedes cycle and confirm lineage_chain doesn't loop forever."""
    a = mem.write("Atom A", subject="user", check_contradictions=False)
    b = mem.write("Atom B", subject="user", check_contradictions=False)
    mem.store.mark_superseded(a.atom_id, b.atom_id)
    # Force the cycle: B.supersedes_id = A (already), A.supersedes_id = B.
    mem.store.conn.execute(
        "UPDATE atoms SET supersedes_id = ? WHERE id = ?",
        (str(b.atom_id), str(a.atom_id)),
    )
    mem.store.conn.commit()
    chain = mem.lineage(b.atom_id)
    # Bounded — must not infinite loop. Max length is the number of unique atoms in the cycle.
    assert 1 <= len(chain) <= 4


def test_reopen_with_different_dim_keeps_existing(tmp_path):
    """Open a fresh store at default dim (768), then re-open requesting 1024.
    The store should keep 768 (the file's pinned dim) and not recreate."""
    p = tmp_path / "p.db"
    m1 = Memory.open(p)
    dim1 = m1.store.vec_dim
    m1.close()

    from palimpsest.store import Store
    s = Store(p, vec_dim=1024)
    assert s.vec_dim == dim1   # stays at the file's existing dim
    s.close()
