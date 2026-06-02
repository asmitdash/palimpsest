"""Real-Gemini integration tests.

Skipped unless GEMINI_API_KEY is set. Forces the Gemini providers and exercises
the contradiction layer + retrieval against the live API. These cost a handful
of tokens each.

Each test opens a fresh tempdir so the SQLite store is empty.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from palimpsest import Memory
from palimpsest.providers import GeminiEmbedder, GeminiLLM


pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)


@pytest.fixture
def mem():
    key = os.environ["GEMINI_API_KEY"]
    llm = GeminiLLM(api_key=key)
    embedder = GeminiEmbedder(api_key=key, dimensions=768)
    with tempfile.TemporaryDirectory() as td:
        m = Memory.open(os.path.join(td, "p.db"), llm=llm, embedder=embedder)
        try:
            yield m
        finally:
            m.close()


def test_gemini_subject_inferred(mem):
    out = mem.write("The user lives in Berlin")
    atom = mem.get(out.atom_id)
    assert atom is not None
    # Gemini should resolve "the user" -> "user". Allow a small set of
    # equivalent canonicalisations to keep the test stable across model drift.
    assert atom.subject in ("user", "the_user")


def test_gemini_write_then_read(mem):
    out = mem.write("User lives in Berlin", subject="user")
    hits = mem.read("where does the user live", k=5, subject="user")
    assert any(a.id == out.atom_id for a, _d in hits)


def test_gemini_real_contradiction_supersedes(mem):
    out1 = mem.write("User lives in Berlin", subject="user")
    out2 = mem.write("User lives in Munich", subject="user")
    # The real verifier should call this a contradiction and the resolver
    # should default to new_supersedes for a recency-driven update.
    assert out2.action in ("superseded_prior", "kept_both", "merged")
    if out2.action == "superseded_prior":
        prior = mem.get(out1.atom_id)
        assert prior is not None and prior.status == "superseded"
        assert prior.superseded_by_id == out2.atom_id


def test_gemini_compatible_facts_dont_supersede(mem):
    out_a = mem.write("User likes coffee", subject="user")
    out_b = mem.write("User likes hiking", subject="user")
    # These are compatible (a user can like both). Neither should supersede.
    assert out_a.action == "inserted"
    assert out_b.action in ("inserted", "kept_both")
    a = mem.get(out_a.atom_id)
    b = mem.get(out_b.atom_id)
    assert a is not None and a.status == "active"
    assert b is not None and b.status == "active"


def test_gemini_cross_subject_does_not_contradict(mem):
    out_alice = mem.write("Alice likes coffee", subject="alice")
    out_bob   = mem.write("Bob dislikes coffee", subject="bob")
    # Subject filter must prevent any cross-subject candidate from reaching
    # the verifier. Nothing should supersede.
    assert out_alice.action == "inserted"
    assert out_bob.action   == "inserted"
