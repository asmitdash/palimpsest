"""Minimal agent-style loop using palimpsest as the memory layer.

Demonstrates contradiction-aware writes (the engine auto-supersedes), retrieval,
and lineage walks. Run with the stub providers — no API keys needed:

    PALIMPSEST_LLM_PROVIDER=stub PALIMPSEST_EMBEDDING_PROVIDER=stub \
        python examples/agent_loop.py
"""

from __future__ import annotations

import os
import tempfile

from palimpsest import Memory


def main() -> None:
    os.environ.setdefault("PALIMPSEST_LLM_PROVIDER", "stub")
    os.environ.setdefault("PALIMPSEST_EMBEDDING_PROVIDER", "stub")

    with tempfile.TemporaryDirectory() as td:
        with Memory.open(f"{td}/agent.db") as mem:
            print(">> writing facts (engine runs the verifier on each write)")
            out_a = mem.write("User lives in Berlin", subject="user")
            print(f"  Berlin -> {out_a.action} ({out_a.atom_id})")
            out_b = mem.write("User likes coffee", subject="user")
            print(f"  coffee -> {out_b.action} ({out_b.atom_id})")
            out_c = mem.write("Alice prefers tea", subject="alice")
            print(f"  tea    -> {out_c.action} ({out_c.atom_id})")

            print("\n>> read 'what does the user drink' (subject=user)")
            for atom, d in mem.read("what does the user drink", k=3, subject="user"):
                print(f"  [{d:.3f}] {atom.content}")

            print("\n>> contradiction-aware update (Berlin -> Munich)")
            out_d = mem.write("User lives in Munich", subject="user")
            print(f"  Munich -> {out_d.action} (superseded {len(out_d.superseded_ids)} prior)")

            print("\n>> lineage of the new atom (oldest first)")
            chain = mem.lineage(out_d.atom_id)
            for a in chain:
                print(f"  {a.id} [{a.status}] {a.content}")

            print("\n>> stats")
            print(mem.stats())


if __name__ == "__main__":
    main()
