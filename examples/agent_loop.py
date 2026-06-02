"""Minimal agent-style loop using palimpsest as the memory layer.

Day-1 demo: write a few atoms, retrieve, walk a lineage chain.
Run with the stub providers (no API keys needed):

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
            print(">> writing facts")
            a1 = mem.write("User lives in Berlin", subject="user")
            a2 = mem.write("User likes coffee", subject="user")
            a3 = mem.write("Alice prefers tea", subject="alice")

            print("\n>> read 'what does the user drink' (subject=user)")
            for atom, d in mem.read("what does the user drink", k=3, subject="user"):
                print(f"  [{d:.3f}] {atom.content}")

            print("\n>> manual supersede demo")
            a4 = mem.write("User now lives in Munich", subject="user")
            mem.store.mark_superseded(a1, a4)
            chain = mem.lineage(a4)
            for a in chain:
                print(f"  {a.id} [{a.status}] {a.content}")

            print("\n>> stats")
            print(mem.stats())


if __name__ == "__main__":
    main()
