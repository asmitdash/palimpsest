"""palimpsest — contradiction-aware agent memory.

A manuscript scraped and reused while traces of the old text remain.

Public surface:
    from palimpsest import Memory, AtomKind
    mem = Memory.open("agent.db")
    atom_id = mem.write("User likes coffee", kind="semantic", subject="user")
    hits = mem.read("what does the user drink?", k=5)
"""

from palimpsest.schemas import Atom, AtomKind, Verdict, Resolution
from palimpsest.memory import Memory

__all__ = ["Atom", "AtomKind", "Verdict", "Resolution", "Memory"]
__version__ = "0.0.1"
