import os

# Force stub providers BEFORE palimpsest is imported anywhere.
os.environ.setdefault("PALIMPSEST_LLM_PROVIDER", "stub")
os.environ.setdefault("PALIMPSEST_EMBEDDING_PROVIDER", "stub")
