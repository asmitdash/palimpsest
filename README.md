# palimpsest

> **palimpsest** *(n.)* — a manuscript on which the original writing has been effaced to make room for later writing, but of which traces remain.

Contradiction-aware memory for LLM agents. Drop-in SDK for any Python agent. Single SQLite file per memory namespace, no infra.

## Why this exists

Every agent memory layer in 2026 (Letta, Mem0, Zep, Anthropic's memory tool) treats memory as append-only or last-write-wins. Within ~50 turns of any non-trivial task the store is internally inconsistent: contradictory facts coexist, the agent retrieves both, and downstream behaviour is non-deterministic.

palimpsest treats every write as a *verification event*. New atom → retrieve close prior atoms on the same subject → LLM verifier → if a contradiction is real, an LLM resolver decides the lineage (new supersedes / old wins / merge). Forgetting is provenance-preserving: superseded atoms stay readable, with a chain back to their replacements.

## Status

**Day 1 — basic SDK only.** Write / read / lineage walk / stats / retract / reinforce. Contradiction detector lands Day 2; resolver Day 3; MCP server + HTTP API Week 2; contradiction-injection eval Week 3.

## Install

```bash
pip install -e .
```

Dependencies: `pydantic`, `sqlite-vec`, `google-genai`, `anthropic`, `typer`, `rich`, `fastapi`, `tenacity`, `structlog`. All free.

## Quickstart

```python
from palimpsest import Memory

# Stub providers — no API keys needed for tests.
import os
os.environ["PALIMPSEST_LLM_PROVIDER"]       = "stub"
os.environ["PALIMPSEST_EMBEDDING_PROVIDER"] = "stub"

with Memory.open("agent.db") as mem:
    aid = mem.write("User lives in Berlin", subject="user")
    for atom, distance in mem.read("where does the user live?", subject="user"):
        print(f"[{distance:.3f}] {atom.content}")
```

CLI:

```bash
palimpsest write "User lives in Berlin" --subject user
palimpsest read  "where does the user live?" --k 5
palimpsest stats
```

## Providers

LLM: **Gemini default**, **Claude Sonnet** swap, **stub** for offline tests. Set `PALIMPSEST_LLM_PROVIDER` to `gemini` / `anthropic` / `stub` and provide the matching API key (`GEMINI_API_KEY` / `ANTHROPIC_API_KEY`).

Embeddings: **Gemini text-embedding-004 (768-d) default**, **stub** for offline. Set `PALIMPSEST_EMBEDDING_PROVIDER`.

## Data model

Every memory unit is an `Atom`:

| Field | Why |
|---|---|
| `content` | the text |
| `kind` | `episodic` / `semantic` / `procedural` |
| `subject` | the entity this is about — fixes the false-positive avalanche that haunts subject-blind stores |
| `status` | `active` / `superseded` / `contradicted` / `retracted` |
| `supersedes_id` / `superseded_by_id` | lineage edges — `mem.lineage(id)` walks these |
| `confidence` / `reinforcement_count` | grow when independent sources re-assert; decay can be applied later |

## Tests

```bash
PALIMPSEST_LLM_PROVIDER=stub PALIMPSEST_EMBEDDING_PROVIDER=stub pytest -v
```

## Roadmap

- [x] Day 1 — schemas, SQLite + sqlite-vec store, providers, subject extractor, write / read / lineage SDK, stub-only tests
- [ ] Day 2 — contradiction detector (verifier loop), automatic supersede on write
- [ ] Day 3 — resolver (new wins / old wins / merge / keep both), confidence dynamics
- [ ] Week 2 — MCP server, HTTP API, durable replay (`as_of`)
- [ ] Week 3 — contradiction-injection eval vs. Letta / Mem0 / append-only baselines

## License

MIT.
