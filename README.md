# palimpsest

> **palimpsest** *(n.)* — a manuscript on which the original writing has been effaced to make room for later writing, but of which traces remain.

Contradiction-aware memory for LLM agents. Drop-in SDK for any Python agent. Single SQLite file per memory namespace, no infra.

## Why this exists

Every agent memory layer in 2026 (Letta, Mem0, Zep, Anthropic's memory tool) treats memory as append-only or last-write-wins. Within ~50 turns of any non-trivial task the store is internally inconsistent: contradictory facts coexist, the agent retrieves both, and downstream behaviour is non-deterministic.

palimpsest treats every write as a *verification event*. New atom → retrieve close prior atoms on the same subject → LLM verifier → if a contradiction is real, an LLM resolver decides the lineage (new supersedes / old wins / merge). Forgetting is provenance-preserving: superseded atoms stay readable, with a chain back to their replacements.

## Status

**v0.0.2 — every contradiction-aware component shipped.**

| Component | What it does |
|---|---|
| Episodic memory store | `kind="episodic"` atoms, time-window retrieval, source-chained replay, fast 14-day decay |
| Semantic memory store | `kind="semantic"` atoms, 90-day decay, periodic consolidation pass merges near-duplicates |
| Procedural store | `kind="procedural"` atoms, slow 180-day decay |
| Contradiction detection engine | On every write, retrieves same-subject candidates, runs the LLM verifier, ignores cross-subject look-alikes |
| Memory verifier | Strict prompt + structured tool output; defaults to "not contradicting" on uncertainty |
| Resolver | `new_supersedes` / `old_wins` / `merge` / `keep_both` — fully wired into write path |
| Lineage graph | Supersedes pointers form a DAG; `mem.lineage(id)` walks it; merged atoms inherit from every input |
| Confidence engine | Time-based decay (kind-specific half-life), contradiction-pressure penalty, independent-source reinforcement |
| Forgetting layer | `mem.decay()` auto-retracts below-threshold atoms; `mem.prune()` enforces per-subject caps. Lineage preserved. |
| MCP server | Stdio JSON-RPC, exposes 8 tools (`palimpsest_write` / `_read` / `_lineage` / `_episodic` / `_consolidate` / `_decay` / `_prune` / `_stats`) |

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

## MCP

palimpsest exposes its SDK as an MCP server over stdio. To wire it into Claude Desktop or any MCP-aware client, point the client at:

```bash
python -m palimpsest.mcp --db /path/to/agent.db
```

Tools advertised: `palimpsest_write`, `palimpsest_read`, `palimpsest_lineage`, `palimpsest_episodic`, `palimpsest_consolidate`, `palimpsest_decay`, `palimpsest_prune`, `palimpsest_stats`.

## Maintenance loop

Three optional passes you call when you want — none of them block writes:

```python
# Periodic confidence decay (e.g. once per agent session boot)
mem.decay()                     # episodic atoms decay fastest, semantic slowest

# Hard prune (rare — call before backups, or after a noisy session)
mem.prune(confidence_threshold=0.10, max_kept_per_subject=200)

# Semantic consolidation (e.g. nightly cron)
mem.consolidate(subject="user")  # merges near-duplicate atoms, preserves lineage
```

## Roadmap

- [x] v0.0.1 (Day 1) — schemas, SQLite + sqlite-vec store, providers, subject extractor, write / read / lineage SDK, stub-only tests
- [x] v0.0.2 — contradiction engine, verifier, resolver, confidence dynamics, episodic store, semantic consolidation, forgetting layer, MCP server, tests for every component
- [ ] v0.0.3 — HTTP API, contradiction-injection eval vs. Letta / Mem0 / append-only baselines, real-LLM regression suite

## License

MIT.
