# palimpsest

> **palimpsest** *(n.)* — a manuscript on which the original writing has been effaced to make room for later writing, but of which traces remain.

**Contradiction-aware memory for LLM agents.** A drop-in Python SDK and MCP server for any agent. Single SQLite file per memory namespace, no infrastructure, no service to run. Built on top of `sqlite-vec`.

[![tests](https://img.shields.io/badge/tests-62%20passed-brightgreen)](https://github.com/asmitdash/palimpsest/tree/main/tests)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![status](https://img.shields.io/badge/status-alpha%20v0.0.2-orange)]()
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why this exists

Every agent memory layer in 2026 (Letta, Mem0, Zep, Anthropic's memory tool) treats memory as either append-only or last-write-wins. Within ~50 turns of any non-trivial agentic task the store becomes internally inconsistent: contradictory facts coexist, the agent retrieves both, and downstream behaviour goes non-deterministic.

palimpsest treats every write as a **verification event**:

1. New atom arrives → embed it, extract its subject.
2. Retrieve close prior atoms about the *same subject*.
3. LLM verifier asks: does the new atom genuinely contradict any prior?
4. If yes, an LLM resolver decides the lineage — `new_supersedes` / `old_wins` / `merge` / `keep_both`.
5. Apply the action atomically. Forgetting is provenance-preserving — superseded atoms stay readable, with a chain to their replacements.

Subject-scoping is the load-bearing fix that makes this work. "User likes coffee" and "Bob dislikes coffee" don't trigger the verifier because their subjects differ. Subject-blind stores hit a wall of false positives.

---

## What ships in v0.0.2

Eight components, all wired into the public `Memory` SDK and covered by tests.

| Component | What it does | Key API |
|---|---|---|
| **Episodic memory store** | Time-bounded events with fast 14-day decay; source-grouped replay | `mem.write(kind="episodic", source=...)`, `mem.episodic_window`, `mem.episodic_chain` |
| **Semantic memory store** | Long-lived facts with 90-day decay; periodic consolidation merges near-duplicates | `mem.write(kind="semantic")`, `mem.consolidate(threshold=)` |
| **Procedural memory store** | Learned how-tos with slow 180-day decay | `mem.write(kind="procedural")` |
| **Contradiction detection engine** | Same-subject candidate retrieval → LLM verifier loop, runs on every write by default | `mem.write(check_contradictions=True)` |
| **Memory verifier** | Strict prompt + structured tool output; defaults to "not contradicting" on uncertainty | `palimpsest/prompts.py`, versioned `verify.v1` |
| **Resolver** | Picks `new_supersedes` / `old_wins` / `merge` / `keep_both` and applies it | `WriteOutcome.action` |
| **Lineage graph** | Supersedes pointers form a DAG; `mem.lineage(id)` walks it; `merge` action inherits from every input | `mem.lineage(atom_id)` |
| **Confidence engine** | Time-based decay (kind-specific half-life), contradiction-pressure penalty, independent-source reinforcement | `mem.reinforce(...)`, `mem.decay()` |
| **Forgetting layer** | Auto-retracts below-threshold atoms; per-subject prune cap; lineage preserved | `mem.decay()`, `mem.prune()` |
| **MCP server** | Stdio JSON-RPC, 8 tools, no MCP SDK dependency | `python -m palimpsest.mcp --db ...` |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Public API                                │
│  Memory.open(path) → write / read / lineage / consolidate / decay /    │
│  prune / retract / reinforce / get / list_subject / stats              │
└──────────┬───────────────────────────────────────────────┬─────────────┘
           │                                               │
           ▼                                               ▼
┌──────────────────────┐                    ┌──────────────────────────┐
│   Write hot path     │                    │     Maintenance loop     │
│   (per write)        │                    │     (periodic)           │
│                      │                    │                          │
│  1. Subject extract  │                    │  - mem.decay()           │
│  2. Embed            │                    │    (time-based decay,    │
│  3. Retrieve same-   │                    │     auto-retract below   │
│     subject candi-   │                    │     threshold)           │
│     dates from vec0  │                    │                          │
│  4. LLM verifier     │                    │  - mem.prune()           │
│  5. LLM resolver     │                    │    (hard cap or thresh)  │
│  6. Apply lineage    │                    │                          │
│     action +         │                    │  - mem.consolidate()     │
│     persist          │                    │    (semantic-store merge │
│                      │                    │     of near-duplicates)  │
└──────────┬───────────┘                    └──────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Storage  (single SQLite file)                    │
│  atoms       (id, content, kind, subject, status, lineage,           │
│               confidence, reinforcement_count, ...)                  │
│  atom_embeddings   (sqlite-vec virtual table)                        │
│  atom_vec_map      (atom UUID ↔ vec0 rowid)                          │
│  reinforcements    (append-only re-assertion log)                    │
│  contradictions    (append-only contradiction event log)             │
└──────────────────────────────────────────────────────────────────────┘
```

The two SQLite-side wedges:

- **One file on disk per namespace.** Drop into any agent process. Letta / Mem0 / Zep require running a service; palimpsest doesn't.
- **vec0 + atom_vec_map indirection.** sqlite-vec indexes by integer rowid; we keep a map so atom UUIDs ride that index without leaking the indirection to callers.

---

## Data model

The unit is an `Atom`:

| Field | Why |
|---|---|
| `id` | UUID, stable across the lifetime of the store |
| `content` | the text — what the agent wants to remember |
| `kind` | `episodic` / `semantic` / `procedural` |
| `subject` | the entity this atom is **about** — the fix for the false-positive avalanche that haunts subject-blind stores |
| `source` | optional opaque ref: turn id, tool-call id, doc id |
| `status` | `active` / `superseded` / `contradicted` / `retracted` |
| `supersedes_id` / `superseded_by_id` | lineage edges; `mem.lineage(id)` walks them |
| `confidence` / `reinforcement_count` | grow when independent sources re-assert; decay can be applied later |

Two atoms only contradict if their *subjects refer to the same entity*. This is the rule that lets the verifier ship at acceptable precision.

---

## Resolution actions

When the verifier confirms a contradiction, the resolver picks one of four actions:

| Action | When | Effect |
|---|---|---|
| `new_supersedes` | Most updates — the new fact is a recency-driven correction | New atom inserted as `active`, prior(s) marked `superseded` and pointed at the new one |
| `old_wins` | The new atom looks like noise / lower trust than the prior | New atom rejected (not inserted); prior(s) get a confidence-pressure penalty |
| `merge` | Both are partially right — the resolver synthesises a combined statement | A *third* atom written with the combined content; both inputs marked `superseded` and pointed at the merged atom |
| `keep_both` | Verifier was wrong on closer inspection | New atom inserted alongside; nothing supersedes |

Every contradiction event lands in the `contradictions` log with severity, rationale, and resolver decision — this is the audit trail.

---

## Install

```bash
git clone https://github.com/asmitdash/palimpsest.git
cd palimpsest
pip install -e .
```

Dependencies (all free): `pydantic`, `sqlite-vec`, `google-genai`, `anthropic`, `typer`, `rich`, `fastapi`, `uvicorn`, `tenacity`, `structlog`.

For tests-only minimum: `pip install pydantic sqlite-vec pytest tenacity`.

---

## Provider configuration

| Env var | Default | Purpose |
|---|---|---|
| `PALIMPSEST_LLM_PROVIDER` | `stub` | `gemini` / `anthropic` / `stub` |
| `PALIMPSEST_EMBEDDING_PROVIDER` | `stub` | `gemini` / `stub` |
| `GEMINI_API_KEY` | — | Required for `gemini` |
| `GEMINI_MODEL` | `gemini-2.5-flash` | LLM model |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `PALIMPSEST_GEMINI_EMBEDDING_DIM` | `768` | Truncation dim via Matryoshka Representation Learning — keeps the store schema consistent across embedders |
| `ANTHROPIC_API_KEY` | — | Required for `anthropic` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model |

**Important note from the v0.0.2 push:** Gemini's `text-embedding-004` returns `404 NOT_FOUND` on the v1beta endpoint as of late 2026. palimpsest defaults to `gemini-embedding-001` truncated to 768d via MRL — the store schema stays consistent across the toolchain.

---

## Quickstart — Python SDK

```python
from palimpsest import Memory

# Stub providers — no API keys needed for tests / local dev.
import os
os.environ["PALIMPSEST_LLM_PROVIDER"]       = "stub"
os.environ["PALIMPSEST_EMBEDDING_PROVIDER"] = "stub"

with Memory.open("agent.db") as mem:
    out_a = mem.write("User lives in Berlin", subject="user")
    out_b = mem.write("User lives in Munich", subject="user")
    print(out_b.action)              # 'superseded_prior'
    print(out_b.superseded_ids)       # [out_a.atom_id]

    for atom, distance in mem.read("where does the user live?", subject="user"):
        print(f"[{distance:.3f}] {atom.content}")    # 'User lives in Munich'

    chain = mem.lineage(out_b.atom_id)
    for a in chain:
        print(f"  {a.id} [{a.status}] {a.content}")
    # Berlin (superseded) → Munich (active)
```

Full demo: [`examples/agent_loop.py`](examples/agent_loop.py).

---

## Quickstart — CLI

```bash
palimpsest write "User lives in Berlin" --subject user
palimpsest write "User lives in Munich" --subject user
palimpsest read  "where does the user live?" --subject user --k 5
palimpsest lineage <atom-id>
palimpsest stats
```

---

## Quickstart — MCP server

palimpsest exposes the SDK as an MCP server over stdio. To wire it into Claude Desktop or any MCP-aware client, point the client at:

```bash
python -m palimpsest.mcp --db /path/to/agent.db
```

Tools advertised: `palimpsest_write`, `palimpsest_read`, `palimpsest_lineage`, `palimpsest_episodic`, `palimpsest_consolidate`, `palimpsest_decay`, `palimpsest_prune`, `palimpsest_stats`.

The server speaks the MCP JSON-RPC spec directly — no MCP SDK dependency.

---

## Maintenance loop

Three optional periodic passes — none of them block writes:

```python
# Periodic confidence decay (e.g. once per agent session boot)
mem.decay()                     # episodic: 14d half-life, semantic: 90d, procedural: 180d

# Hard prune (rare — call before backups, or after a noisy session)
mem.prune(confidence_threshold=0.10, max_kept_per_subject=200)

# Semantic consolidation (e.g. nightly cron)
mem.consolidate(subject="user", threshold=0.85)
# threshold 0.85 is right for production embedders; pass lower for stub or weak embedders
```

---

## Tests

```bash
PALIMPSEST_LLM_PROVIDER=stub PALIMPSEST_EMBEDDING_PROVIDER=stub pytest -v
```

**62 passed, 1 skipped** (Anthropic test, gated on `ANTHROPIC_API_KEY`) — verified on commit `633eb25` against Python 3.11, sqlite-vec 0.1.9, pydantic 2.13.4.

Test file layout:

| File | Tests | Coverage |
|---|---|---|
| `test_day1_core.py` | 7 | write/read SDK surface, manual lineage, reinforce, retract, stats |
| `test_contradiction.py` | 8 | auto-supersede, no-cross-subject, merge / old_wins / keep_both, log |
| `test_episodic.py` | 3 | time-window filter, source-grouped chains, kind interaction |
| `test_confidence_decay.py` | 5 | decay reduces, retracts below threshold, independent-source bonus, pressure penalty, prune cap |
| `test_consolidation.py` | 3 | cluster merge, no-cluster skip, LLM-refusal path |
| `test_mcp.py` | 12 | every tool, JSON-RPC error path, real stdio subprocess |
| `test_cli.py` | 4 | write+read, stats JSON, lineage CLI, no-hits path |
| `test_edge_cases.py` | 13 | empty content, dim mismatch, multi-candidate, merge fallback, cross-kind, as_of, cycle protection, etc. |
| `test_examples.py` | 1 | runs `agent_loop.py` as a subprocess |
| `test_gemini_integration.py` | 5 (gated) | live Gemini smoke: subject extract, write+read, real contradiction supersede, compatible facts, cross-subject |
| `test_anthropic_integration.py` | 1 (gated) | live Claude subject extraction |

---

## Roadmap

- [x] **v0.0.1 (Day 1)** — schemas, SQLite + sqlite-vec store, providers, subject extractor, write / read / lineage SDK, stub-only tests
- [x] **v0.0.2** — contradiction engine, verifier, resolver, confidence dynamics, episodic store, semantic consolidation, forgetting layer, MCP server, CLI, edge-case tests, live Gemini integration
- [ ] **v0.0.3** — HTTP API server, contradiction-injection eval vs. Letta / Mem0 / append-only baselines, real-LLM regression suite, performance benchmarks at 10K+ atoms
- [ ] **v0.1.0** — Hosted SaaS surface (optional), multi-tenant boundary enforcement, Anthropic vision input for memory atoms

---

## License

MIT. See [`LICENSE`](LICENSE).
