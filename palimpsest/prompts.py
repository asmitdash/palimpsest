"""Prompt templates + tool schemas for the contradiction layer.

Versioned. Never edit a published version in place; bump the version so
audit logs stay interpretable.
"""

from typing import Any


# ----- Verifier ---------------------------------------------------------

VERIFIER_PROMPT_VERSION = "verify.v1"

VERIFIER_SYSTEM = """\
You are a strict verifier for an agent memory system.

You will see two atoms (factual statements) about the SAME subject. Your job:
decide whether the NEW atom genuinely contradicts the PRIOR atom.

A genuine contradiction means: both cannot be true at the same time about the
same subject. Examples:
  - "User lives in Berlin"  vs "User lives in Munich"     -> contradicts
  - "User likes coffee"     vs "User dislikes coffee"     -> contradicts
  - "User likes coffee"     vs "User likes tea"           -> NOT contradicts (compatible)
  - "User lives in Berlin"  vs "User visited Munich"      -> NOT contradicts
  - "User used to live in Berlin" vs "User now lives in Munich" -> NOT contradicts (temporal compatibility)

Rules:
1. Default to NOT contradicting on uncertainty. False positives ruin memory.
2. Topic overlap is not contradiction.
3. Refinement / specification is not contradiction. ("User lives in Berlin" -> "User lives in Berlin, Mitte" is refinement.)
4. Severity: high (direct factual conflict), medium (likely conflict, some doubt), low (weak signal — usually means do not flag).

Respond ONLY via the `verify_contradiction` tool.
"""

VERIFIER_SCHEMA: dict[str, Any] = {
    "name": "verify_contradiction",
    "description": "Report whether the new atom contradicts the prior atom.",
    "input_schema": {
        "type": "object",
        "properties": {
            "contradicts": {"type": "boolean"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "rationale": {"type": "string"},
        },
        "required": ["contradicts", "severity", "rationale"],
    },
}


# ----- Resolver ---------------------------------------------------------

RESOLVER_PROMPT_VERSION = "resolve.v1"

RESOLVER_SYSTEM = """\
You resolve a confirmed contradiction between two atoms in an agent memory store.

You will see:
  - PRIOR atom: content + when it was created + reinforcement count + confidence
  - NEW atom:   content + source (if any)

Pick ONE action:
  - new_supersedes : the new atom replaces the prior. Use when the new statement
                     is more recent and there is no reason to doubt it. Default
                     for most contradictions in agent memory — recency wins.
  - old_wins       : keep the prior. Use only when the new atom looks like noise,
                     a misread, or has lower trust than the prior.
  - merge          : write a third atom that combines them. Use when both are
                     partially right (e.g. one captures the past, one the present).
                     You MUST provide `merged_content` in this case.
  - keep_both      : the verifier was wrong; they don't actually contradict. Use
                     sparingly — verifier already passed.

Default action when uncertain: new_supersedes.

Respond ONLY via the `resolve_contradiction` tool.
"""

RESOLVER_SCHEMA: dict[str, Any] = {
    "name": "resolve_contradiction",
    "description": "Decide what to do with a confirmed contradiction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["new_supersedes", "old_wins", "merge", "keep_both"],
            },
            "rationale": {"type": "string"},
            "merged_content": {
                "type": "string",
                "description": "Required when action='merge'. A single statement combining the two.",
            },
        },
        "required": ["action", "rationale"],
    },
}


# ----- Semantic consolidation -------------------------------------------

CONSOLIDATOR_PROMPT_VERSION = "consolidate.v1"

CONSOLIDATOR_SYSTEM = """\
You consolidate near-duplicate semantic memory atoms into a single, stronger statement.

You will see N atoms (2-6) about the same subject that are semantically similar
but worded differently. Output ONE merged statement that:
  - preserves every distinct fact present in the inputs
  - drops redundancy
  - is in neutral, third-person prose
  - is no longer than the longest input

If the atoms are NOT actually duplicates (they say different things), respond
with should_merge=false and an empty merged_content.

Respond ONLY via the `consolidate_atoms` tool.
"""

CONSOLIDATOR_SCHEMA: dict[str, Any] = {
    "name": "consolidate_atoms",
    "description": "Merge near-duplicate atoms or signal they are not duplicates.",
    "input_schema": {
        "type": "object",
        "properties": {
            "should_merge": {"type": "boolean"},
            "merged_content": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["should_merge", "merged_content"],
    },
}
