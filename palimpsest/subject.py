"""Subject extraction.

`subject` is the entity an atom is about. Two atoms only contradict if their
subjects refer to the same entity — this single rule is what makes the
contradiction layer accurate enough to ship.

If the caller passes `subject=` to write(), we trust them and skip the LLM.
Otherwise we call the provider with a tight schema.
"""

from __future__ import annotations

from typing import Any

from palimpsest.providers import LLMProvider, get_llm_provider


_SUBJECT_SCHEMA: dict[str, Any] = {
    "name": "extract_subject",
    "description": "Extract the canonical subject (entity this statement is about).",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": (
                    "Lowercase canonical entity name. Use 'user' for the end-user, "
                    "'agent' for the AI agent itself, otherwise the most specific "
                    "identifier mentioned (a person's name, a project, a system)."
                ),
            }
        },
        "required": ["subject"],
    },
}


_SUBJECT_SYSTEM = """\
You extract the SUBJECT of a single statement — the entity it is about.

Rules:
1. Use 'user' for the end-user the agent is serving.
2. Use 'agent' for the AI agent itself.
3. Otherwise pick the most specific named entity: a person's first name, a project
   identifier, a system or service. Lowercase. No spaces — use underscores.
4. If multiple entities are mentioned, pick the GRAMMATICAL subject of the sentence.
5. If no entity is identifiable, use 'unknown'.

Output via the `extract_subject` tool only.
"""


def extract_subject(
    content: str, *, provider: LLMProvider | None = None,
) -> str:
    """Return the canonical subject for `content`.

    A thin wrapper — if you already know the subject, pass it directly to
    Memory.write() and skip this entirely.
    """
    p = provider or get_llm_provider()
    result = p.call(system=_SUBJECT_SYSTEM, user=content, schema=_SUBJECT_SCHEMA)
    raw = (result.payload.get("subject") or "unknown").strip().lower()
    return _canonicalise(raw)


def _canonicalise(s: str) -> str:
    # collapse whitespace, replace spaces with underscores, strip punctuation tails
    s = "_".join(s.split())
    return "".join(ch for ch in s if ch.isalnum() or ch == "_") or "unknown"
