"""Anthropic provider integration test, gated on ANTHROPIC_API_KEY.

This intentionally stays minimal — one round-trip through Claude Sonnet to
prove the swap works. Skipped (not failed) when the env var is missing, the
same convention as test_gemini_integration.py.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def test_anthropic_subject_extraction():
    from palimpsest.providers import AnthropicLLM

    llm = AnthropicLLM(api_key=os.environ["ANTHROPIC_API_KEY"])
    schema = {
        "name": "extract_subject",
        "description": "Extract subject",
        "input_schema": {
            "type": "object",
            "properties": {"subject": {"type": "string"}},
            "required": ["subject"],
        },
    }
    res = llm.call(
        system='Extract the subject of a sentence. Lowercase. Use "user" for the end-user.',
        user="The user lives in Berlin",
        schema=schema,
    )
    assert res.payload.get("subject") in ("user", "the_user")
    assert res.input_tokens > 0
    assert res.output_tokens > 0
