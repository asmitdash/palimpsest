"""LLM + embedding providers.

Same shape as palimpsest's other neighbours but written from scratch — the
EDM provider module isn't lifted; the schemas here differ enough that copying
would create more confusion than savings.

Selection order at runtime:
  1. EDM_LLM_PROVIDER / PALIMPSEST_LLM_PROVIDER env override
  2. PALIMPSEST_LLM_PROVIDER  default = stub (offline tests)
  3. Explicit constructor argument

Default LLM is Gemini (gemini-2.5-flash). Claude Sonnet is the swap.
Default embedder is Gemini (text-embedding-004, 768d). Stub embedder for tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, Protocol

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMResult:
    payload: dict[str, Any]
    raw: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int


class LLMProvider(Protocol):
    name: str
    model: str

    def call(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult: ...


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]: ...


# ----- Gemini -----------------------------------------------------------

class GeminiLLM:
    name = "gemini"

    def __init__(self, *, api_key: str, model: str = "gemini-2.5-flash") -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def _to_gemini_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Lower JSON Schema -> Gemini-accepted subset."""

        def lower(node: Any) -> Any:
            if isinstance(node, dict):
                t = node.get("type")
                if isinstance(t, list):
                    node = dict(node)
                    node["type"] = next((x for x in t if x != "null"), t[0])
                return {k: lower(v) for k, v in node.items() if k != "additionalProperties"}
            if isinstance(node, list):
                return [lower(x) for x in node]
            return node

        return lower(schema["input_schema"])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=self._to_gemini_schema(schema),
            max_output_tokens=2048,
            temperature=0.0,
        )
        response = self._client.models.generate_content(
            model=self.model, contents=user, config=cfg,
        )
        text = response.text or ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Gemini returned non-JSON: {text[:300]}") from e
        usage = getattr(response, "usage_metadata", None)
        return LLMResult(
            payload=payload,
            raw={"text": text, "model": self.model},
            model=self.model,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )


class GeminiEmbedder:
    name = "gemini"
    dimensions = 768

    def __init__(self, *, api_key: str, model: str = "text-embedding-004") -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        if not texts:
            return []
        # google-genai >= 0.3 exposes embed_content via the embeddings api.
        results: list[list[float]] = []
        for t in texts:
            r = self._client.models.embed_content(model=self.model, contents=t)
            # response.embeddings is a list of {values: [...]} objects
            emb = r.embeddings[0]
            vals = getattr(emb, "values", None) or emb["values"]  # type: ignore[index]
            results.append(list(vals))
        return results


# ----- Anthropic --------------------------------------------------------

class AnthropicLLM:
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def call(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        tool_name = schema["name"]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            tools=[schema],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user}],
        )
        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise RuntimeError(f"Anthropic did not invoke tool {tool_name}")
        return LLMResult(
            payload=tool_use.input,
            raw=json.loads(response.model_dump_json()),
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ----- Stub (offline tests) ---------------------------------------------

class StubLLM:
    """Deterministic stub. Tool-name dispatch keeps unit tests reproducible."""

    name = "stub"
    model = "stub-llm-v1"

    def call(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        tool = schema.get("name", "")
        if tool == "extract_subject":
            payload = self._stub_subject(user)
        elif tool == "verify_contradiction":
            payload = self._stub_verify(user)
        elif tool == "resolve_contradiction":
            payload = self._stub_resolve(user)
        else:
            payload = {}
        return LLMResult(
            payload=payload,
            raw={"stub": True, "tool": tool, "input_hash": _hash(user)},
            model=self.model,
            input_tokens=len(user) // 4,
            output_tokens=len(json.dumps(payload)) // 4,
        )

    def _stub_subject(self, user: str) -> dict[str, Any]:
        u = user.lower()
        if " user " in f" {u} " or u.startswith("user ") or " the user " in u:
            return {"subject": "user"}
        if "agent" in u:
            return {"subject": "agent"}
        # otherwise pull the first capitalised noun from the original text
        for word in re.findall(r"[A-Z][a-z]+", user):
            return {"subject": word.lower()}
        return {"subject": "unknown"}

    def _stub_verify(self, user: str) -> dict[str, Any]:
        # Trigger contradiction when prompt contains both an X and a not-X marker.
        u = user.lower()
        # Explicit antonym pair example (likes/dislikes)
        if ("likes" in u and "dislikes" in u) or ("loves" in u and "hates" in u):
            return {
                "contradicts": True, "severity": "high",
                "rationale": "Stub: detected antonym pair on same subject.",
            }
        if "berlin" in u and "munich" in u:
            return {
                "contradicts": True, "severity": "medium",
                "rationale": "Stub: location atoms differ.",
            }
        return {"contradicts": False, "severity": "low", "rationale": "Stub: no conflict."}

    def _stub_resolve(self, user: str) -> dict[str, Any]:
        # Default policy: new wins.
        return {
            "action": "new_supersedes",
            "rationale": "Stub: most recent assertion wins by default.",
            "merged_content": None,
        }


class StubEmbedder:
    """Hash-bag-of-words deterministic embedder. Words map to fixed dims so
    overlapping texts have non-trivial cosine similarity."""

    name = "stub"
    dimensions = 768

    def embed(self, texts: list[str], *, input_type: str = "document") -> list[list[float]]:
        return [_text_to_vec(t, self.dimensions) for t in texts]


def _text_to_vec(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for token in _tokenize(text):
        h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
    n = sum(x * x for x in vec) ** 0.5
    if n == 0:
        return vec
    return [x / n for x in vec]


def _tokenize(text: str) -> list[str]:
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


# ----- Factories --------------------------------------------------------

def get_llm_provider() -> LLMProvider:
    name = (os.environ.get("PALIMPSEST_LLM_PROVIDER") or "stub").lower()
    if name == "stub":
        return StubLLM()
    if name == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        return GeminiLLM(api_key=api_key, model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    if name == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return AnthropicLLM(api_key=api_key, model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
    raise ValueError(f"Unknown PALIMPSEST_LLM_PROVIDER: {name}")


def get_embedding_provider() -> EmbeddingProvider:
    name = (os.environ.get("PALIMPSEST_EMBEDDING_PROVIDER") or "stub").lower()
    if name == "stub":
        return StubEmbedder()
    if name == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set for embeddings")
        return GeminiEmbedder(api_key=api_key, model=os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004"))
    raise ValueError(f"Unknown PALIMPSEST_EMBEDDING_PROVIDER: {name}")
