"""palimpsest MCP server (stdio).

Speaks the Model Context Protocol over stdin/stdout. No MCP SDK dependency —
the spec is small and we keep deps minimal.

Tools exposed:
  palimpsest_write     — insert / supersede / merge an atom
  palimpsest_read      — vector search
  palimpsest_lineage   — walk the supersedes chain
  palimpsest_episodic  — episodic-window retrieval
  palimpsest_consolidate — run semantic consolidation pass
  palimpsest_decay     — run confidence decay pass
  palimpsest_stats     — counts

Run as:
    python -m palimpsest.mcp --db /path/to/agent.db

An MCP-aware client (Claude Desktop, etc.) can then point at this binary in
its config and use the tools as it would any other MCP server.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from palimpsest import Memory


PROTOCOL_VERSION = "2024-11-05"

# ----- tool definitions ------------------------------------------------

def _tool_defs() -> list[dict[str, Any]]:
    return [
        {
            "name": "palimpsest_write",
            "description": (
                "Write a memory atom. Runs contradiction detection by default. "
                "Returns the action taken (inserted / reinforced / superseded_prior / "
                "rejected_old_wins / merged / kept_both)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {"type": "string", "enum": ["episodic", "semantic", "procedural"], "default": "semantic"},
                    "subject": {"type": "string"},
                    "source": {"type": "string"},
                    "confidence": {"type": "number", "default": 1.0},
                    "check_contradictions": {"type": "boolean", "default": True},
                },
                "required": ["content"],
            },
        },
        {
            "name": "palimpsest_read",
            "description": "Vector search active memory. Returns atoms ordered by distance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 8},
                    "subject": {"type": "string"},
                    "kind": {"type": "string", "enum": ["episodic", "semantic", "procedural"]},
                    "include_superseded": {"type": "boolean", "default": False},
                    "as_of": {"type": "string", "description": "ISO 8601 timestamp"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "palimpsest_lineage",
            "description": "Walk the supersedes chain for an atom (oldest first).",
            "inputSchema": {
                "type": "object",
                "properties": {"atom_id": {"type": "string"}},
                "required": ["atom_id"],
            },
        },
        {
            "name": "palimpsest_episodic",
            "description": "Episodic-window retrieval: atoms within a time bound.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601"},
                    "end":   {"type": "string", "description": "ISO 8601"},
                    "limit": {"type": "integer", "default": 200},
                },
            },
        },
        {
            "name": "palimpsest_consolidate",
            "description": "Run semantic consolidation pass (merge near-duplicates). Returns ids of new merged atoms.",
            "inputSchema": {
                "type": "object",
                "properties": {"subject": {"type": "string"}},
            },
        },
        {
            "name": "palimpsest_decay",
            "description": "Apply time-based confidence decay. Atoms below threshold are retracted (lineage preserved).",
            "inputSchema": {
                "type": "object",
                "properties": {"forget_threshold": {"type": "number", "default": 0.05}},
            },
        },
        {
            "name": "palimpsest_prune",
            "description": "Hard prune by confidence and / or per-subject cap.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "confidence_threshold": {"type": "number", "default": 0.10},
                    "max_kept_per_subject": {"type": "integer"},
                },
            },
        },
        {
            "name": "palimpsest_stats",
            "description": "Return counts of atoms by status, plus the embedding dimension.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


# ----- dispatch --------------------------------------------------------

class Server:
    def __init__(self, mem: Memory) -> None:
        self.mem = mem

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        rid = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            return _ok(rid, {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": "palimpsest", "version": "0.0.2"},
                "capabilities": {"tools": {}},
            })
        if method == "tools/list":
            return _ok(rid, {"tools": _tool_defs()})
        if method == "tools/call":
            return _ok(rid, self._call_tool(params))
        if method == "notifications/initialized":
            return None  # no response for notifications
        return _err(rid, -32601, f"Method not found: {method}")

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            payload = self._dispatch(name, args)
            return {
                "content": [{"type": "text", "text": json.dumps(payload, default=_json_default, indent=2)}],
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"error: {e}"}],
                "isError": True,
            }

    def _dispatch(self, name: str | None, args: dict[str, Any]) -> Any:
        if name == "palimpsest_write":
            outcome = self.mem.write(
                content=args["content"],
                kind=args.get("kind", "semantic"),
                subject=args.get("subject"),
                source=args.get("source"),
                confidence=float(args.get("confidence", 1.0)),
                check_contradictions=bool(args.get("check_contradictions", True)),
            )
            return {
                "action": outcome.action,
                "atom_id": str(outcome.atom_id) if outcome.atom_id else None,
                "superseded_ids": [str(x) for x in outcome.superseded_ids],
            }

        if name == "palimpsest_read":
            as_of = args.get("as_of")
            hits = self.mem.read(
                args["query"],
                k=int(args.get("k", 8)),
                subject=args.get("subject"),
                kind=args.get("kind"),
                include_superseded=bool(args.get("include_superseded", False)),
                as_of=datetime.fromisoformat(as_of) if as_of else None,
            )
            return [
                {"atom_id": str(a.id), "content": a.content, "kind": a.kind,
                 "subject": a.subject, "confidence": a.confidence, "distance": d}
                for a, d in hits
            ]

        if name == "palimpsest_lineage":
            chain = self.mem.lineage(UUID(args["atom_id"]))
            return [
                {"atom_id": str(a.id), "status": a.status, "content": a.content,
                 "supersedes_id": str(a.supersedes_id) if a.supersedes_id else None}
                for a in chain
            ]

        if name == "palimpsest_episodic":
            atoms = self.mem.episodic_window(
                subject=args.get("subject"),
                start=datetime.fromisoformat(args["start"]) if args.get("start") else None,
                end=datetime.fromisoformat(args["end"]) if args.get("end") else None,
                limit=int(args.get("limit", 200)),
            )
            return [
                {"atom_id": str(a.id), "content": a.content, "subject": a.subject,
                 "source": a.source, "created_at": a.created_at.isoformat()}
                for a in atoms
            ]

        if name == "palimpsest_consolidate":
            ids = self.mem.consolidate(subject=args.get("subject"))
            return {"merged_atom_ids": [str(x) for x in ids], "count": len(ids)}

        if name == "palimpsest_decay":
            s = self.mem.decay(forget_threshold=float(args.get("forget_threshold", 0.05)))
            return {"inspected": s.inspected, "decayed": s.decayed, "retracted": s.retracted}

        if name == "palimpsest_prune":
            s = self.mem.prune(
                confidence_threshold=float(args.get("confidence_threshold", 0.10)),
                max_kept_per_subject=args.get("max_kept_per_subject"),
            )
            return {"inspected": s.inspected, "retracted": s.retracted}

        if name == "palimpsest_stats":
            return self.mem.stats()

        raise ValueError(f"unknown tool: {name}")


# ----- IO helpers ------------------------------------------------------

def _ok(rid: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _json_default(o: Any) -> Any:
    if isinstance(o, UUID):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"unserialisable: {type(o).__name__}")


# ----- entrypoint ------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="palimpsest.mcp", description="palimpsest MCP server (stdio)")
    parser.add_argument("--db", type=Path, required=True, help="path to the palimpsest sqlite file")
    args = parser.parse_args(argv)

    mem = Memory.open(args.db)
    server = Server(mem)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps(_err(None, -32700, f"parse error: {e}")) + "\n")
            sys.stdout.flush()
            continue
        response = server.handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, default=_json_default) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
