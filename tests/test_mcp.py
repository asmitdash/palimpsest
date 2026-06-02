"""MCP server smoke test — protocol negotiation + tool dispatch over JSON-RPC."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from palimpsest import Memory
from palimpsest.mcp import Server


@pytest.fixture
def server():
    with tempfile.TemporaryDirectory() as td:
        mem = Memory.open(os.path.join(td, "p.db"))
        try:
            yield Server(mem)
        finally:
            mem.close()


def _rpc(method: str, params: dict | None = None, rid: int = 1):
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}


def test_initialize_returns_capabilities(server):
    response = server.handle(_rpc("initialize"))
    assert response is not None
    assert response["result"]["serverInfo"]["name"] == "palimpsest"
    assert "tools" in response["result"]["capabilities"]


def test_tools_list_contains_every_advertised_tool(server):
    response = server.handle(_rpc("tools/list"))
    names = {t["name"] for t in response["result"]["tools"]}  # type: ignore[index]
    assert {
        "palimpsest_write", "palimpsest_read", "palimpsest_lineage",
        "palimpsest_episodic", "palimpsest_consolidate", "palimpsest_decay",
        "palimpsest_prune", "palimpsest_stats",
    }.issubset(names)


def test_tool_call_write_then_read(server):
    write_resp = server.handle(_rpc("tools/call", {
        "name": "palimpsest_write",
        "arguments": {"content": "User lives in Berlin", "subject": "user"},
    }))
    assert write_resp["result"]["isError"] is False  # type: ignore[index]
    payload = json.loads(write_resp["result"]["content"][0]["text"])  # type: ignore[index]
    assert payload["action"] == "inserted"
    assert payload["atom_id"]

    read_resp = server.handle(_rpc("tools/call", {
        "name": "palimpsest_read",
        "arguments": {"query": "where does the user live", "subject": "user", "k": 5},
    }))
    hits = json.loads(read_resp["result"]["content"][0]["text"])  # type: ignore[index]
    assert any("Berlin" in h["content"] for h in hits)


def test_unknown_method_returns_jsonrpc_error(server):
    response = server.handle(_rpc("not/a/real/method"))
    assert response is not None and "error" in response
    assert response["error"]["code"] == -32601


def test_initialized_notification_returns_none(server):
    response = server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert response is None
