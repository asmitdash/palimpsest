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


def _call(server, name: str, args: dict):
    """Helper: tools/call -> parsed result dict."""
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": name, "arguments": args}}
    resp = server.handle(rpc)
    assert resp is not None
    res = resp["result"]
    assert res["isError"] is False, res
    return json.loads(res["content"][0]["text"])


def test_tool_lineage_walks_supersedes(server):
    a = _call(server, "palimpsest_write", {"content": "User lives in Berlin", "subject": "user"})
    b = _call(server, "palimpsest_write", {"content": "User lives in Munich", "subject": "user"})
    assert b["action"] == "superseded_prior"
    chain = _call(server, "palimpsest_lineage", {"atom_id": b["atom_id"]})
    assert len(chain) == 2
    assert chain[0]["status"] == "superseded"
    assert chain[1]["content"].endswith("Munich")


def test_tool_episodic_filters_and_chains(server):
    _call(server, "palimpsest_write",
          {"content": "Step 1", "kind": "episodic", "subject": "user", "source": "turn_1"})
    _call(server, "palimpsest_write",
          {"content": "Step 2", "kind": "episodic", "subject": "user", "source": "turn_1"})
    _call(server, "palimpsest_write",
          {"content": "Step 3", "kind": "episodic", "subject": "user", "source": "turn_2"})
    atoms = _call(server, "palimpsest_episodic", {"subject": "user", "limit": 50})
    assert len(atoms) == 3
    # confirm both turn_1 and turn_2 are represented
    sources = {a["source"] for a in atoms}
    assert sources == {"turn_1", "turn_2"}


def test_tool_consolidate_with_low_threshold(server):
    # Consolidation tool doesn't expose threshold yet, so this just exercises
    # the dispatch + happy path; it should return an empty list at default threshold.
    _call(server, "palimpsest_write",
          {"content": "User prefers coffee morning",         "subject": "user", "check_contradictions": False})
    _call(server, "palimpsest_write",
          {"content": "User prefers coffee in the morning",  "subject": "user", "check_contradictions": False})
    res = _call(server, "palimpsest_consolidate", {"subject": "user"})
    assert "merged_atom_ids" in res
    assert isinstance(res["count"], int)


def test_tool_decay_returns_summary(server):
    _call(server, "palimpsest_write", {"content": "User likes jazz", "subject": "user"})
    res = _call(server, "palimpsest_decay", {"forget_threshold": 0.05})
    assert {"inspected", "decayed", "retracted"}.issubset(res.keys())


def test_tool_prune_returns_summary(server):
    for i in range(4):
        _call(server, "palimpsest_write",
              {"content": f"Episodic {i}", "kind": "episodic", "subject": "user", "source": f"e{i}"})
    res = _call(server, "palimpsest_prune", {"max_kept_per_subject": 2})
    assert res["inspected"] >= 4
    assert res["retracted"] >= 2


def test_tool_stats_reports_dim(server):
    _call(server, "palimpsest_write", {"content": "Hello", "subject": "user"})
    s = _call(server, "palimpsest_stats", {})
    assert s["atoms_total"] == 1
    assert s["vec_dim"] >= 64


def test_tool_call_unknown_returns_error(server):
    rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "palimpsest_does_not_exist", "arguments": {}}}
    resp = server.handle(rpc)
    res = resp["result"]
    assert res["isError"] is True
    assert "unknown tool" in res["content"][0]["text"]


def test_mcp_stdio_subprocess(tmp_path):
    """Smoke-test the actual stdio main loop end-to-end."""
    import pathlib
    import subprocess
    import sys

    db = tmp_path / "p.db"
    repo_root = str(pathlib.Path(__file__).resolve().parents[1])
    env = {**os.environ,
           "PALIMPSEST_LLM_PROVIDER": "stub",
           "PALIMPSEST_EMBEDDING_PROVIDER": "stub",
           "PYTHONPATH": repo_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "palimpsest.mcp", "--db", str(db)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        # Send three RPCs: initialize, tools/list, tools/call write
        rpcs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "palimpsest_write",
                        "arguments": {"content": "Hello stdio", "subject": "user"}}},
        ]
        stdin = "\n".join(json.dumps(r) for r in rpcs) + "\n"
        out, err = proc.communicate(stdin, timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        raise

    assert proc.returncode == 0, f"stderr: {err}\nstdout: {out}"
    lines = [l for l in out.strip().splitlines() if l.strip()]
    assert len(lines) == 3, f"expected 3 responses, got {len(lines)}: {lines!r}"
    init_r, list_r, call_r = (json.loads(l) for l in lines)
    assert init_r["id"] == 1 and init_r["result"]["serverInfo"]["name"] == "palimpsest"
    assert list_r["id"] == 2 and any(
        t["name"] == "palimpsest_write" for t in list_r["result"]["tools"]
    )
    assert call_r["id"] == 3 and call_r["result"]["isError"] is False
    payload = json.loads(call_r["result"]["content"][0]["text"])
    assert payload["action"] in ("inserted", "reinforced")
