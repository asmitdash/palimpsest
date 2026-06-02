"""CLI smoke tests via typer.testing.CliRunner.

We exercise the `palimpsest` CLI end-to-end against a temporary db. Every
invocation gets the same --db path so writes and reads share state.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from palimpsest.cli import app


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td) / "p.db"


def _run(runner: CliRunner, db: Path, *args: str):
    return runner.invoke(app, [args[0], "--db", str(db), *args[1:]])


def test_cli_write_and_read(db_path):
    runner = CliRunner()
    w = _run(runner, db_path, "write", "User lives in Berlin", "--subject", "user")
    assert w.exit_code == 0, w.stdout
    assert "wrote" in w.stdout

    r = _run(runner, db_path, "read", "where does the user live", "--subject", "user", "--k", "3")
    assert r.exit_code == 0, r.stdout
    assert "Berlin" in r.stdout


def test_cli_stats_outputs_valid_json(db_path):
    runner = CliRunner()
    _run(runner, db_path, "write", "User likes coffee", "--subject", "user")
    res = _run(runner, db_path, "stats")
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout.strip())
    assert payload["atoms_total"] == 1
    assert payload["atoms_active"] == 1


def test_cli_lineage_walks_supersedes(db_path):
    runner = CliRunner()
    # Two contradicting writes auto-supersede via the engine.
    a = _run(runner, db_path, "write", "User lives in Berlin", "--subject", "user")
    b = _run(runner, db_path, "write", "User lives in Munich", "--subject", "user")
    assert a.exit_code == 0 and b.exit_code == 0

    # Pull the most recent atom id from `read`.
    r = _run(runner, db_path, "read", "where does user live", "--subject", "user", "--k", "5")
    assert r.exit_code == 0
    # We can't easily extract atom ids from the rich table, but lineage of the
    # active "Munich" atom should report 2 entries (Berlin -> Munich).
    from palimpsest import Memory
    mem = Memory.open(db_path)
    actives = mem.list_subject("user", status="active")
    mem.close()
    assert len(actives) == 1, actives
    munich_id = actives[0].id

    out = _run(runner, db_path, "lineage", str(munich_id))
    assert out.exit_code == 0, out.stdout
    # Two lines: Berlin (superseded) then Munich (active)
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    assert len(lines) == 2
    assert "superseded" in lines[0]
    assert "Munich" in lines[1]


def test_cli_read_with_no_hits(db_path):
    runner = CliRunner()
    res = _run(runner, db_path, "read", "anything", "--k", "5")
    assert res.exit_code == 0
    assert "no hits" in res.stdout
