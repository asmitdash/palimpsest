"""Sanity-check that bundled examples actually run without raising.

Currently just examples/agent_loop.py — adding more example scripts later
extends here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_agent_loop_example_runs_cleanly():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "agent_loop.py"
    assert script.is_file(), script

    # Repo is installed in editable mode in CI, but during local pytest the
    # subprocess inherits a fresh sys.path. Make the package importable by
    # putting the repo root on PYTHONPATH for the child.
    env = {**os.environ,
           "PALIMPSEST_LLM_PROVIDER": "stub",
           "PALIMPSEST_EMBEDDING_PROVIDER": "stub",
           "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo_root),
        env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    # Spot-check the demo output mentions both lineage states.
    assert "Berlin" in proc.stdout
    assert "Munich" in proc.stdout
