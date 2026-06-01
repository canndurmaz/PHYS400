"""Integration smoke test for src/MEAM/runner.py.

Skipped by default. Enable with:
    RUN_LAMMPS_TESTS=1 LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH \
        ./phys/bin/python3 -m pytest tests/meam/test_runner_smoke.py -v -s

Wall time on an idle host is ~10-15 s. The test budgets up to 300 s so it
still passes under heavy concurrent load (the NNIP DFT pipeline can saturate
all cores with pw.x workers — LAMMPS then runs an order of magnitude slower).
"""

import json
import math
import os
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LAMMPS_TESTS") != "1",
    reason="set RUN_LAMMPS_TESTS=1 to run LAMMPS-touching tests",
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNNER = os.path.join(_REPO, "src", "MEAM", "runner.py")
# The venv is in the main repo, not in the worktree
_MAIN_REPO = os.path.abspath(os.path.join(_REPO, "..", "..")) if ".worktrees" in _REPO else _REPO
_PY = os.path.join(_MAIN_REPO, "phys", "bin", "python3")


def _last_json_line(stdout: str, type_: str) -> dict | None:
    for raw in reversed(stdout.strip().splitlines()):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == type_:
            return obj
    return None


def test_runner_returns_result_for_pure_al():
    spec = {
        # 2x2x2 FCC supercell of pure Al — 32 atoms, the smallest box that
        # still gives a stable strain sweep. Keeps wall time minimal under
        # the test's load-tolerant 300 s budget.
        "composition": {"Al": 1.0},
        "knobs": {
            "box_size_m": 8e-10, "temperature": 300.0,
            "total_steps": 50, "thermo_interval": 10, "dump_interval": 50,
        },
        "do_viz": False,
        "render_output_path": None,
    }
    proc = subprocess.run(
        [_PY, _RUNNER],
        input=json.dumps(spec),
        capture_output=True,
        text=True,
        env={**os.environ,
             "LD_LIBRARY_PATH": os.path.expanduser("~/.local/lib")
                                + ":" + os.environ.get("LD_LIBRARY_PATH", "")},
        timeout=300,
    )
    assert proc.returncode == 0, f"runner failed: {proc.stderr}\n---\n{proc.stdout}"
    result = _last_json_line(proc.stdout, "result")
    assert result is not None, f"no result line in:\n{proc.stdout}"
    for k in ("C11_GPa", "C12_GPa", "E_GPa", "nu"):
        assert math.isfinite(result[k]), f"{k} non-finite: {result[k]}"
    # Al-FCC ballpark sanity (very loose — MEAM not DFT)
    assert 40 < result["C11_GPa"] < 250
    assert 0 < result["nu"] < 0.5
