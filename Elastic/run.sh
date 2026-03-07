#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1

PYTHON=/home/kenobi/Workspaces/PHYS400/phys/bin/python

echo "=== Running elastic tensor calculation ==="
$PYTHON elastic.py

echo "=== Running visualization ==="
$PYTHON viz.py

echo "=== Done ==="
