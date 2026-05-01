#!/bin/bash
# Regenerate every figure / LaTeX-table asset for reports/interim/.
# Each Python module is independent; running them all is safe.
set -euo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-../../phys/bin/python3}"

echo "== dataset_stats =="
"$PY" dataset_stats.py
echo
echo "== dft_stats =="
"$PY" dft_stats.py
echo
echo "== meam_init_stats =="
"$PY" meam_init_stats.py
echo
echo "All report assets refreshed."
