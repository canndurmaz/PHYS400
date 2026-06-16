#!/usr/bin/env bash
# Launch the prediction UIs:
#   src/ML/app.py    -> http://127.0.0.1:5000  (NN surrogate)
#   src/MEAM/app.py  -> http://127.0.0.1:5001  (on-demand LAMMPS MD)
#   src/inv/app.py   -> http://127.0.0.1:5002  (inverse design: (E,ν) -> composition)
# Ctrl-C stops all.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/phys/bin/python3"

cleanup() {
    trap - INT TERM
    kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

(cd "$ROOT/src/ML"   && "$PY" app.py) &
(cd "$ROOT/src/MEAM" && "$PY" app.py) &
(cd "$ROOT/src/inv"  && "$PY" app.py) &

echo "ML   app: http://127.0.0.1:5000"
echo "MEAM app: http://127.0.0.1:5001"
echo "inv  app: http://127.0.0.1:5002"
wait
