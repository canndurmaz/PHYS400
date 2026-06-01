#!/usr/bin/env bash
# Launch the MEAM Flask app on http://127.0.0.1:5001
set -euo pipefail
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
cd "$(dirname "$0")"
exec /home/kenobi/Workspaces/PHYS400/phys/bin/python3 app.py
