#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
/home/kenobi/Workspaces/PHYS400/phys/bin/python "$SCRIPT_DIR/generate.py" "$@"
