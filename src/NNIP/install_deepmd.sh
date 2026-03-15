#!/bin/bash
# Install DeePMD-kit in the phys virtual environment.
#
# DeePMD-kit provides:
#   - dp CLI for training/freezing NN potentials
#   - pair_style deepmd for LAMMPS (via plugin or rebuilt LAMMPS)
#
# Usage: bash src/NNIP/install_deepmd.sh

set -euo pipefail

VENV="/home/kenobi/Workspaces/PHYS400/phys"

echo "=========================================="
echo "DeePMD-kit Installation"
echo "=========================================="

# Activate venv
source "$VENV/bin/activate"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Install deepmd-kit with TensorFlow backend
echo ""
echo "[1] Installing deepmd-kit..."
pip install --upgrade deepmd-kit

# Verify dp CLI
echo ""
echo "[2] Verifying dp CLI..."
if command -v dp &> /dev/null; then
    echo "dp version: $(dp --version 2>&1 || echo 'installed')"
    echo "dp location: $(which dp)"
else
    echo "WARNING: dp command not found in PATH"
    echo "Try: python -m deepmd --version"
fi

# Check if LAMMPS plugin is available
echo ""
echo "[3] Checking LAMMPS integration..."
echo "Note: The system LAMMPS (20240207) may need to be rebuilt with DeePMD plugin."
echo "Options:"
echo "  a) Use deepmd-kit's built-in LAMMPS: pip install deepmd-kit[lmp]"
echo "  b) Rebuild system LAMMPS with -D PKG_PLUGIN=ON and DeePMD plugin"
echo "  c) Use ASE calculator wrapper (calculator.py) as fallback"

# Try to check if pair_style deepmd is available
python -c "
from lammps import lammps
L = lammps(cmdargs=['-log', 'none', '-screen', 'none'])
try:
    L.command('pair_style deepmd')
    print('pair_style deepmd: AVAILABLE')
except:
    print('pair_style deepmd: NOT available in current LAMMPS build')
    print('Will use ASE calculator wrapper as fallback')
L.close()
" 2>/dev/null || echo "Could not test LAMMPS pair_style"

echo ""
echo "=========================================="
echo "Installation complete"
echo "=========================================="
