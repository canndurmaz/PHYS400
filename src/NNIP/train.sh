#!/bin/bash
# Train the DeePMD neural network interatomic potential.
#
# Prerequisites:
#   1. DFT data generated and converted to DeePMD format
#   2. DeePMD-kit installed (run install_deepmd.sh)
#
# Usage: bash src/NNIP/train.sh

set -euo pipefail

VENV="/home/kenobi/Workspaces/PHYS400/phys"
PROJECT="/home/kenobi/Workspaces/PHYS400"
INPUT="$PROJECT/src/NNIP/input.json"
MODEL_DIR="$PROJECT/models"

source "$VENV/bin/activate"

echo "=========================================="
echo "DeePMD-kit Training"
echo "=========================================="

# Check prerequisites
if ! command -v dp &> /dev/null; then
    echo "ERROR: dp not found. Run install_deepmd.sh first."
    exit 1
fi

if [ ! -d "$PROJECT/data/training/deepmd" ]; then
    echo "ERROR: Training data not found at data/training/deepmd/"
    echo "Run the data conversion step first."
    exit 1
fi

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# Train
echo ""
echo "[1] Training..."
dp train "$INPUT" 2>&1 | tee train.log

# Freeze model
echo ""
echo "[2] Freezing model..."
dp freeze -o model.pb

echo ""
echo "[3] Model saved to: $MODEL_DIR/model.pb"
echo "    Training log: $MODEL_DIR/train.log"
echo "    Learning curve: $MODEL_DIR/lcurve.out"

# Quick check
ls -lh "$MODEL_DIR/model.pb"

echo ""
echo "=========================================="
echo "Training complete"
echo "=========================================="
