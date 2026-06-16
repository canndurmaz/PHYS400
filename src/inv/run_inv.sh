#!/bin/bash
# Train the inverse-design ensemble. Mirrors src/ML/run_nn.sh.
#
# Requires the forward surrogate (src/ML/alloy_model_*.keras) to exist --
# it is loaded frozen for the cycle loss. Train it first with src/ML/run_nn.sh
# if you haven't.
set -e
cd "$(dirname "$0")"

PYTHON_VENV="../../phys/bin/python3"

# TF CUDA wheels (mirrors run_nn.sh): add nvidia/*/lib to the linker path
# when present; silent no-op on a CPU-only install.
_NV_LIB_GLOB=../../phys/lib/python3.12/site-packages/nvidia/*/lib
if compgen -G "$_NV_LIB_GLOB" > /dev/null; then
    _NV_LIBS=$(printf '%s:' $_NV_LIB_GLOB)
    export LD_LIBRARY_PATH="${_NV_LIBS%:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Deep ensemble: 5 independently-trained members give diverse candidate
# compositions for the same target. Override with ENSEMBLE_SIZE=1 ./run_inv.sh.
export ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-5}"

echo "Training inverse-design ensemble of $ENSEMBLE_SIZE members..."
$PYTHON_VENV inv_design.py
