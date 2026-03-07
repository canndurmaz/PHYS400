#!/bin/bash

# Path to the virtual environment python
PYTHON_VENV="../../phys/bin/python3"

# Ensure we are in the script's directory so relative paths work
cd "$(dirname "$0")"

echo "Training Alloy Neural Network and predicting for predict.json..."
$PYTHON_VENV nn_alloy.py
