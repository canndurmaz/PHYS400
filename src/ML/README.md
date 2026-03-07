# Machine Learning (ML) Subfolder

This folder contains machine learning tools for predicting alloy elastic properties.

## Contents

### 1. Alloy Elastic Property Predictor (`nn_alloy.py`)
A TensorFlow neural network trained on MD simulation results.
- **Workflow**: Reads `results.json`, trains a 3-layer MLP, saves the model to `alloy_model.keras`, and generates predictions for alloys in `predict.json`.
- **Run**: `./run_nn.sh` (or `python nn_alloy.py`)

### 2. Standalone Predictor (`predict_from_model.py`)
Loads the saved `.keras` model to instantly predict properties for any composition JSON.
- **Run**: `python predict_from_model.py path/to/composition.json`

### 3. MNIST Example (`nn_example.py`)
A baseline script to verify the TensorFlow installation using the MNIST dataset.

## Data Files

- **`results.json`**: The ground-truth dataset. This is automatically updated by the MD module (`src/MD/lmp.py`) with calculated $E$ and $\nu$ values.
- **`predict.json`**: A batch input file for testing the model on novel randomized or custom compositions.
- **`alloy_model.keras`**: The persistent trained weights of the neural network.

## Feature Mapping (Element Indices)
Compositions are converted into 10-dimensional feature vectors using this order:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Symbol** | Al | Co | Cr | Cu | Fe | Mg | Mn | Ni | Ti | Zn |

## Dependencies
- TensorFlow
- NumPy
- Matplotlib (optional)
