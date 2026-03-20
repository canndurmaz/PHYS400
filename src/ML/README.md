# Machine Learning (ML) Subfolder

This folder contains machine learning tools for predicting alloy elastic properties. It serves as both a standalone prediction tool and the data bridge between MD simulations and the NNIP optimization pipeline.

## Role in the Pipeline

```
Config Generation → MD Simulation → [results.json] → DFT + NN Optimization
                                          ↓
                                     This module
                                   (train / predict)
```

- **`results.json`** is populated by the MD module (`src/MD/lmp.py`) with computed E and ν for each alloy config
- The NNIP pipeline (`src/NNIP/`) reads `results.json` as training targets for NN surrogate optimization
- This module can also train a standalone predictor on the same data

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

- **`results.json`**: The ground-truth dataset. Automatically populated by MD simulations (`src/MD/lmp.py`) with calculated E and ν values. Also used as training targets by the NNIP optimization pipeline.
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
