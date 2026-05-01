# Machine Learning (ML) Subfolder

This folder contains machine learning tools for predicting alloy elastic properties. It serves as both a standalone prediction tool and the data bridge between MD simulations and the NNIP optimization pipeline.

## Role in the Pipeline

```
Config Generation → MD Simulation → [results.json] → DFT + NN Optimization
                                          ↓
                                     This module
                                  (train / predict;
                                  divergent ML branch
                                  in the report)
```

- **`results.json`** is populated by the MD module (`src/MD/lmp.py`) with `E_GPa`, `nu`, `C11_GPa`, `C12_GPa` for each alloy config
- The NNIP pipeline (`src/NNIP/`) reads `results.json` as training targets for NN surrogate optimization
- This module is presented in the interim report as the divergent ML *sanity-check* branch off Stage 2 — see `reports/interim/sections/composition_nn.tex`

## Contents

### 1. Alloy Elastic Property Predictor (`nn_alloy.py`)
A TensorFlow neural network that maps the 10-element composition vector to the cubic elastic constants $(C_{11}, C_{12})$ and recovers $(E, \nu)$ analytically. Architecture: `10 → 32 → 20 → 2`, trained on a 70/30 split with the **Adam** optimizer and **Huber** loss (robust to a few extreme samples coming out of LAMMPS for Mg/Zn-rich compositions).

**Why $C_{ij}$ targets instead of $(E, \nu)$ directly?** The Poisson ratio $\nu = C_{12}/(C_{11}+C_{12})$ is ill-conditioned in composition space — a small absolute error in either $C_{ij}$ produces a much larger relative error in $\nu$. Predicting $(C_{11}, C_{12})$ is a bijective reparametrisation under cubic isotropy and gives a much smoother regression surface; $(E, \nu)$ are recovered analytically for evaluation and reporting. Records with $\nu \geq 0.48$ are dropped at load time to avoid the $(1-2\nu) \to 0$ singularity in the inverse algebra.

**Stopping gate**: training stops once the *mean* relative error on the derived $(E, \nu)$ over the validation set is within `MAX_ERROR_PCT = 15%` for both `E` and `ν` — i.e. `mean(|ΔE|/E) ≤ 15%` *and* `mean(|Δν|/ν) ≤ 15%`. Per-sample pass/fail counts (using the same 15% threshold) are still reported alongside, but only as diagnostics. Up to `MAX_RESTARTS = 10` rebuilds are attempted, with `BATCH_EPOCHS = 200` epochs per check and a hard cap of `MAX_EPOCHS = 20000` per attempt.

**Performance notes**:
- Inference uses `model(X, training=False)` instead of `model.predict()` (~20× faster on small in-memory arrays — bypasses the `predict()` dataset/batching pipeline).
- Training batch size is `TRAIN_BATCH_SIZE = 128`.
- A status line (with `loss`, `val_loss`, `MAPE`, and `max_err: E=…% nu=…%`) is printed every 100 epochs (plus the first 5).

**Outputs**:
- `alloy_model.keras` — saved TF model.
- `nn_metrics.json` — per-target MAE / RMSE / R² / MAPE for `C11`, `C12`, `E_GPa`, `nu`.
- `plots/training_history.png`, `parity_train.png`, `parity_validation.png`, `error_dist_validation.png` — derived $(E, \nu)$ views (filenames are stable so the report's `\includegraphics` directives don't need changes).
- `plots/parity_train_cij.png`, `parity_validation_cij.png` — direct $C_{ij}$ parity (diagnostics only).

**Report integration**: after training, the script copies all PNGs to `reports/interim/figures/` and writes a complete LaTeX `\begin{table}…` block to `reports/interim/sections/_auto_ml_metrics.tex` (label `tab:nn_aux`) using `tabulate(tablefmt="latex_raw", floatfmt=".3f")`. The report includes the auto file via `\input{sections/_auto_ml_metrics}`, so each rerun of `./run_nn.sh` transparently refreshes the report's table and figures.

**Run**: `./run_nn.sh` (or `python nn_alloy.py`).

### 2. Standalone Predictor (`predict_from_model.py`)
Loads the saved `.keras` model to instantly predict properties for any composition JSON.
- **Run**: `python predict_from_model.py path/to/composition.json`

## Data Files

- **`results.json`**: ground-truth dataset, populated by `src/MD/lmp.py` with `composition`, `E_GPa`, `nu`, `C11_GPa`, `C12_GPa`. Also the training-target source for `src/NNIP/nn_optimizer.py`. The loader prefers the stored $C_{ij}$ when present and falls back to algebraic conversion for older entries.
- **`predict.json`**: batch input file for predicting on novel custom compositions. *Not* a generated artifact — preserved by `clean.sh`.
- **`alloy_model.keras`**: persistent trained weights of the neural network.
- **`nn_metrics.json`**: per-target metrics from the most recent training run.

## Feature Mapping (Element Indices)
Compositions are converted into 10-dimensional feature vectors using this order:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Symbol** | Al | Co | Cr | Cu | Fe | Mg | Mn | Ni | Ti | Zn |

## Dependencies
- TensorFlow
- NumPy
- Matplotlib (optional, for plots)
- `tabulate` (for the auto-generated LaTeX metrics table)
