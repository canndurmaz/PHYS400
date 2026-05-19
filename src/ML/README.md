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
Loads the saved `.keras` model to instantly predict properties for any composition JSON. The model outputs *scaled* $(C_{11}, C_{12})$; this script undoes the `C_SCALE = 200` normalisation and recovers $(E, \nu)$ via the cubic-isotropy identity. The reusable entry point is `predict_properties(composition_dict) -> {"E_GPa", "nu", "C11_GPa", "C12_GPa", "unknown_elements"}`.
- **Run (CLI)**: `python predict_from_model.py path/to/composition.json` — accepts a single `{"composition": {...}}` blob, a bare `{...}` dict, or a `predict.json`-style batch.
- **Run (library)**: `from predict_from_model import predict_properties` (the model is lazy-loaded on first call).

### 3. Web UI (`app.py`)
Flask single-page app that wraps the predictor behind a form: enter mole fractions for any subset of the 10 trained elements and the predicted $(E, \nu, C_{11}, C_{12})$ are returned via `POST /api/predict`. The composition is re-normalised on the server, so non-unit sums are accepted (with a visible warning). Lazy model load — first prediction takes ~2 s, subsequent calls are <100 ms.
- **Run**: `python app.py` &nbsp;→&nbsp; <http://127.0.0.1:5000>
- **Routes**: `GET /` (form) · `POST /api/predict` (JSON in/out)
- **Files**: `templates/index.html`, `static/styles.css`, `static/app.js`
- **Dependency**: Flask (`pip install flask` — already in the project venv).

## Data Files

- **`results.json`**: ground-truth dataset, populated by `src/MD/lmp.py` with `composition`, `E_GPa`, `nu`, `C11_GPa`, `C12_GPa`. Also the training-target source for `src/NNIP/nn_optimizer.py`. The loader prefers the stored $C_{ij}$ when present and falls back to algebraic conversion for older entries.
- **`predict.json`**: batch input file for predicting on novel custom compositions. *Not* a generated artifact — preserved by `clean.sh`.
- **`alloy_model.keras`**: persistent trained weights of the neural network.
- **`nn_metrics.json`**: per-target metrics from the most recent training run.

## Feature Mapping (Element Indices)
Compositions are converted into 12-dimensional feature vectors using this order (alphabetical, matching the full MEAM library coverage):

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Symbol** | Al | Co | Cr | Cu | Fe | Mg | Mn | Mo | Ni | Si | Ti | Zn |

## Dependencies
- TensorFlow
- NumPy
- Flask (for the web UI in `app.py`)
- Matplotlib (optional, for plots)
- `tabulate` (for the auto-generated LaTeX metrics table)

### Optional: NVIDIA GPU acceleration

`nn_alloy.py` runs on CPU by default. If a CUDA-capable GPU is present
(driver ≥ 525 for CUDA 12), install TensorFlow's bundled CUDA wheels:

```bash
phys/bin/python3 -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0'
```

This installs the matching `libcudart` / `libcudnn` as pip packages — no
system CUDA toolkit needed. Confirm detection:

```bash
phys/bin/python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The network is tiny (`10 → 32 → 20 → 2`), so GPU speedup over CPU is
modest (≈2-3× per epoch on a low-end card); CPU is perfectly usable.
If a CUDA-related warning appears at startup *without* the libraries
installed, force CPU explicitly: `export CUDA_VISIBLE_DEVICES=""`.
