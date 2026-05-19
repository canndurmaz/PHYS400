# PHYS400 — Computational Materials Science

End-to-end pipeline for developing interatomic potentials for metallic alloys: random **config generation**, **LAMMPS** molecular dynamics for elastic properties, **DFT** (Quantum Espresso) for reference data, and **Neural Network** surrogate optimization via **TensorFlow**.

---

## 1. Installation & Environment Setup

### Python Environment
```bash
python3 -m venv --system-site-packages phys
source phys/bin/activate
pip install lammps ovito numpy tensorflow matplotlib ase mpi4py
```

#### Optional: GPU acceleration for the ML stage

If you have an NVIDIA GPU (driver ≥ 525 for CUDA 12), install TensorFlow's
bundled CUDA wheels — this pulls matching `libcudart` / `libcudnn` as pip
packages, so no system CUDA toolkit is required:

```bash
phys/bin/python3 -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0'
```

Verify with `phys/bin/python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
If no GPU is detected, the ML stage transparently falls back to CPU (the
composition surrogate is small enough that CPU training is fine).

### Custom LAMMPS Build
The default LAMMPS build limits MEAM potentials to 5 elements. Rebuild with `maxelt=20` to support multi-element alloys:

```bash
git clone -b patch_7Feb2024 --depth 1 https://github.com/lammps/lammps.git
cd lammps/src
sed -i 's/maxelt = 5/maxelt = 20/' MEAM/meam.h
cd .. && mkdir build && cd build
cmake ../cmake -D BUILD_SHARED_LIBS=yes \
               -D PKG_MEAM=yes \
               -D CMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc) && make install
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Quantum Espresso (for DFT stage)
See [src/QE/README.md](src/QE/README.md) for build instructions. QE 7.5 with MPI support.

---

## 2. Project Structure

```
PHYS400/
├── EAM/                # MEAM potential files (base, merged, optimized)
│   ├── dft_initialized/  # MEAM init output
│   └── optimized/        # NN-optimized output
├── pseudopotentials/   # QE pseudopotential files (*.UPF, auto-downloaded)
├── reports/interim/    # LaTeX interim report (compile.sh auto-refreshes
│                       # tables/figures via src/stats/run_all.sh)
├── src/
│   ├── configs/        # Alloy composition configs (.json) + generator
│   ├── MD/             # Molecular Dynamics module (LAMMPS + OVITO)
│   ├── ML/             # Machine Learning module (TensorFlow)
│   ├── NNIP/           # Full pipeline orchestrator (DFT + NN optimisation)
│   ├── QE/             # Quantum Espresso tests
│   └── stats/          # Report-asset generators (figures + LaTeX tables
│                       # via tabulate; consumed by reports/interim/compile.sh)
├── phys/               # Python virtual environment
├── clean.sh            # Wipe build artefacts (preserves results.json
│                       # + dft_results.json + sources)
└── README.md
```

---

## 3. Pipeline Workflow

The pipeline follows four stages, each feeding into the next:

```
  Stage 1: Config Generation
          |  Random alloy compositions from available MEAM elements
          |  -> src/configs/*.json
          v
  Stage 2: MD Simulation (LAMMPS)
          |  Young's modulus (E) and Poisson's ratio (nu) per config
          |  -> src/ML/results.json
          v
  Stage 3: DFT Reference (Quantum Espresso)
          |  EOS fits, elastic constants, formation energies
          |  Pseudopotentials auto-downloaded if missing
          |  -> src/NNIP/dft_results.json
          v
  Stage 4: NN Surrogate Optimization
          |  Train NN on LAMMPS evaluations, inverse-optimize MEAM params
          |  -> EAM/optimized/
```

### A. Config Generation
Generate random alloy compositions with elements covered by available MEAM potentials.

```bash
# Generate 1000 random configs (default)
./src/configs/generate.sh

# Generate a specific number
./src/configs/generate.sh --samples 500

# Remove generated configs (keeps manually created ones like AL7075_simple.json)
./src/configs/clean.sh

# Preview what would be deleted
./src/configs/clean.sh --dry-run
```

Compositions use 1e-4 precision, Al is always dominant when present, and fractions sum to exactly 1.0. See existing examples in `src/configs/` (e.g., `AL7075_simple.json`).

### B. MD Simulation & Elastic Properties
Run LAMMPS simulations to calculate Young's Modulus and Poisson's Ratio for each config.

```bash
# Run a specific config
./src/MD/run.sh src/configs/AL7075_simple.json

# With visualization
./src/MD/run.sh --viz src/configs/AL7075_simple.json

# Configure interactively via web GUI
python3 src/MD/gui.py
```

Results are appended to `src/ML/results.json` automatically.

### C. DFT Reference Calculations
Compute quantum-mechanical reference data for MEAM parameter initialization. Pseudopotentials are auto-downloaded from the QE library if missing.

```bash
cd src/NNIP

# Run DFT for specific elements
./run_pipeline.sh --skip-optimize --skip-verify Al Cu Zn Mg

# Or auto-discover all elements from EAM/ library files
./run_pipeline.sh --skip-optimize --skip-verify
```

### D. NN Optimization
Train a neural network surrogate on LAMMPS evaluations, then inverse-optimize MEAM parameters to match target elastic properties.

```bash
cd src/NNIP

# Full pipeline (DFT + NN optimization + verification)
./run_pipeline.sh Al Cu Zn Mg Fe Cr Mn Si Ti

# Skip DFT (reuse existing results), tune perturbations and parallelism
./run_pipeline.sh --skip-dft --perturbations 50 --parallel 6 Al Cu Zn Mg

# Use active learning instead of random Phase-1 sampling (~3× fewer LAMMPS calls
# for the same surrogate quality)
./run_pipeline.sh --skip-dft --sampling active \
    --perturbations 60 --batch-size 6 --parallel 6 Al Cu Zn Mg
```

See [src/NNIP/README.md](src/NNIP/README.md) for full pipeline documentation.

### E. ML Training & Prediction (Standalone)
Train a neural network directly on MD results, or predict for new compositions. The composition surrogate predicts $(C_{11}, C_{12})$ and derives $(E, \nu)$ analytically (Huber loss). Re-running `run_nn.sh` automatically refreshes the corresponding figures and metrics table in the interim report.

```bash
./src/ML/run_nn.sh
./phys/bin/python3 src/ML/predict_from_model.py path/to/composition.json
```

### F. Interim Report

Compile the LaTeX interim report. `compile.sh` runs `src/stats/run_all.sh` first, which regenerates every auto-derived figure (element coverage, $E$–$\nu$ scatter coloured by dominant element, DFT formation-energy heatmap, Stage 5 verification bars, …) and every auto-derived LaTeX table (dataset stats, DFT elements, MEAM init, NN validation, pipeline timings, …) from the canonical project data files.

```bash
cd reports/interim && bash compile.sh   # regenerates assets, then 4 pdflatex passes
```

### G. Maintenance

```bash
./clean.sh --dry-run           # preview every artefact that would be removed
./clean.sh                     # interactive (asks Y/N)
./clean.sh -y                  # silent

python3 src/MD/fill_results_cij.py   # one-shot: add C11,C12 to legacy results.json entries
python3 src/NNIP/fill_elastic_constants.py --force   # one-shot: fill missing/unphysical C_ij in dft_results.json
```

`clean.sh` preserves `src/ML/results.json`, `src/ML/predict.json`, and `src/NNIP/dft_results.json`; everything else (PDFs, plots, model checkpoints, scratch dirs, `__pycache__`) is regenerable.

---

## 4. Element Support

Elements are auto-discovered from MEAM library files in `EAM/`. Pseudopotentials for DFT are auto-downloaded when missing. Base MEAM files are auto-merged when no single file covers all selected elements.

Currently available across all library files:

| Al | Co | Cr | Cu | Fe | Mg | Mn | Mo | Ni | Si | Ti | Zn |
|---|---|---|---|---|---|---|---|---|---|---|---|

---

## 5. Parallelism

| Level | Mechanism | Flag |
|-------|-----------|------|
| **MPI** | Distribute samples across ranks | `mpirun -np N` |
| **Multiprocessing** | Parallel LAMMPS / DFT workers per rank | `--parallel N` |
| **OpenMP** | LAMMPS internal thread parallelism | Auto: `cpu_count / (workers × ranks)` |

---

## 6. Modules Documentation
- [**NNIP Pipeline**](src/NNIP/README.md): Full Config→MD→DFT→NN pipeline, auto-merge, verification, $C_{ij}$+Huber surrogate
- [**MD Module**](src/MD/README.md): LAMMPS simulation, parallel elastic property calculation, OVITO visualization, physicality filter
- [**ML Module**](src/ML/README.md): Composition surrogate ($C_{ij}$ targets, Huber loss), prediction, auto-sync to report
- [**QE Module**](src/QE/README.md): Quantum Espresso build, configuration, validation tests
- [**Stats / Report Assets**](src/stats/README.md): Figure + LaTeX-table generators consumed by `reports/interim/compile.sh`
