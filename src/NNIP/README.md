# NNIP: Neural Network Interatomic Potential Pipeline

DFT-driven N-element MEAM potential generation and optimization. Select elements, run ab-initio reference calculations, initialize MEAM parameters from first principles, then optimize via a neural network surrogate trained against experimental alloy data.

---

## Pipeline Architecture

```
GUI: Select N elements from pseudopotentials/
  |
Stage 1: dft_reference.py
  QE SCF for N elements + all binary pairs
  -> dft_results.json
  |
Stage 2: dft_to_meam.py
  Overlay DFT values onto merged MEAM base files
  -> initial MEAM library + params files
  |
Stage 3: nn_optimizer.py
  Train NN surrogate using src/ML/results.json
  (composition, E_GPa, nu) as training targets
  loss < 5 or 10000 epochs
  -> optimized MEAM files in EAM/optimized/
  |
Stage 4: verify
  Validate optimized potential against known properties
```

---

## Quick Start

```bash
# Full pipeline with GUI element selection
./src/NNIP/run_pipeline.sh

# Specify elements directly (skips GUI)
./src/NNIP/run_pipeline.sh Al Cu Zn Mg

# Skip DFT (use existing dft_results.json), 50 NN samples
./src/NNIP/run_pipeline.sh --skip-dft --samples 50 Al Cu Zn Mg

# Only run DFT + MEAM init, no optimization
./src/NNIP/run_pipeline.sh --skip-optimize --skip-verify Al Cu
```

Or call `pipeline.py` directly:

```bash
source phys/bin/activate
python src/NNIP/pipeline.py --elements Al Cu Zn Mg --samples 30
python src/NNIP/pipeline.py --skip-dft --elements Al Cu Zn Mg
python src/NNIP/pipeline.py --help
```

---

## Stage Details

### Stage 0: Element Selection (`element_selector.py`)

Web GUI on `http://127.0.0.1:8472`. Scans `pseudopotentials/` for available UPF files and shows clickable element cards. Returns sorted list of selected symbols.

**Available elements** (depends on pseudopotentials/):
Al, Au, B, C, Cr, Cu, Fe, H, Mg, Mn, Mo, N, O, Rh, S, Si, Ti, Zn

**Bypass GUI**: use `--elements Al Cu Fe` on the CLI.

### Stage 1: DFT Reference (`dft_reference.py`)

Runs Quantum Espresso (pw.x) via ASE for each selected element:

1. **EOS fit** (7 strain points, Birch-Murnaghan) -> equilibrium lattice constant `a0`, cohesive energy `E_coh`, bulk modulus `B`
2. **Elastic constants** (stress-strain, cubic elements only) -> `C11`, `C12`
3. **Binary formation energies** (all N*(N-1)/2 pairs) -> `E_form` per pair

**Binary reference structures** (chosen by lattice type combination):

| Pair lattices     | Structure | Example        |
|-------------------|-----------|----------------|
| fcc + fcc         | L1_2      | Cu3Au          |
| bcc + bcc         | B2        | FeCr           |
| fcc + bcc         | B2        | FeAl, CuZn     |
| fcc + hcp         | L1_2      | Al3Mg, Al3Ti   |
| bcc + hcp         | B2        | TiFe           |
| fcc + diamond     | L1_2      | Al3Si          |
| bcc + diamond     | B2        | FeSi           |
| hcp + hcp         | B2        | MgZn           |
| diamond + hcp     | L1_2      | SiTi           |
| diamond + diamond | B2        | (fallback)     |

**Magnetic elements** (Fe, Cr, Mn) use `nspin=2`.

**QE settings**: ecutwfc=40 Ry, ecutrho=320 Ry, kpts=(6,6,6), Marzari-Vanderbilt smearing.

**Output**: `src/NNIP/dft_results.json` — saved incrementally after each element and pair so progress is not lost on failure.

**Scratch files**: `src/NNIP/dft_scratch/<Element>/eos_0..6/`, `elastic_base/`, `elastic_strain/`, `binary_X_Y/`

**Standalone usage**:
```bash
python src/NNIP/dft_reference.py Al Cu Fe --output my_results.json
```

### Stage 2: DFT-to-MEAM Initialization (`dft_to_meam.py`)

Overlays DFT reference values onto merged MEAM base files:

- **esub** (cohesive energy) <- DFT `E_coh`
- **alat** (lattice constant) <- DFT `a0`
- **alpha** (Rose equation parameter) <- computed from `E_coh`, `B`, atomic volume
- **Ec(i,j)** (cross-term cohesive energy) <- average of pure `esub` + DFT `E_form`
- **re(i,j)** (cross-term equilibrium distance) <- average nearest-neighbor distances
- **alpha(i,j)** <- average of pure alphas
- All other parameters retain base values from the merged potential

**Input**: DFT results + existing merged MEAM files in `EAM/`

**Output**: `EAM/dft_initialized/library_<elements>.meam` + `<elements>.meam`

**Standalone usage**:
```bash
python src/NNIP/dft_to_meam.py \
    --dft-results src/NNIP/dft_results.json \
    --elements Al Cu Zn Mg \
    --eam-dir EAM/
```

### Stage 3: NN Optimization (`nn_optimizer.py`)

Trains a neural network surrogate model on the mapping `MEAM_params -> (E, nu)` for all alloy compositions in `src/ML/results.json`, then performs inverse optimization.

**Training targets** (from `src/ML/results.json`):

| Config          | Composition         | E (GPa) | nu    |
|-----------------|---------------------|---------|-------|
| AL7075_simple   | Al91/Mg2.9/Zn6.1    | -70.46  | 0.406 |
| AL5052_simple   | Al97.5/Mg2.5        | 83.25   | 0.311 |
| AL2024_noMg     | Al95.1/Cu4.4/Mn0.5  | 67.78   | 0.342 |
| AL2219_simple   | Al93.7/Cu6.3        | 65.38   | 0.330 |
| manual_config   | Cu50/Al50           | 92.15   | 0.296 |

**Workflow**:
1. **Sample**: Random perturbations of initial MEAM vector, evaluate LAMMPS for each composition
2. **Train NN**: Input(params) -> Dense(64) -> Dense(64) -> Dense(32) -> Output(N_entries x 2)
3. **Early stop**: normalized loss < 5.0 or 10,000 epochs
4. **Inverse design**: gradient descent through trained NN to minimize distance to all targets
5. **Validate**: final LAMMPS run with optimized parameters

**Output**: `EAM/optimized/optimized_library_<name>.meam` + `optimized_<name>.meam`

**Standalone usage**:
```bash
python src/NNIP/nn_optimizer.py \
    --library EAM/dft_initialized/library_AlCuZnMg.meam \
    --params  EAM/dft_initialized/AlCuZnMg.meam \
    --samples 50
```

### Stage 4: Verification (`verify_7075.py`)

Validates optimized potential against results.json targets by running LAMMPS for each composition and comparing (E, nu). Reports per-entry and average percentage errors.

**Pass criteria**: average error < 10% for both E and nu.

**Standalone usage**:
```bash
python src/NNIP/verify_7075.py
```

---

## Supporting Modules

### `meam_io.py` — MEAM File I/O

- `parse_library(path)` -> dict of element data (header + 14 params)
- `parse_params(path)` -> ordered list of (key, value) tuples
- `write_library(lib_data, element_order, path)`
- `write_params(entries, path)`
- `params_to_vector(lib_data, param_entries, opt_spec)` -> (numpy array, names)
- `vector_to_files(vec, names, base_lib, base_params, out_dir)` -> (lib_path, params_path)

### `merge_potentials.py` — MEAM Potential Merger

Combines multiple source MEAM files and literature data into a single multi-element potential. Handles index remapping, missing cross-terms via mixing rules, and proper ordering.

```bash
python src/NNIP/merge_potentials.py --config src/configs/meam_merge_7075.json
```

### `reference_data.py` — Config Loader

Loads `config_optimize.json` for reference targets, weights, and bounds used by the optimizer.

---

## File Structure

```
src/NNIP/
  README.md              # This file
  __init__.py
  pipeline.py            # Pipeline orchestrator (main entry point)
  run_pipeline.sh        # Shell wrapper with logging
  element_selector.py    # Web GUI for element selection
  dft_reference.py       # DFT calculations (QE + ASE)
  dft_to_meam.py         # DFT-to-MEAM parameter initialization
  nn_optimizer.py        # NN surrogate optimization
  verify_7075.py         # Validation against known properties
  meam_io.py             # MEAM file parsing/writing
  merge_potentials.py    # Multi-element potential merger
  reference_data.py      # Config/reference data loader
  config_optimize.json   # Optimization bounds and weights
  dft_results.json       # DFT output (generated)
  dft_scratch/           # QE scratch files (generated)
  tmp_nn/                # NN temp files (generated)
  logs/                  # Pipeline log files (generated)
```

---

## Output Locations

| Stage | Output | Location |
|-------|--------|----------|
| DFT reference | Per-element and binary pair data | `src/NNIP/dft_results.json` |
| DFT scratch | QE input/output files | `src/NNIP/dft_scratch/` |
| MEAM init | DFT-initialized potentials | `EAM/dft_initialized/` |
| NN optimization | Optimized potentials | `EAM/optimized/` |
| Pipeline logs | Timestamped run logs | `src/NNIP/logs/` |

---

## Prerequisites

- **Python 3.12** with venv at `phys/` (created with `--system-site-packages`)
- **Quantum Espresso 7.5** — `pw.x` at `/home/kenobi/Workspaces/qe/bin/pw.x`
- **LAMMPS** with MEAM support and `maxelt >= 20`
- **TensorFlow** for NN surrogate
- **ASE** for Espresso calculator interface
- **Pseudopotentials** in `pseudopotentials/` directory

---

## Troubleshooting

### "Too many elements extracted from MEAM library"
LAMMPS compiled with low `maxelt` limit (default 5). Use a custom build with `maxelt=20`:
```bash
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

### DFT calculations hang or fail
- Check that `pw.x` is accessible: `which pw.x`
- Check pseudopotential files exist: `ls pseudopotentials/`
- Intermediate results are saved after each element — restart with `--skip-dft` if partial results are sufficient

### NN optimization fails to converge
- Increase `--samples` (more data for the surrogate)
- Ensure initial MEAM files are stable (test with `verify_7075.py` first)
- Check that target properties in `results.json` are physically reachable

### GUI doesn't respond after selection
- Check terminal for `[POST /select]` and `[SHUTDOWN]` log messages
- If the server doesn't shut down, Ctrl+C and use `--elements` CLI flag instead
