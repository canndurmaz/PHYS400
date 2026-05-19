# LAMMPS MEAM Simulation — Configurable HEA Alloys

This module is **Stage 2** of the overall pipeline: it takes alloy composition configs (generated in Stage 1 via `src/configs/generate.sh`) and runs LAMMPS MD simulations to compute Young's modulus (E) and Poisson's ratio (ν). Results feed into the DFT and NN optimization stages downstream.

## Prerequisites

- Python 3.12+
- OVITO (installed in venv)
- LAMMPS with MEAM package (`maxelt >= 8`)

## Rebuilding LAMMPS with maxelt=8

The default Ubuntu LAMMPS package has `maxelt=5`, which is too small for the 8-element FeMnNiTiCuCrCoAl MEAM potential. Rebuild from source:

```bash
# 1. Get the source
git clone -b patch_7Feb2024 --depth 1 https://github.com/lammps/lammps.git
cd lammps/src

# 2. Increase maxelt from 5 to 8
sed -i 's/maxelt = 5/maxelt = 8/' MEAM/meam.h

# 3. Build as a shared library
cd ../
mkdir build && cd build
cmake ../cmake -D BUILD_SHARED_LIBS=yes \
               -D PKG_MEAM=yes \
               -D CMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install

# 4. Make the new library available
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Architecture

### `element.py` — Dynamic Potential Parsing

Instead of hardcoding element properties, this module dynamically scans the `EAM/` directory for MEAM potential pairs (`library_*.meam` and `*.meam`).

- **`parse_meam_library(path)`**: Extracts element symbols, lattice types, masses, and lattice constants directly from the MEAM library file.
- **`scan_eam_dir(dir)`**: Identifies all available potentials (e.g., `AlMgZn`, `FeMnNiTiCuCrCoAl`) and their constituent elements.
- **`ELEMENTS`**: A merged lookup dictionary containing all elements discovered across all available potentials.

### `config.py` — Potential Auto-Detection & Shared Config

The single source of truth for simulation parameters. It handles loading configuration JSON files and deriving physical quantities.

- **`find_potential(symbols)`**: Automatically selects the first MEAM potential in `EAM/` that contains all elements requested in the composition.
- **`potential`**: Exports a dictionary containing the paths to the matching `library` and `params` files, along with the required element ordering.
- **Derived Quantities**: Calculates `a_mean` (weighted lattice constant) and `n_repeats` based on the selected alloy.

### `gui.py` — Web Configuration Tool

Provides a modern, browser-based interface to configure simulations without editing JSON manually.
- **Dynamic Elements**: Polls `element.py` to show all elements available in the `EAM/` directory.
- **Interactive Preview**: Features a 3D animated cube that reflects the selected alloy composition.
- **Validation**: Ensures percentages sum to 100% and generates valid `config.json` files.

### `lmp.py` — LAMMPS Simulation

Constructs and executes the simulation using the dynamic potential info:
- **`pair_coeff`**: Auto-generated using the specific library and parameter files detected by `config.py`.
- **Composition Mapping**: Handles the mapping between LAMMPS atom types and MEAM library indices for any supported potential.
- **Elastic Estimation**: Includes `get_elastic_moduli(L)` which calculates and returns the cubic elastic constants `C11`, `C12` together with their derived `(E, ν)` via axial strain perturbations in all 3 directions. The C_ij are persisted alongside `(E, ν)` in `results.json` so the downstream surrogates can read them directly without algebraic reconstruction.
- **Physicality filter**: a config is discarded (and its JSON file deleted) if any of the following hold, matching the filter used by `src/ML/nn_alloy.py` and `src/NNIP/nn_optimizer.py`:
    - `E < 0` (negative Young's modulus)
    - `ν < 0` (negative Poisson's ratio)
    - `ν ≥ 0.48` (close to the `(1−2ν)=0` singularity in the cubic-isotropic algebra)
    - `C11 < C12` (mechanical instability — Cauchy violation)

#### Elastic Property Calculation Method

The simulation employs a **static deformation method** to estimate mechanical properties. This process occurs in several stages:

1. **Ground State Relaxation**: The system undergoes anisotropic pressure minimization (box + atoms) followed by an atom-only re-minimize to reach a stress-free state.
2. **Baseline Measurement**: The diagonal stress components ($\sigma_{xx}$, $\sigma_{yy}$, $\sigma_{zz}$) are recorded as a reference.
3. **3-Direction Strain**: For each direction (x, y, z), a symmetric central-difference strain ($\pm\delta$, where $\delta = 0.001$) is applied. Atoms are re-minimized at each strained geometry to obtain equilibrium stress. This 3-direction averaging reduces noise from local disorder in random alloy supercells.
4. **Stress Response**: For each strain direction, the axial stress derivative gives $C_{11}$ and the transverse stress derivatives give $C_{12}$. Results are averaged (3 $C_{11}$ samples, 6 $C_{12}$ samples).
5. **Analytical Derivation**: Using standard relations for cubic crystals:
   - **Young's Modulus ($E$)**: $E = \frac{(C_{11} - C_{12})(C_{11} + 2C_{12})}{C_{11} + C_{12}}$
   - **Poisson's Ratio ($\nu$)**: $\nu = \frac{C_{12}}{C_{11} + C_{12}}$
6. **Quality Filter**: If $E < 0$, $\nu < 0$, $\nu \geq 0.48$, or $C_{11} < C_{12}$, the result is discarded and the config file is deleted. The $\nu \geq 0.48$ bound matches the singularity guard used by both downstream surrogates so `results.json` only ever contains entries the surrogates can use.
7. **Automatic Integration**: Valid results are appended to `src/ML/results.json` with all four quantities (`E_GPa`, `nu`, `C11_GPa`, `C12_GPa`).

#### Element Mapping and Indices

For both simulation consistency and machine learning feature mapping, the following index order is used for all compositions:

| Index | Element | Symbol |
|---|---|---|
| 0 | Aluminum | Al |
| 1 | Cobalt | Co |
| 2 | Chromium | Cr |
| 3 | Copper | Cu |
| 4 | Iron | Fe |
| 5 | Magnesium | Mg |
| 6 | Manganese | Mn |
| 7 | Nickel | Ni |
| 8 | Titanium | Ti |
| 9 | Zinc | Zn |

### `viz.py` — OVITO Visualization

Renders high-quality animations of the simulation results.
- **Expanded Color Map**: Includes specialized colors for common elements (Fe, Cu, Al, Mg, Zn, etc.).
- **Deterministic Fallback**: Uses a hash-based generator to assign unique, consistent colors to any unknown elements found in new MEAM potentials.

## Pipeline Integration

In the full pipeline workflow:
1. **Config Generation** (`src/configs/generate.sh --samples N`) creates random alloy compositions
2. **MD Simulation** (this module) computes E and ν for each config
3. Results are appended to `src/ML/results.json`, which feeds DFT and NN optimization stages

## Running

### Running a specific config
```bash
./src/MD/run.sh src/configs/AL7075_simple.json

# With visualization
./src/MD/run.sh --viz src/configs/AL7075_simple.json
```

### Running all configs in a directory
```bash
# Run every .json in the configs folder, in parallel by default
./src/MD/run.sh --all src/configs/

# Force the sequential code path (e.g. for one-off debugging)
./src/MD/run.sh --all src/configs/ --jobs 1

# Cap the worker pool explicitly
./src/MD/run.sh --all src/configs/ --jobs 4

# With visualization (forces sequential — OVITO render is per-process)
./src/MD/run.sh --viz --all src/configs/

# With NVT molecular dynamics (forces sequential)
./src/MD/run.sh --simMD --all src/configs/
```

By default the elastic-only batch path runs in parallel via `ProcessPoolExecutor` (`max_workers = cpu_count // 2`, `OMP_NUM_THREADS = 1`). All `results.json` writes happen in the main process so there is no race. Progress is printed as `[i/N]` with elapsed time and ETA. Results are appended to `src/ML/results.json` automatically.

### One-shot fill of `results.json` (`fill_results_cij.py`)
For datasets that pre-date the `C_ij` storage change, run once to add `C11_GPa`/`C12_GPa` to every entry that has `(E, ν)` but no $C_{ij}$ yet:

```bash
python3 src/MD/fill_results_cij.py --dry-run   # preview
python3 src/MD/fill_results_cij.py             # write
```

The fill is idempotent — entries that already have `C_ij` are skipped, and unphysical entries (`ν ≥ 0.49`, mechanically unstable) are left alone.

### Using the GUI
```bash
python3 src/MD/gui.py
```
This opens a web interface. Save your configuration, then run the simulation.
