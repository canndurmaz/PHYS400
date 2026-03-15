# NNIP: Neural Network Interatomic Potential

Train a DeePMD-kit neural network potential for the 9-element Al alloy system:
**Al, Zn, Mg, Mn, Cu, Si, Cr, Fe, Ti**

The goal is to replace empirical MEAM potentials with a DFT-accurate neural network
that can predict energies, forces, and stresses for arbitrary compositions of these
elements. The trained model deploys in LAMMPS via `pair_style deepmd` or as a
standalone ASE calculator.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Pipeline Overview](#pipeline-overview)
4. [Phase 1: Pseudopotentials](#phase-1-download--validate-pseudopotentials)
5. [Phase 2: Training Data](#phase-2-generate-dft-training-data)
6. [Phase 3: Install DeePMD-kit](#phase-3-install-deepmd-kit)
7. [Phase 4: Train the Model](#phase-4-train-the-model)
8. [Phase 5: Validate](#phase-5-validate-the-model)
9. [Phase 6: Deploy](#phase-6-deploy-in-lammps)
10. [Configuration Reference](#configuration-reference)
11. [File Structure](#file-structure)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Activate the virtual environment
source phys/bin/activate

# 1. Download pseudopotentials and run SCF validation
python src/NNIP/download_pseudos.py

# 2. Generate atomic configurations (edit config.json first if needed)
python src/NNIP/generate_configs.py

# 3. Run DFT on all configurations (long — supports resumption)
python src/NNIP/run_dft.py

# 4. Convert results to DeePMD format
python src/NNIP/convert_to_deepmd.py

# 5. Install DeePMD-kit (if not already installed)
pip install deepmd-kit

# 6. Train the neural network
bash src/NNIP/train.sh

# 7. Validate against held-out data
python src/NNIP/validate.py

# 8. Run MD or compute elastic properties
python src/NNIP/run_md.py
python src/NNIP/elastic.py
```

---

## Prerequisites

| Software | Version | Location |
|----------|---------|----------|
| Python | 3.12.3 | system |
| Quantum ESPRESSO | 7.5 | `/home/kenobi/Workspaces/qe/bin/pw.x` |
| ASE | 3.27.0 | venv (`phys`) |
| DeePMD-kit | 3.1.2 | venv (`phys`) |
| TensorFlow | 2.21.0 | venv (`phys`) |
| LAMMPS | 20240207 | system (no DeePMD plugin) |

The virtual environment at `phys/` was created with `--system-site-packages` so it
inherits the system LAMMPS Python bindings. Activate it before running any script:

```bash
source /home/kenobi/Workspaces/PHYS400/phys/bin/activate
```

The venv activation script also adds the QE `bin/` directory to `PATH`.

---

## Pipeline Overview

```
 config.json                    input.json
     │                              │
     ▼                              ▼
┌────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ generate_  │──▶│ run_     │──▶│ convert_ │──▶│ dp train │──▶│ dp freeze│
│ configs.py │   │ dft.py   │   │ deepmd.py│   │          │   │          │
└────────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
  470 XYZ files    QE SCF on      DeePMD raw     400K steps     model.pb
  in configs/      each config    format          training       frozen
                   → dft_results/ → deepmd/       → lcurve.out   graph

                                                        │
                                              ┌─────────┴─────────┐
                                              ▼                   ▼
                                        ┌──────────┐       ┌──────────┐
                                        │validate  │       │ run_md   │
                                        │  .py     │       │ elastic  │
                                        └──────────┘       └──────────┘
                                        parity plots       LAMMPS MD or
                                        MAE metrics        ASE fallback
```

---

## Phase 1: Download & Validate Pseudopotentials

```bash
python src/NNIP/download_pseudos.py
```

Downloads PBE PAW pseudopotentials from `pseudopotentials.quantum-espresso.org` for
all 9 elements into `pseudopotentials/`. Skips files that already exist. After
downloading, runs a single-atom SCF calculation on each element's ground-state bulk
crystal to verify the pseudopotential produces a converged result.

### Type Map

The **type map** defines the integer index ↔ element mapping used everywhere:
in DeePMD training data, the frozen model, and LAMMPS `pair_style deepmd`.
Changing this order requires regenerating all data and retraining.

| Index | Element | Pseudopotential | Bulk Structure | a (A) |
|-------|---------|-----------------|----------------|-------|
| 0 | Al | `Al.pbe-n-kjpaw_psl.1.0.0.UPF` | FCC | 4.05 |
| 1 | Zn | `Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF` | HCP | 2.66 |
| 2 | Mg | `Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF` | HCP | 3.21 |
| 3 | Mn | `Mn.pbe-spn-kjpaw_psl.0.3.1.UPF` | BCC | 8.91 |
| 4 | Cu | `Cu.pbe-dn-kjpaw_psl.1.0.0.UPF` | FCC | 3.61 |
| 5 | Si | `Si.pbe-n-kjpaw_psl.1.0.0.UPF` | Diamond | 5.43 |
| 6 | Cr | `Cr.pbe-spn-kjpaw_psl.1.0.0.UPF` | BCC | 2.91 |
| 7 | Fe | `Fe.pbe-spn-kjpaw_psl.1.0.0.UPF` | BCC | 2.87 |
| 8 | Ti | `Ti.pbe-spn-kjpaw_psl.1.0.0.UPF` | HCP | 2.95 |

All pseudopotentials are PBE exchange-correlation, PAW (Projector Augmented Wave)
type, from the `pslibrary` collection. This ensures consistent DFT accuracy across
the full composition space.

### Validation Details

Each element is tested with a single SCF calculation on its bulk crystal:
- **ecutwfc**: 40 Ry for transition metals (Mn, Fe, Cr, Ti, Cu, Zn), 30 Ry for others
- **ecutrho**: 8 × ecutwfc
- **k-points**: 6×6×6 Monkhorst-Pack
- **Smearing**: Marzari-Vanderbilt (`mv`), degauss = 0.02 Ry
- **Spin polarization**: Enabled for Fe, Cr, Mn (magnetic elements)

A validation pass requires SCF convergence and physically reasonable energy.

---

## Phase 2: Generate DFT Training Data

This phase has three steps: generate configurations, run DFT, convert to DeePMD format.

### Step 2a: Generate Configurations

```bash
python src/NNIP/generate_configs.py
python src/NNIP/generate_configs.py --config my_config.json      # custom config
python src/NNIP/generate_configs.py --output /path/to/output/dir  # custom output
```

Reads `src/NNIP/config.json` and produces extended XYZ files in `data/training/configs/`.
Also writes a `manifest.json` index listing every configuration with its type, atom
count, and element list.

The default config generates **470 configurations** across 5 categories:

| Category | Count | Description |
|----------|-------|-------------|
| `pure` | 81 | 9 elements × (7 volumetric strains + 2 shear strains) |
| `binary` | 216 | C(9,2) = 36 pairs × 3 fractions × 2 supercell sizes |
| `ternary` | 30 | 30 random 3-element compositions |
| `alalloy` | 20 | 10 realistic Al-alloy compositions × 2 supercell sizes |
| `vacancy` | 24 | 9 elements × 2 vacancy counts + 3 alloy vacancies × 2 |
| `rattled` | 99 | 9 elements × 3 temps × 3 snapshots + 2 alloys × 3 × 3 |

Atom counts range from 6 to 54 per configuration.

See [Configuration Reference](#configuration-reference) for full details on
`config.json` fields.

### Step 2b: Run DFT

```bash
python src/NNIP/run_dft.py
```

Runs Quantum ESPRESSO SCF calculations on every configuration in `manifest.json`.
For each configuration it computes total energy, atomic forces, and the stress tensor,
then saves the result as an extended XYZ file in `data/training/dft_results/`.

**Resumption**: The script checks which result files already exist and skips them.
You can safely interrupt with Ctrl-C and restart — it picks up where it left off.
Failed calculations are logged to `data/training/dft_results/failed.json`.

**Runtime**: Each SCF takes 30s–5min depending on atom count and element types.
The full 470-config batch takes roughly 6–24 hours on a single core. Progress is
printed every 50 configs.

**QE Calculator Settings** (set automatically per-configuration):

| Parameter | Value | Notes |
|-----------|-------|-------|
| ecutwfc | 30 or 40 Ry | 40 if any transition metal present |
| ecutrho | 8 × ecutwfc | PAW augmentation charge cutoff |
| k-points | adaptive | ~0.05 A⁻¹ spacing, capped at 6×6×6 |
| smearing | Marzari-Vanderbilt | degauss = 0.02 Ry |
| conv_thr | 1.0e-6 | SCF convergence threshold |
| mixing_beta | 0.3 | Charge mixing for stability |
| electron_maxstep | 200 | Max SCF iterations |
| nspin | 2 | Enabled if Fe, Cr, or Mn present |
| tprnfor / tstress | True | Force and stress output |

**Output format**: Each result file stores:
- `dft_energy` — total energy (eV)
- `dft_energy_per_atom` — energy per atom (eV/atom)
- `dft_forces` — per-atom force vectors (eV/A)
- `dft_stress` — 3×3 stress tensor (eV/A³)

### Step 2c: Convert to DeePMD Format

```bash
python src/NNIP/convert_to_deepmd.py
```

Reads the extended XYZ results and writes DeePMD-kit raw format to
`data/training/deepmd/` and `data/validation/deepmd/`. Configurations are grouped
by atom count (DeePMD requires uniform atom count within each system directory).

Each system directory contains:
```
sys_NNN/
├── type.raw          # Element type index per atom (one integer per line)
├── type_map.raw      # Element names (Al, Zn, Mg, ...)
└── set.000/
    ├── energy.npy    # Total energies, shape (nframes,)
    ├── force.npy     # Forces, shape (nframes, natoms*3)
    ├── coord.npy     # Coordinates, shape (nframes, natoms*3)
    ├── box.npy       # Cell vectors, shape (nframes, 9)
    └── virial.npy    # Virial tensor, shape (nframes, 9)
```

---

## Phase 3: Install DeePMD-kit

```bash
pip install deepmd-kit
# or
bash src/NNIP/install_deepmd.sh
```

Installs the `deepmd-kit` package which provides:
- **`dp` CLI** — training, freezing, testing, compression
- **`deepmd.infer.DeepPot`** — Python inference API
- **TensorFlow backend** — model training

The install script also checks whether the system LAMMPS has `pair_style deepmd`
available. The current LAMMPS build (20240207) does **not** include the DeePMD
plugin. Two options:

1. **Rebuild LAMMPS** with `-D PKG_PLUGIN=ON` and the DeePMD shared library
2. **Use the ASE calculator fallback** (`calculator.py`) — no LAMMPS rebuild needed

Option 2 is used by default in `run_md.py` and `elastic.py`.

---

## Phase 4: Train the Model

```bash
bash src/NNIP/train.sh
```

This runs `dp train` with `src/NNIP/input.json` and then freezes the trained model
to `models/model.pb`. Training outputs:

| File | Description |
|------|-------------|
| `models/model.pb` | Frozen TensorFlow graph for LAMMPS / inference |
| `models/lcurve.out` | Learning curve (energy/force/virial RMSE vs step) |
| `models/train.log` | Full training log |
| `models/model.ckpt.*` | Checkpoints (saved every 10K steps) |

### Model Architecture (`input.json`)

**Descriptor** — `se_e2_a` (Smooth Edition, two-body embedding):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `type` | `se_e2_a` | Standard multi-element descriptor |
| `rcut` | 6.0 A | Interaction cutoff radius |
| `rcut_smth` | 0.5 A | Smooth cutoff onset |
| `sel` | [80] × 9 | Max neighbors per element type within cutoff |
| `neuron` | [25, 50, 100] | Embedding network layer sizes |
| `axis_neuron` | 16 | Embedding axis dimension |
| `resnet_dt` | false | No residual connections in embedding |

The descriptor encodes the local chemical environment of each atom into a
rotationally-invariant feature vector. The `se_e2_a` type considers all pairwise
distances within the cutoff, weighted by element type, and passes them through a
learnable embedding network.

**Fitting Network** — maps descriptor → atomic energy:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `neuron` | [240, 240, 240] | Three hidden layers, 240 neurons each |
| `resnet_dt` | true | Residual connections enabled |

Total energy = sum of atomic energies. Forces and virial are computed as analytical
derivatives of the energy with respect to coordinates and cell vectors.

**Learning Rate** — exponential decay:

| Parameter | Value |
|-----------|-------|
| `start_lr` | 0.001 |
| `stop_lr` | 1e-8 |
| `decay_steps` | 5000 |
| `numb_steps` | 400,000 |

The learning rate decays exponentially from 1e-3 to 1e-8 over 400K steps.

**Loss Function** — weighted sum of energy, force, and virial errors:

| Term | Start Weight | Final Weight | Behavior |
|------|-------------|-------------|----------|
| Energy | 0.02 | 1.0 | Ramps up — focus on forces first |
| Force | 1000 | 1.0 | Ramps down — forces dominate early |
| Virial | 0.02 | 1.0 | Ramps up with energy |

This schedule ensures the model first learns accurate forces (which are abundant
per-atom data), then gradually shifts focus to per-frame energy and virial accuracy.

**Training Data**:

| Parameter | Value |
|-----------|-------|
| `systems` | `data/training/deepmd/` |
| `batch_size` | `auto` (adapts to system size) |
| `disp_freq` | 1000 (print metrics every 1K steps) |
| `save_freq` | 10000 (checkpoint every 10K steps) |

### Modifying Training

Edit `src/NNIP/input.json` directly. Key knobs:

- **More accuracy**: increase `numb_steps`, widen `neuron` layers, increase `rcut`
- **Faster training**: decrease `numb_steps`, narrow layers, decrease `sel`
- **Better force accuracy**: increase `start_pref_f`
- **Different elements**: update `type_map` (must match `type.raw` in training data)

---

## Phase 5: Validate the Model

```bash
python src/NNIP/validate.py
```

Loads the frozen model and evaluates it on held-out validation configurations
(from `data/validation/dft_results/`). Produces:

1. **Energy parity plot** — NN predicted vs DFT energy per atom
2. **Force parity plot** — NN predicted vs DFT force components
3. **Metrics summary** — MAE, RMSE, max error for energy and forces

Output goes to `models/validation_results/`:
```
models/validation_results/
├── energy_parity.png    # Energy parity scatter plot
├── force_parity.png     # Force component parity scatter plot
└── metrics.json         # MAE, RMSE, max error in JSON
```

### Accuracy Targets

| Quantity | Target | Good | Excellent |
|----------|--------|------|-----------|
| Energy MAE | < 5 meV/atom | < 2 meV/atom | < 1 meV/atom |
| Force MAE | < 100 meV/A | < 50 meV/A | < 20 meV/A |

If targets are not met, consider:
- Adding more training data (especially for underrepresented compositions)
- Increasing training steps
- Widening the fitting network
- Checking for outlier configurations in the training set

---

## Phase 6: Deploy in LAMMPS

### MD Simulation

```bash
python src/NNIP/run_md.py
```

Creates a 4×4×4 FCC Al supercell (256 atoms), substitutes 5% with Cu (Al-2024 like),
and runs NVT molecular dynamics at 300K for 10,000 timesteps (10 ps with 1 fs step).

The script automatically detects whether `pair_style deepmd` is available in LAMMPS:
- **If available**: runs native LAMMPS MD with full performance
- **If not**: falls back to ASE Langevin dynamics using `calculator.py`

LAMMPS trajectory is saved to `src/NNIP/md_traj.lammpstrj` (or `.traj` for ASE).

After simulation completes, the trajectory is automatically rendered to an animated
MP4 video using OVITO (`viz.py`). The video is saved to `src/NNIP/visualization/`.

### Visualization

The `viz.py` module provides OVITO-based trajectory rendering with:
- Per-element coloring (consistent color scheme across the project)
- Composition legend overlay (element symbol + atomic fraction)
- Title overlay with alloy designation
- Perspective camera with auto-zoom

**Standalone usage:**

```bash
# Render a LAMMPS trajectory
python src/NNIP/viz.py md_traj.lammpstrj -c "Al:0.95,Cu:0.05"

# Render with custom output path and framerate
python src/NNIP/viz.py md_traj.lammpstrj -c "Al:0.90,Zn:0.06,Mg:0.04" -o output.mp4 --fps 15
```

**From Python:**

```python
from viz import render_lammps, render_ase, render_trajectory

# LAMMPS trajectory (uses TYPE_MAP ordering for atom types)
render_lammps("md_traj.lammpstrj", {"Al": 0.95, "Cu": 0.05})

# ASE trajectory
render_ase("md_traj.traj", {"Al": 0.95, "Cu": 0.05})

# Generic (auto-detect element ordering from composition)
render_trajectory("traj.xyz", {"Al": 0.90, "Zn": 0.06, "Mg": 0.04},
                  output="custom_output.mp4", size=(1920, 1080), fps=30)
```

**Element color scheme:**

| Element | Color | RGB |
|---------|-------|-----|
| Al | Light Gray | (0.80, 0.80, 0.80) |
| Zn | Slate Blue | (0.50, 0.50, 1.00) |
| Mg | Yellow | (1.00, 1.00, 0.00) |
| Mn | Purple | (0.60, 0.00, 0.60) |
| Cu | Orange | (1.00, 0.50, 0.00) |
| Si | Green | (0.00, 0.60, 0.00) |
| Cr | Cyan | (0.00, 0.90, 0.90) |
| Fe | Brown | (0.55, 0.27, 0.07) |
| Ti | Dark Gray | (0.40, 0.40, 0.40) |

Unlisted elements get a deterministic hash-based color.

### Elastic Properties

```bash
python src/NNIP/elastic.py
```

Computes elastic constants (C11, C12), Young's modulus (E), and Poisson's ratio (nu)
via finite-difference strain-stress analysis:

1. Measure baseline stress on relaxed structure
2. Apply small axial strain (+0.1%) in x direction
3. Measure stress response
4. Extract C11 = d(sigma_xx)/d(eps_xx), C12 = d(sigma_yy)/d(eps_xx)
5. Compute E = (C11 - C12)(C11 + 2*C12) / (C11 + C12)
6. Compute nu = C12 / (C11 + C12)

Also compares NNIP predictions against MEAM reference values from
`src/ML/results.json` for all previously simulated alloy compositions.

### ASE Calculator

```python
from calculator import DeepMDCalculator

atoms.calc = DeepMDCalculator()  # uses models/model.pb by default
atoms.calc = DeepMDCalculator(model_path="/path/to/other/model.pb")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

The calculator lazily loads the model on first use, and supports energy, forces, and
stress (Voigt convention). It can be used anywhere ASE expects a calculator:
geometry optimization, NEB, phonon calculations, etc.

---

## Configuration Reference

All configuration generation is driven by `src/NNIP/config.json`. Edit this file
to control which elements, compositions, strains, and perturbations are included
in the training set.

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `elements` | object | Element definitions (structure, lattice param) |
| `pure` | object | Pure element strain configurations |
| `binary` | object | Binary alloy configurations |
| `ternary` | object | Random ternary alloy configurations |
| `al_alloys` | object | Custom Al-rich alloy compositions |
| `vacancy` | object | Vacancy defect configurations |
| `rattled` | object | Thermally displaced configurations |
| `seed` | integer | Global random seed for reproducibility |

### `elements`

Defines the element pool and their ground-state crystal properties. Every element
listed here is available for alloy generation.

```json
"elements": {
    "Al": {"structure": "fcc", "a": 4.05},
    "Cu": {"structure": "fcc", "a": 3.61},
    ...
}
```

| Field | Type | Description |
|-------|------|-------------|
| `structure` | string | Crystal structure: `fcc`, `bcc`, `hcp`, or `diamond` |
| `a` | float | Lattice parameter in Angstroms |

To add a new element, add it here **and** add its pseudopotential to
`download_pseudos.py` and `run_dft.py` `PSEUDOPOTENTIALS` dict.

### `pure`

Controls generation of single-element bulk configurations at various strains.

```json
"pure": {
    "enabled": true,
    "supercell_size": 2,
    "volumetric_strains": [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05],
    "shear_strains": [0.02, 0.05]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Set `false` to skip pure element configs |
| `supercell_size` | int | `2` | N×N×N supercell (2 = 8-16 atoms depending on structure) |
| `volumetric_strains` | float[] | ±5% | Isotropic strain values (0.0 = equilibrium) |
| `shear_strains` | float[] | [0.02, 0.05] | Shear strain magnitudes in xy plane |

Each element gets `len(volumetric_strains) + len(shear_strains)` configurations.
With 9 elements and the default values, this produces 9 × (7 + 2) = 81 configs.

### `binary`

Controls systematic binary alloy generation.

```json
"binary": {
    "enabled": true,
    "fractions": [0.25, 0.50, 0.75],
    "supercell_sizes": [2, 3],
    "all_pairs": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Set `false` to skip binary alloys |
| `fractions` | float[] | [0.25, 0.50, 0.75] | Solute fractions for element B |
| `supercell_sizes` | int[] | [2, 3] | 2×2×2 and/or 3×3×3 supercells |
| `all_pairs` | bool | `true` | Generate all C(N,2) element pairs |

With `all_pairs: true`, 9 elements produce C(9,2) = 36 pairs. Combined with
3 fractions and 2 supercell sizes: 36 × 3 × 2 = 216 configs. Set `all_pairs: false`
to disable (use `al_alloys` for targeted compositions instead).

### `ternary`

Random ternary alloy sampling.

```json
"ternary": {
    "enabled": true,
    "n_random": 30,
    "supercell_sizes": [2, 3],
    "seed": 42
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Set `false` to skip |
| `n_random` | int | `30` | Number of random ternary compositions to sample |
| `supercell_sizes` | int[] | [2, 3] | Supercell sizes to choose from (random per config) |
| `seed` | int | `42` | Seed for ternary sampling (independent of global seed) |

Three elements are randomly chosen from the pool, with random fractions
(Dirichlet-like: 3 uniform random values normalized to sum to 1).

### `al_alloys`

Targeted Al-rich alloy compositions representing real commercial alloys.

```json
"al_alloys": {
    "enabled": true,
    "supercell_sizes": [2, 3],
    "compositions": [
        {"elements": ["Al", "Cu"], "fractions": [0.95, 0.05]},
        {"elements": ["Al", "Zn", "Mg", "Cu"], "fractions": [0.88, 0.06, 0.03, 0.03]},
        ...
    ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Set `false` to skip |
| `supercell_sizes` | int[] | [2, 3] | Each composition at each supercell size |
| `compositions` | object[] | (see default) | List of `{elements, fractions}` pairs |

Each composition entry has:
- `elements`: ordered list of element symbols (first element determines the lattice)
- `fractions`: corresponding atomic fractions (must sum to ~1.0)

The default compositions correspond to:
- **Al-Cu** (2xxx series): 5% and 10% Cu
- **Al-Mg** (5xxx series): 5% and 10% Mg
- **Al-Zn** (7xxx series): 6% Zn
- **Al-Zn-Mg** (7075-like): 90/6/4
- **Al-Cu-Mn** (2024-like): 93/5/2
- **Al-Si-Mg** (6xxx series): 93/4/3
- **Al-Cu-Mg-Mn** (2024-full): 91/4/3/2
- **Al-Zn-Mg-Cu** (7075-full): 88/6/3/3

Add or remove entries freely. Fractions are approximate — actual atom counts are
rounded to the nearest integer for the given supercell size.

### `vacancy`

Point defect configurations.

```json
"vacancy": {
    "enabled": true,
    "max_vacancies": 2,
    "alloy_compositions": [
        {"elements": ["Al", "Cu"], "fractions": [0.90, 0.10]},
        ...
    ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Set `false` to skip |
| `max_vacancies` | int | `2` | Generate 1 to N vacancy configs per structure |
| `alloy_compositions` | object[] | (3 entries) | Alloy compositions to create vacancies in |

Vacancies are created by randomly removing atoms from 2×2×2 supercells. Both pure
element vacancies (all 9 elements × max_vacancies) and alloy vacancies are generated.

### `rattled`

Thermally displaced configurations that simulate finite-temperature snapshots.

```json
"rattled": {
    "enabled": true,
    "temperatures": {"300K": 0.05, "600K": 0.10, "1000K": 0.15},
    "snapshots_per_temp": 3,
    "alloy_compositions": [
        {"elements": ["Al", "Cu"], "fractions": [0.90, 0.10]},
        ...
    ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Set `false` to skip |
| `temperatures` | object | 3 entries | Label → Gaussian displacement stdev (A) |
| `snapshots_per_temp` | int | `3` | Independent random snapshots per temperature |
| `alloy_compositions` | object[] | (2 entries) | Alloy compositions to rattle |

The stdev values are rough approximations of thermal RMS displacement:
- **0.05 A** ≈ 300K for typical metals
- **0.10 A** ≈ 600K
- **0.15 A** ≈ 1000K

Gaussian noise is added to all atomic positions in a 2×2×2 supercell. This teaches
the model about the potential energy surface away from equilibrium, which is critical
for accurate force predictions during MD.

### Example: Minimal Config

To generate a small test set with only Al-Cu binaries:

```json
{
    "elements": {
        "Al": {"structure": "fcc", "a": 4.05},
        "Cu": {"structure": "fcc", "a": 3.61}
    },
    "pure": {
        "enabled": true,
        "supercell_size": 2,
        "volumetric_strains": [-0.03, 0.0, 0.03],
        "shear_strains": [0.03]
    },
    "binary": {
        "enabled": true,
        "fractions": [0.25, 0.50, 0.75],
        "supercell_sizes": [2],
        "all_pairs": true
    },
    "ternary": {"enabled": false},
    "al_alloys": {
        "enabled": true,
        "supercell_sizes": [2],
        "compositions": [
            {"elements": ["Al", "Cu"], "fractions": [0.95, 0.05]}
        ]
    },
    "vacancy": {"enabled": false},
    "rattled": {
        "enabled": true,
        "temperatures": {"300K": 0.05},
        "snapshots_per_temp": 2,
        "alloy_compositions": []
    },
    "seed": 42
}
```

Run with: `python src/NNIP/generate_configs.py --config my_minimal.json`

---

## File Structure

```
src/NNIP/
├── config.json            # Configuration generation settings (edit this)
├── input.json             # DeePMD-kit training hyperparameters
├── download_pseudos.py    # Phase 1: Download & validate pseudopotentials
├── generate_configs.py    # Phase 2a: Generate atomic configurations
├── run_dft.py             # Phase 2b: Run QE SCF calculations
├── convert_to_deepmd.py   # Phase 2c: Convert XYZ → DeePMD raw format
├── install_deepmd.sh      # Phase 3: DeePMD-kit installation
├── train.sh               # Phase 4: Training + freeze wrapper
├── validate.py            # Phase 5: Validation plots & metrics
├── run_md.py              # Phase 6: LAMMPS/ASE MD simulation
├── elastic.py             # Phase 6: Elastic property computation
├── calculator.py          # Phase 6: ASE Calculator wrapper
├── viz.py                 # Phase 6: OVITO trajectory visualization
├── visualization/         # Rendered MP4 videos (auto-created)
└── README.md              # This file

data/
├── training/
│   ├── configs/           # Generated XYZ configurations + manifest.json
│   ├── dft_results/       # QE results (XYZ with energy/forces/stress)
│   ├── deepmd/            # DeePMD raw format (sys_NNN/ directories)
│   └── qe_scratch/        # Temporary QE working directories
└── validation/
    ├── dft_results/       # Held-out validation DFT results
    └── deepmd/            # Validation data in DeePMD format

models/
├── model.pb               # Frozen DeePMD model
├── model.ckpt.*           # Training checkpoints
├── lcurve.out             # Learning curve data
├── train.log              # Training log
└── validation_results/    # Parity plots and metrics.json

pseudopotentials/          # PBE PAW pseudopotential files (.UPF)
```

---

## Troubleshooting

### numpy version conflict

DeePMD-kit 3.1.2 requires numpy < 2.x due to some dependencies compiled against
numpy 1.x. If you see `numpy.dtype size changed` errors:
```bash
pip install 'numpy<2'
```

### `pair_style deepmd` not available in LAMMPS

The system LAMMPS (20240207) was not built with the DeePMD plugin. The scripts
automatically fall back to the ASE calculator. To enable native LAMMPS support,
rebuild LAMMPS with:
```bash
cmake -D PKG_PLUGIN=ON -D PLUGIN_PATH=/path/to/deepmd/lib ..
```

### QE SCF fails to converge

Some strained or rattled configurations may not converge. `run_dft.py` catches
these failures and logs them to `data/training/dft_results/failed.json`. Common
causes:
- **mixing_beta too high**: reduce to 0.1 for problematic configs
- **degauss too small**: increase to 0.04 for metallic systems
- **ecutwfc too low**: try 50 Ry for configurations with short bonds

### DFT run is too slow

Each SCF takes 30s–5min. To speed up:
- Reduce `supercell_sizes` to `[2]` only (skip 3×3×3)
- Reduce `binary.fractions` to fewer values
- Set `ternary.n_random` lower
- Disable categories you don't need (`enabled: false`)
- Run QE with MPI: edit `QE_BIN` in `run_dft.py` to `mpirun -np 4 pw.x`

### Training does not converge

Check `models/lcurve.out` for the loss trajectory. Common issues:
- **Energy loss flat**: not enough diverse training data
- **Force loss oscillating**: reduce `start_lr` or increase `decay_steps`
- **Validation loss diverges**: overfitting — add more data or reduce model size
