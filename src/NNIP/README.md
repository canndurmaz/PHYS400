# NNIP: Neural Network Interatomic Potential Pipeline

Automated generation of multi-element MEAM interatomic potentials. The pipeline follows: **Config Generation** → **MD Simulation (E, ν)** → **DFT Reference** → **NN Optimization**. Elements are auto-discovered from MEAM library files and pseudopotentials are auto-downloaded.

---

## Scientific Background

### The Problem

Molecular dynamics simulations of alloys require interatomic potentials that accurately reproduce mechanical properties (Young's modulus, Poisson's ratio) across compositions. Developing these potentials traditionally requires extensive manual fitting against experimental data -- a process that scales poorly with the number of elements.

### The Approach

This pipeline combines four stages into an automated workflow:

1. **Config Generation** produces random alloy compositions from elements available in the MEAM library files (`EAM/library_*.meam`). Each config specifies element fractions at 1e-4 precision, with Al always dominant when present.

2. **MD Simulation** (LAMMPS) evaluates each composition to compute Young's modulus (E) and Poisson's ratio (ν) via static deformation. Results populate the training dataset (`results.json`).

3. **Density Functional Theory (DFT)** provides quantum-mechanical reference data: equilibrium lattice constants, cohesive energies, bulk moduli, elastic constants, and binary formation energies. Pseudopotentials are auto-downloaded from the QE library if missing. DFT results initialize MEAM parameters via:
   - **Cohesive energy** → `esub` (sublimation energy parameter)
   - **Lattice constant** → `alat` (equilibrium lattice spacing)
   - **Rose equation**: `alpha = sqrt(9 * B * Omega / E_coh)` (universal binding curve shape)
   - **Formation energies** → `Ec(i,j)` cross-terms

4. **Neural Network Surrogate Optimization** bridges the gap between DFT-initialized parameters and MD-computed alloy properties. A neural network learns the mapping `MEAM_params → (C11, C12)` from sampled LAMMPS evaluations under a **Huber loss** — predicting the cubic elastic constants directly is better-conditioned than the ratio-based $(E, \nu)$ target, especially for Mg/Zn-rich compositions whose $(E, \nu)$ regression collapsed in the previous formulation. Gradient descent through the trained network finds parameters that reproduce target $C_{ij}$ across all compositions simultaneously, and the reportable $(E, \nu)$ are recovered analytically from the inverse-optimised predictions.

---

## Workflow

```
  Stage 0: Element Discovery
          |  Auto-detect from EAM/library_*.meam (or --elements CLI)
          v
  Stage 1: DFT Reference (Quantum Espresso)
          |  Auto-download pseudopotentials if missing
          |  EOS fits, elastic constants, formation energies
          |  → dft_results.json
          v
  Stage 2: MEAM Initialization
          |  Map DFT values onto MEAM parameter space
          |  → EAM/dft_initialized/
          v
  Stage 3: NN Surrogate Optimization
          |  k-means medoid select (k=100) → 70/30 train/val split
          |  Sample (checkpointed) → Train NN on train set → Inverse optimize
          |  → EAM/optimized/, nn_checkpoint.json, nn_diagnostics.json
          v
  Stage 4: Verification
          |  Held-out 30% val sweep (always)
          |  Full results.json sweep (opt-in via --full-set-validation)
          v
  Stage 5: Visualization (optional)
```

The upstream stages (config generation + MD simulation) are run separately to build the training dataset before invoking this pipeline. See the [project README](../../README.md) for the full end-to-end workflow.

---

## Usage

Elements are auto-discovered from `EAM/library_*.meam` files when `--elements` is not provided. Pseudopotentials are auto-downloaded from the QE library when missing.

```bash
# From src/NNIP/:
./run_pipeline.sh                                       # Auto-discover elements from EAM/
./run_pipeline.sh Al Cu Zn Mg                           # Specify elements explicitly
./run_pipeline.sh --skip-dft Al Cu Zn Mg                # Reuse existing DFT results
./run_pipeline.sh --skip-dft --parallel 6 \
    --perturbations 200 --k-representatives 50          # Tune sampling + representative count
./run_pipeline.sh --skip-dft --full-set-validation \    # Add post-optimization sweep over
    Al Cu Zn Mg                                         #   every entry in results.json
./run_pipeline.sh --clean                               # Remove pipeline outputs (preserves DFT)
```

### CLI Options

| Flag                       | Effect                                                                       |
|----------------------------|------------------------------------------------------------------------------|
| `--elements X Y Z`         | Specify elements (auto-discovered from EAM/ if omitted)                      |
| `--skip-dft`               | Use existing `dft_results.json`                                              |
| `--skip-optimize`          | Stop after MEAM initialization                                               |
| `--skip-verify`            | Skip verification stage                                                      |
| `--resume`                 | Auto-detect completed stages and skip them                                   |
| `--no-plots`               | Skip visualization generation                                                |
| `--perturbations N`        | MEAM parameter perturbations sampled in Phase 1 (default: 150)               |
| `--parallel N`             | Max parallel workers for DFT / NN sampling / full-set validation (default: 4)|
| `--k-representatives N`    | k-means medoids picked from `results.json` before train/val split (default: 100) |
| `--val-frac F`             | Fraction of representatives held out for validation (default: 0.3)           |
| `--split-seed N`           | Seed for k-means + train/val split (default: 0)                              |
| `--full-set-validation`    | After the val-set check, sweep every entry in `results.json` (slow)          |
| `--clean`                  | Remove NNIP-generated content and exit. **DFT artifacts always preserved.**  |

---

## Pipeline Stages

### Stage 0: Element Discovery

Dynamically scans all `EAM/library_*.meam` files to discover available elements. No GUI or manual selection required. Pass `--elements` to override with a specific subset.

### Stage 1: DFT Reference

For each element, Quantum Espresso computes:
- **Equation of state** (7 strain points, Birch-Murnaghan fit) → `a0`, `E_coh`, `B`
- **Elastic constants** (stress-strain with `cubic=True`) → `C11`, `C12`
- **Isolated atom energy** for true cohesive energy: `E_coh = E_atom - E_bulk/N`

Missing pseudopotentials are automatically downloaded from `pseudopotentials.quantum-espresso.org` before calculations begin.

For all N(N-1)/2 binary pairs:
- **Formation energy** in an appropriate reference structure (L1₂, B2, etc. chosen by lattice type)

Results are saved incrementally -- partial progress survives failures.

### Stage 2: MEAM Initialization

DFT values are mapped onto the MEAM parameter space. Base MEAM files are auto-discovered from `EAM/` (any `library_*.meam` covering the selected elements). Cross-terms use mixing rules where DFT data is unavailable.

### Stage 3: NN Optimization

The pipeline first calls `select_representatives.py` to reduce the full
~7800-entry `results.json` to `k` (default 100) **k-means medoid alloys** in
element-fraction composition space. Each medoid is a *real* entry from
`results.json` nearest to its k-means cluster centre, so the original
DFT-computed `(C11, C12, E, ν)` targets carry over verbatim. The medoids are
then split 70/30 into `results_train.json` and `results_val.json` (configurable
via `--val-frac` / `--split-seed`). At full scale this cuts Phase 1 LAMMPS cost
by ~100× while keeping composition-space coverage broad.

`optimize_nn` then runs four phases:

1. **Sample**: Perturb initial MEAM parameters by $\pm 10\%$, evaluate each with LAMMPS across the **training-set alloys**, flatten the per-alloy `(E, ν)` to $(C_{11}, C_{12})$ via the cubic-isotropic algebra. Samples whose $C_{11}$ falls below `C_REJECT_MIN = 30 GPa` for too many compositions are rejected. **Each accepted sample is atomically appended to `nn_checkpoint.json`** — killing the process and re-running picks up exactly where it left off, gated by a hash over `(lib, params, opt_spec, train_entries)`.
2. **Train**: Feed-forward NN (`20 → 20 → 10 → 2·N_train` architecture) learns the params-to-$C_{ij}$ mapping using **Huber loss** (delta=1.0 in normalised units). Targets prefer the stored `C11_GPa`/`C12_GPa`, falling back to algebraic conversion only for legacy entries that pre-date the C_ij storage change.
3. **Optimize**: Gradient descent through the trained NN toward the train $C_{ij}$ target vector, with the parameter vector clipped to $\pm 2.5\sigma$ around the training mean.
4. **Held-out validation**: Final LAMMPS evaluation against the **30% validation alloys the surrogate never saw**. Reports per-entry $(C_{11}, C_{12}, E, \nu)$ — target vs optimized — plus aggregate mean and RMSE on $E$ and $\nu$. This replaces the prior train-set self-evaluation, which understated generalisation error.

The diagnostics file `nn_diagnostics.json` carries `train_target_vec_cij`,
`train_target_vec_enu`, the `train_path` / `val_path`, the full
`val_predictions[*]` table, and `val_metrics` (rmse_E_pct / rmse_nu_pct /
mean_E_pct / mean_nu_pct / n_valid / n_val).

### Stage 4: Verification

Reuses the optimized potential against the held-out 30% validation set, then
optionally — under `--full-set-validation` — sweeps every entry in
`results.json` (parallelised across `--parallel` workers, progress reported
every 100 alloys). Per-entry errors and aggregate mean/RMSE for $E$ and $\nu$
are written to `pipeline_summary.json` under `verification` (val-set) and
`full_set_verification` (full sweep, when requested). Pass criterion: average
error < 10%.

### Stage 5: Visualization

Generates diagnostic plots from NN training and verification results.

---

## File Structure

```
src/NNIP/
  pipeline.py                  Main entry point (orchestrator)
  run_pipeline.sh              Shell wrapper (generic --* flag passthrough)
  download_pseudopotentials.py Auto-download missing QE pseudopotentials
  dft_reference.py             DFT calculations via QE + ASE
  fill_elastic_constants.py    One-shot fill of missing/unphysical C11,C12
                               in dft_results.json (uses B + assumed ν per
                               element); idempotent
  dft_to_meam.py               DFT → MEAM parameter mapping
  select_representatives.py    k-means medoid selection + 70/30 train/val
                               split from results.json (numpy-only k-means++)
  nn_optimizer.py              NN surrogate training + inverse optimization
                               (C_ij targets, Huber loss). Phase-1 sample
                               checkpointing via nn_checkpoint.json; held-out
                               val-set validation in Phase 4.
  meam_io.py                   MEAM library/params file I/O
  merge_potentials.py          Multi-source MEAM potential merger
  visualize.py                 Diagnostic plotting
  logging_config.py            Centralized logging setup
  verify_7075.py               Standalone verification script
```

#### `fill_elastic_constants.py`

`dft_reference.py:_elastic_constants` only computes $C_{11}, C_{12}$ for cubic (FCC/BCC) elements; HCP and diamond elements are skipped. A handful of cubic entries also occasionally come out clearly broken (e.g.\ Fe with $C_{11} < C_{12}$ — Cauchy / mechanical-stability violation indicating a miss-converged DFT run). For Stage 4's MEAM seeding, those entries still need *some* value, so this script reconstructs them from the bulk modulus $B$ and a per-element Poisson-ratio default under the cubic-isotropic relation
$C_{11} = 3B(1-\nu)/(1+\nu)$, $C_{12} = 3B\nu/(1+\nu)$ (so $C_{11}+2C_{12} \equiv 3B$).

```bash
python3 fill_elastic_constants.py --dry-run     # preview
python3 fill_elastic_constants.py               # only fill missing entries
python3 fill_elastic_constants.py --force       # also fix unphysical entries
```

**Auto-discovered inputs** (no manual configuration needed):
- `EAM/library_*.meam` + matching `*.meam` -- base MEAM potentials (determines available elements)
- `pseudopotentials/*.UPF` -- QE pseudopotential files (auto-downloaded if missing)
- `src/ML/results.json` -- MD-computed training targets (composition, E, ν)

**Generated outputs:**
- `src/NNIP/dft_results.json` -- DFT reference data (**preserved by `--clean`**)
- `src/NNIP/dft_scratch/` -- QE working files (**preserved by `--clean`**)
- `src/NNIP/nn_checkpoint.json` -- Phase-1 sample checkpoint for resume after kill
- `src/NNIP/nn_diagnostics.json` -- Loss history + val predictions + val metrics
- `src/NNIP/pipeline_summary.json` -- Timing, verification, and (opt.) full-set verification
- `src/NNIP/logs/` -- Timestamped pipeline logs
- `src/ML/results_train.json` -- k-means medoid training subset (regenerated each run)
- `src/ML/results_val.json` -- 30% held-out validation subset
- `EAM/dft_initialized/` -- DFT-initialized MEAM potentials
- `EAM/optimized/` -- Final optimized MEAM potentials

---

## Prerequisites

- **Python 3.12** with venv (`phys/`) created using `--system-site-packages`
- **Quantum Espresso 7.5** (`pw.x` accessible via venv PATH)
- **LAMMPS** with MEAM support (system install, `maxelt >= 20`)
- **ASE** for QE calculator interface
- **TensorFlow** for NN surrogate
- **Pseudopotentials**: auto-downloaded to `pseudopotentials/` (PAW-PBE UPF format)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Too many elements" from LAMMPS | Rebuild LAMMPS with `maxelt=20`; set `LD_LIBRARY_PATH=$HOME/.local/lib` |
| DFT hangs or fails | Check `which pw.x`; pseudopotentials are auto-downloaded; partial results are saved |
| NN won't converge | Increase `--perturbations`; bump `--k-representatives` for broader composition coverage; check that `results.json` targets are physically reasonable |
| Phase 1 killed midway | Just re-run — `nn_checkpoint.json` resumes from the last accepted sample. Pass `--no-resume` (via `nn_optimizer.py` directly) to ignore the checkpoint and start fresh |
| Missing pseudopotential | Run `python download_pseudopotentials.py Al Cu Fe` to fetch manually |
