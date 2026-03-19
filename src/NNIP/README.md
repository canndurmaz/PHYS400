# NNIP: Neural Network Interatomic Potential Pipeline

Automated generation of multi-element MEAM interatomic potentials from first principles. The only required input is **element selection** -- everything else is derived automatically.

---

## Scientific Background

### The Problem

Molecular dynamics simulations of alloys require interatomic potentials that accurately reproduce mechanical properties (Young's modulus, Poisson's ratio) across compositions. Developing these potentials traditionally requires extensive manual fitting against experimental data -- a process that scales poorly with the number of elements.

### The Approach

This pipeline combines three computational methods into an automated workflow:

1. **Density Functional Theory (DFT)** provides quantum-mechanical reference data: equilibrium lattice constants, cohesive energies, bulk moduli, elastic constants, and binary formation energies. These are computed from the electronic structure with no empirical parameters.

2. **Modified Embedded Atom Method (MEAM)** is a semi-empirical potential form used by LAMMPS for fast MD simulations. Each element requires ~14 parameters (embedding function coefficients, electron density weights, lattice geometry), plus cross-terms for each binary pair. DFT results initialize these parameters via:
   - **Cohesive energy** -> `esub` (sublimation energy parameter)
   - **Lattice constant** -> `alat` (equilibrium lattice spacing)
   - **Rose equation**: `alpha = sqrt(9 * B * Omega / E_coh)` (universal binding curve shape)
   - **Formation energies** -> `Ec(i,j)` cross-terms

3. **Neural Network Surrogate Optimization** bridges the gap between DFT-initialized parameters and experimental alloy properties. A neural network learns the mapping `MEAM_params -> (E, nu)` from sampled LAMMPS evaluations, then gradient descent through the trained network finds parameters that reproduce target mechanical properties across all compositions simultaneously.

---

## Workflow

```
  Element Selection (GUI or CLI)
          |
          v
  Stage 1: DFT Reference (Quantum Espresso)
          |  EOS fits, elastic constants, formation energies
          |  -> dft_results.json
          v
  Stage 2: MEAM Initialization
          |  Map DFT values onto MEAM parameter space
          |  -> EAM/dft_initialized/
          v
  Stage 3: NN Surrogate Optimization
          |  Sample -> Train NN -> Inverse optimize
          |  -> EAM/optimized/
          v
  Stage 4: Verification
          |  Compare against experimental alloy data
          v
  Stage 5: Visualization (optional)
```

---

## Usage

**The only configuration required is element selection.** The pipeline auto-discovers pseudopotentials, base MEAM files, and training targets.

```bash
# From src/NNIP/:
./run_pipeline.sh                              # Full pipeline with GUI
./run_pipeline.sh Al Cu Zn Mg                  # Skip GUI, specify elements
./run_pipeline.sh --skip-dft Al Cu Zn Mg       # Reuse existing DFT results
./run_pipeline.sh --samples 50 Al Cu Zn Mg     # More NN training samples
```

### CLI Options

| Flag               | Effect                                          |
|--------------------|--------------------------------------------------|
| `--elements X Y Z` | Bypass GUI element selection                     |
| `--skip-dft`       | Use existing `dft_results.json`                  |
| `--skip-optimize`  | Stop after MEAM initialization                   |
| `--skip-verify`    | Skip verification stage                          |
| `--samples N`      | NN parameter samples (default: 30)               |
| `--parallel N`     | Max parallel DFT workers (default: 4)            |
| `--no-plots`       | Skip visualization generation                    |

---

## Pipeline Stages

### Stage 0: Element Selection

A web GUI (`http://127.0.0.1:8472`) scans `pseudopotentials/` for available UPF files and displays clickable element cards. Alternatively, pass `--elements` on the CLI.

### Stage 1: DFT Reference

For each selected element, Quantum Espresso computes:
- **Equation of state** (7 strain points, Birch-Murnaghan fit) -> `a0`, `E_coh`, `B`
- **Elastic constants** (stress-strain with `cubic=True`) -> `C11`, `C12`
- **Isolated atom energy** for true cohesive energy: `E_coh = E_atom - E_bulk/N`

For all N(N-1)/2 binary pairs:
- **Formation energy** in an appropriate reference structure (L1_2, B2, etc. chosen by lattice type)

Results are saved incrementally -- partial progress survives failures.

### Stage 2: MEAM Initialization

DFT values are mapped onto the MEAM parameter space. Base MEAM files are auto-discovered from `EAM/` (any `library_*.meam` covering the selected elements). Cross-terms use mixing rules where DFT data is unavailable.

### Stage 3: NN Optimization

1. **Sample**: Perturb initial MEAM parameters, evaluate each with LAMMPS across all target compositions
2. **Train**: Feed-forward NN (64-64-32 architecture) learns params-to-properties mapping
3. **Optimize**: Gradient descent through the trained NN toward experimental targets from `src/ML/results.json`
4. **Validate**: Final LAMMPS evaluation with optimized parameters

### Stage 4: Verification

Compares optimized potential against all target compositions. Reports per-entry and average percentage errors for Young's modulus and Poisson's ratio. Pass criterion: average error < 10%.

### Stage 5: Visualization

Generates diagnostic plots from NN training and verification results.

---

## File Structure

```
src/NNIP/
  pipeline.py            Main entry point (orchestrator)
  run_pipeline.sh        Shell wrapper
  element_selector.py    Web GUI for element selection
  dft_reference.py       DFT calculations via QE + ASE
  dft_to_meam.py         DFT -> MEAM parameter mapping
  nn_optimizer.py        NN surrogate training + inverse optimization
  meam_io.py             MEAM library/params file I/O
  merge_potentials.py    Multi-source MEAM potential merger
  visualize.py           Diagnostic plotting
  logging_config.py      Centralized logging setup
  verify_7075.py         Standalone verification script
```

**Auto-discovered inputs** (no manual configuration needed):
- `pseudopotentials/*.UPF` -- QE pseudopotential files (determines available elements)
- `EAM/library_*.meam` + matching `*.meam` -- base MEAM potentials
- `src/ML/results.json` -- experimental training targets (composition, E, nu)

**Generated outputs:**
- `src/NNIP/dft_results.json` -- DFT reference data
- `src/NNIP/dft_scratch/` -- QE working files
- `EAM/dft_initialized/` -- DFT-initialized MEAM potentials
- `EAM/optimized/` -- Final optimized MEAM potentials
- `src/NNIP/pipeline_summary.json` -- Timing and results summary
- `src/NNIP/logs/` -- Timestamped pipeline logs

---

## Prerequisites

- **Python 3.12** with venv (`phys/`) created using `--system-site-packages`
- **Quantum Espresso 7.5** (`pw.x` accessible via venv PATH)
- **LAMMPS** with MEAM support (system install, `maxelt >= 20`)
- **ASE** for QE calculator interface
- **TensorFlow** for NN surrogate
- **Pseudopotentials** in `pseudopotentials/` (PAW-PBE UPF format)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Too many elements" from LAMMPS | Rebuild LAMMPS with `maxelt=20`; set `LD_LIBRARY_PATH=$HOME/.local/lib` |
| DFT hangs or fails | Check `which pw.x`; verify pseudopotentials exist; partial results are saved |
| NN won't converge | Increase `--samples`; check that `results.json` targets are physically reasonable |
| GUI unresponsive | Use `--elements` CLI flag instead |
