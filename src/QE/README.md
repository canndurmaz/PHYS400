# Quantum Espresso

## Version
Quantum ESPRESSO v7.5 (PWSCF)

## Build Details
- **Source:** GitHub release `qe-7.5` from [QEF/q-e](https://github.com/QEF/q-e)
- **Build location:** `/home/kenobi/Workspaces/qe/`
- **Build date:** 2026-03-15
- **Compiler:** gfortran 13.3.0
- **MPI:** OpenMPI 4.1.6 (parallel build)
- **FFT:** FFTW3 3.3.10
- **BLAS/LAPACK:** QE internal (bundled)

## Build Steps
```bash
# 1. Download source
mkdir -p /home/kenobi/Workspaces/qe
cd /home/kenobi/Workspaces/qe
curl -L -o qe-7.5.tar.gz "https://api.github.com/repos/QEF/q-e/tarball/refs/tags/qe-7.5"
tar xzf qe-7.5.tar.gz --strip-components=1

# 2. Configure (auto-detects MPI, FFTW3, BLAS/LAPACK)
./configure

# 3. Build pw.x only
make -j8 pw

# 4. PATH setup — added to phys venv activation script
# In phys/bin/activate, QE bin dir is prepended to PATH
```

## System Dependencies
All were pre-installed on this system:
- `gfortran` 13.3.0
- `libopenmpi-dev` (OpenMPI 4.1.6)
- `libfftw3-dev` (FFTW3 3.3.10)
- `libblas-dev` / `liblapack-dev` (3.12.0)
- `cmake` 3.28.3

## Paths

| Resource | Path |
|----------|------|
| QE binary (`pw.x`) | `/home/kenobi/Workspaces/qe/bin/pw.x` |
| Pseudopotentials | `/home/kenobi/Workspaces/PHYS400/pseudopotentials/` |
| Test script | `/home/kenobi/Workspaces/PHYS400/src/QE/test_qe.py` |
| Run script | `/home/kenobi/Workspaces/PHYS400/src/QE/run.sh` |

## Pseudopotentials

Stored in the project root at `pseudopotentials/`:

| Element | File | Type | XC Functional |
|---------|------|------|---------------|
| Al | `Al.pbe-n-kjpaw_psl.1.0.0.UPF` | PAW (Projector Augmented Wave) | PBE |

Additional pseudopotentials can be downloaded from the [QE pseudopotential library](https://pseudopotentials.quantum-espresso.org/) and placed in the `pseudopotentials/` directory.

## Python Interface

QE is driven from Python using **ASE** (Atomic Simulation Environment, v3.27.0).

### Setup

```python
from ase.calculators.espresso import Espresso, EspressoProfile

QE_BIN = "/home/kenobi/Workspaces/qe/bin/pw.x"
PSEUDO_DIR = "/home/kenobi/Workspaces/PHYS400/pseudopotentials"

profile = EspressoProfile(command=QE_BIN, pseudo_dir=PSEUDO_DIR)
```

### Calculator Configuration

The test script (`test_qe.py`) uses the following default settings:

```python
calc = Espresso(
    profile=profile,
    pseudopotentials={"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
    input_data={
        "control": {
            "tprnfor": True,       # compute forces
            "tstress": True,       # compute stress tensor
        },
        "system": {
            "ecutwfc": 30,         # wavefunction cutoff (Ry)
            "ecutrho": 240,        # charge density cutoff (Ry)
            "occupations": "smearing",
            "smearing": "mv",      # Marzari-Vanderbilt cold smearing
            "degauss": 0.02,       # smearing width (Ry)
        },
        "electrons": {
            "conv_thr": 1.0e-6,    # SCF convergence threshold (Ry)
        },
    },
    kpts=(4, 4, 4),                # Monkhorst-Pack k-point grid
)
```

### Parameter Notes
- **`tprnfor` / `tstress`**: Required for force and stress calculations; both enabled by default in the test
- **`occupations = "smearing"`**: Mandatory for metals like Al (Fermi surface needs broadening)
- **`smearing = "mv"`**: Marzari-Vanderbilt cold smearing — recommended for metals
- **`degauss = 0.02`**: Smearing width in Ry; small enough for accuracy, large enough for convergence
- **`ecutwfc = 30`**: Sufficient for Al with the PAW pseudopotential; increase for production
- **`kpts=(4,4,4)`**: Adequate for quick tests; use `(6,6,6)` or higher for converged results

## Test Script (`test_qe.py`)

Runs 3 validation tests on bulk FCC Aluminum (a = 4.05 A):

| Test | What it checks | Pass criteria |
|------|----------------|---------------|
| **SCF Energy** | Single-point DFT energy | Energy in range (-540, -530) eV |
| **Forces** | Atomic forces on perfect crystal | max \|F\| < 0.01 eV/A |
| **Stress** | Stress tensor symmetry (cubic) | Off-diagonal < 0.001, diagonal spread < 0.001 |

### Running the test
```bash
# Via the shell script
./src/QE/run.sh

# Or directly
source phys/bin/activate
python3 src/QE/test_qe.py
```

### Expected output
```
==================================================
Quantum Espresso + ASE Integration Test
==================================================

[SCF Energy]
  Energy:  -537.1662 eV  [PASS]

[Forces]
  Forces:  max |F| = 0.000000 eV/A  [PASS]

[Stress]
  Stress:  P = 0.19 GPa, spread = 0.00e+00  [PASS]

==================================================
Results: 3 passed, 0 failed
==================================================
Test output cleaned up.
```

## CLI Usage

```bash
# Activate the phys venv (adds QE to PATH automatically)
source /home/kenobi/Workspaces/PHYS400/phys/bin/activate

# Run a calculation
pw.x < input.in > output.out

# Run in parallel with MPI
mpirun -np 4 pw.x < input.in > output.out
```
