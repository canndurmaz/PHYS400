# PHYS400 — Computational Materials Science

Molecular dynamics (MD) simulations of metallic alloys using **LAMMPS** and **Python**.
The project spans an initial Cu prototype through a full elastic tensor calculation for Fe-V alloys.

## Background

**Molecular dynamics** evolves a system of atoms in time by solving Newton's equations of motion
under an interatomic potential. Here we use **EAM/FS** (Embedded Atom Method / Finnis-Sinclair)
potentials — a standard choice for transition metals that captures metallic bonding through an
electron-density embedding function.

All simulations run in **LAMMPS metal units**: lengths in Å, energies in eV, pressures in bar.

---

## Project Structure

```
PHYS400/
├── src/          # Prototype: Cu FCC MD at 300 K
├── EAM/          # Interatomic potential library (EAM/FS files)
└── Elastic/      # Main module: Fe-V elastic tensor calculator
```

---

## Modules

### `src/` — Copper MD Prototype

A minimal end-to-end MD pipeline used to learn the LAMMPS Python API.

| File | Role |
|---|---|
| `lmp.py` | Build a 5×5×5 Cu FCC supercell, minimize, run NVT at 300 K for 1000 steps, collect temperature |
| `viz.py` | Render the trajectory to `copper_vibration.mp4` using OVITO |
| `run.sh` | Run `lmp.py` then `viz.py` |
| `test.py` | MPI sanity check (`mpirun -np 4 python test.py`) |

Run:
```bash
cd src
bash run.sh
```

---

### `EAM/` — Potential Library

A collection of EAM and EAM/FS potential files for common metallic systems including Fe, V, Cu, Al, Ni, Zr, W, and their alloys. The Fe-V elastic calculation uses `VFe_mm.eam.fs`.

---

### `Elastic/` — Fe-V Elastic Tensor

The main research module. Computes the full **6×6 elastic tensor C_ij** of an Fe-V alloy
supercell via central finite differences on stress, then derives polycrystalline moduli via
Voigt-Reuss-Hill averaging.

See [`Elastic/README.md`](Elastic/README.md) for full documentation.

**Quick summary of results** (Fe + 10 at.% V, 5×5×5 supercell):

| Modulus | Value |
|---|---|
| Young's modulus E | 257.0 GPa |
| Bulk modulus K | 212.0 GPa |
| Shear modulus G | 99.0 GPa |
| Poisson's ratio ν | 0.298 |

Run:
```bash
cd Elastic
bash run.sh
```

---

## Environment

| Dependency | Version / Notes |
|---|---|
| Python | 3.12 (system) |
| LAMMPS | 20240207 — system install, Python bindings via `lammps` module |
| OVITO | 3.14.1 — installed in venv |
| NumPy / Matplotlib | standard scientific Python stack |

Set up the virtual environment (inherits system LAMMPS):
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install ovito matplotlib numpy pytest
```
