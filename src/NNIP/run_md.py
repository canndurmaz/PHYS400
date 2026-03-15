#!/usr/bin/env python3
"""Run LAMMPS MD simulation using the trained DeePMD potential.

Tests the frozen model.pb on an Al alloy supercell with NVT dynamics.
Falls back to ASE MD if pair_style deepmd is not available.
"""

import os
import sys

import numpy as np

PROJECT_DIR = "/home/kenobi/Workspaces/PHYS400"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model.pb")


def run_lammps_deepmd():
    """Run MD via LAMMPS with pair_style deepmd."""
    from lammps import lammps

    L = lammps(cmdargs=["-log", "none", "-screen", "none"])

    L.command("units metal")
    L.command("atom_style atomic")
    L.command("boundary p p p")

    # Create Al-Cu alloy supercell (Al-2024 like: 95% Al, 5% Cu)
    L.command("lattice fcc 4.05")
    L.command("region box block 0 4 0 4 0 4")
    L.command("create_box 9 box")  # 9 element types
    L.command("create_atoms 1 box")

    # Set 5% atoms to Cu (type 5 in type_map: Al=1, Zn=2, Mg=3, Mn=4, Cu=5, ...)
    L.command("set type 1 type/fraction 5 0.05 12345")

    # Masses (must match type_map order)
    masses = {
        1: 26.982,   # Al
        2: 65.38,    # Zn
        3: 24.305,   # Mg
        4: 54.938,   # Mn
        5: 63.546,   # Cu
        6: 28.086,   # Si
        7: 51.996,   # Cr
        8: 55.845,   # Fe
        9: 47.867,   # Ti
    }
    for t, m in masses.items():
        L.command(f"mass {t} {m}")

    # DeePMD potential
    L.command(f"pair_style deepmd {MODEL_PATH}")
    L.command("pair_coeff * *")

    # Minimize
    L.command("minimize 1.0e-4 1.0e-6 100 1000")

    # NVT MD at 300K
    L.command("velocity all create 300 12345 dist gaussian")
    L.command("fix 1 all nvt temp 300 300 0.1")

    # Thermo output
    L.command("thermo 100")
    L.command("thermo_style custom step temp pe ke etotal press")

    # Dump trajectory
    traj_path = os.path.join(PROJECT_DIR, "src", "NNIP", "md_traj.lammpstrj")
    L.command(f"dump 1 all custom 100 {traj_path} id type x y z fx fy fz")

    print("Running NVT MD (10000 steps, 300K)...")
    L.command("run 10000")

    # Final properties
    L.command("run 0")
    temp = L.get_thermo("temp")
    pe = L.get_thermo("pe")
    ke = L.get_thermo("ke")
    press = L.get_thermo("press")

    print(f"\nFinal state:")
    print(f"  Temperature: {temp:.1f} K")
    print(f"  PE: {pe:.4f} eV")
    print(f"  KE: {ke:.4f} eV")
    print(f"  Pressure: {press:.2f} bar")
    print(f"  Trajectory: {traj_path}")

    L.close()

    composition = {"Al": 0.95, "Cu": 0.05}
    return traj_path, composition, "lammps"


def run_ase_md():
    """Fallback: run MD via ASE with DeePMD calculator."""
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.langevin import Langevin
    from ase import units
    from ase.io.trajectory import Trajectory

    from calculator import DeepMDCalculator

    print("Using ASE MD fallback (pair_style deepmd not available in LAMMPS)")

    # Create Al-Cu alloy
    al = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    n_atoms = len(al)
    symbols = list(al.get_chemical_symbols())

    # Substitute 5% with Cu
    n_cu = max(1, int(0.05 * n_atoms))
    np.random.seed(42)
    cu_indices = np.random.choice(n_atoms, n_cu, replace=False)
    for idx in cu_indices:
        symbols[idx] = "Cu"
    al.set_chemical_symbols(symbols)

    al.calc = DeepMDCalculator()

    # Initialize velocities at 300K
    MaxwellBoltzmannDistribution(al, temperature_K=300)

    # Langevin dynamics
    traj_path = os.path.join(PROJECT_DIR, "src", "NNIP", "md_traj.traj")
    dyn = Langevin(al, 1.0 * units.fs, temperature_K=300, friction=0.01)

    traj = Trajectory(traj_path, "w", al)
    dyn.attach(traj.write, interval=10)

    def print_status():
        e = al.get_potential_energy()
        t = al.get_kinetic_energy() / (1.5 * units.kB * n_atoms)
        print(f"  Step {dyn.nsteps}: T={t:.1f} K, E={e:.4f} eV "
              f"({e/n_atoms:.4f} eV/atom)")

    dyn.attach(print_status, interval=100)

    print(f"Running Langevin MD (1000 steps, 300K, {n_atoms} atoms)...")
    dyn.run(1000)

    traj.close()
    print(f"Trajectory: {traj_path}")

    # Compute actual composition from final symbols
    from collections import Counter
    counts = Counter(symbols)
    composition = {e: c / n_atoms for e, c in counts.items()}
    return traj_path, composition, "ase"


def main():
    print("=" * 60)
    print("NNIP MD Simulation")
    print("=" * 60)

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Train and freeze model first (train.sh)")
        sys.exit(1)

    # Try LAMMPS first, fall back to ASE
    try:
        from lammps import lammps
        L = lammps(cmdargs=["-log", "none", "-screen", "none"])
        L.command(f"pair_style deepmd {MODEL_PATH}")
        L.close()
        print("LAMMPS DeePMD available, using native LAMMPS MD\n")
        traj_path, composition, backend = run_lammps_deepmd()
    except Exception:
        print("LAMMPS DeePMD not available, falling back to ASE\n")
        traj_path, composition, backend = run_ase_md()

    # Render visualization
    print("\n" + "=" * 60)
    print("Rendering visualization...")
    print("=" * 60)
    try:
        from viz import render_lammps, render_ase
        if backend == "lammps":
            render_lammps(traj_path, composition)
        else:
            render_ase(traj_path, composition)
    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    main()
