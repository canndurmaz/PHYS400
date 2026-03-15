#!/usr/bin/env python3
"""Run QE SCF calculations on generated atomic configurations.

Reads configurations from data/training/configs/, runs QE DFT on each,
and stores results (energy, forces, stress) in extended XYZ format.

Supports resumption: skips configs that already have results.
"""

import json
import os
import shutil
import sys
import traceback

import numpy as np
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile

QE_BIN = "/home/kenobi/Workspaces/qe/bin/pw.x"
PSEUDO_DIR = "/home/kenobi/Workspaces/PHYS400/pseudopotentials"
PROJECT_DIR = "/home/kenobi/Workspaces/PHYS400"
CONFIG_DIR = os.path.join(PROJECT_DIR, "data", "training", "configs")
RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "training", "dft_results")
WORK_DIR = os.path.join(PROJECT_DIR, "data", "training", "qe_scratch")

# Pseudopotential mapping (must match downloaded files exactly)
PSEUDOPOTENTIALS = {
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF",
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "Mn": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Cr": "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
}

# Elements that need spin polarization
MAGNETIC_ELEMENTS = {"Fe", "Cr", "Mn"}


def make_calculator(atoms, calc_dir):
    """Create a QE calculator appropriate for the given configuration."""
    profile = EspressoProfile(command=QE_BIN, pseudo_dir=PSEUDO_DIR)

    elements = set(atoms.get_chemical_symbols())
    pseudos = {e: PSEUDOPOTENTIALS[e] for e in elements}

    # Use higher cutoffs for transition metals
    has_tm = bool(elements & {"Mn", "Fe", "Cr", "Ti", "Cu", "Zn"})
    ecutwfc = 40 if has_tm else 30
    ecutrho = ecutwfc * 8

    # Determine k-point grid based on cell size
    cell_lengths = np.linalg.norm(atoms.get_cell(), axis=1)
    # Target ~0.05 A^-1 k-spacing
    kpts = tuple(max(1, int(np.ceil(2 * np.pi / (L * 0.05)))) for L in cell_lengths)
    # Cap at 6x6x6 for efficiency
    kpts = tuple(min(k, 6) for k in kpts)

    input_data = {
        "control": {
            "tprnfor": True,
            "tstress": True,
        },
        "system": {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": 0.02,
        },
        "electrons": {
            "conv_thr": 1.0e-6,
            "mixing_beta": 0.3,
            "electron_maxstep": 200,
        },
    }

    # Add spin polarization if any magnetic element is present
    if elements & MAGNETIC_ELEMENTS:
        input_data["system"]["nspin"] = 2
        # Set starting magnetization for each species
        elem_list = sorted(elements)
        for i, e in enumerate(elem_list, 1):
            if e in MAGNETIC_ELEMENTS:
                input_data["system"][f"starting_magnetization({i})"] = 0.5

    return Espresso(
        profile=profile,
        pseudopotentials=pseudos,
        input_data=input_data,
        kpts=kpts,
        directory=calc_dir,
    )


def run_single(config_path, result_path, calc_dir):
    """Run DFT on a single configuration and save results."""
    atoms = read(config_path, format="extxyz")

    atoms.calc = make_calculator(atoms, calc_dir)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress_3x3 = atoms.get_stress(voigt=False)  # 3x3 tensor

    # Detach calculator to avoid conflict with extxyz writer
    atoms.calc = None

    # Store results with dft_ prefix to avoid ASE extxyz conflicts
    atoms.info["dft_energy"] = energy
    atoms.info["dft_energy_per_atom"] = energy / len(atoms)
    atoms.info["dft_stress"] = stress_3x3.tolist()
    atoms.arrays["dft_forces"] = forces

    write(result_path, atoms, format="extxyz")
    return energy, forces, stress_3x3


def main():
    print("=" * 60)
    print("DFT Training Data Generation")
    print("=" * 60)

    if not os.path.isfile(QE_BIN):
        print(f"ERROR: pw.x not found at {QE_BIN}")
        sys.exit(1)

    # Load manifest
    manifest_path = os.path.join(CONFIG_DIR, "manifest.json")
    if not os.path.isfile(manifest_path):
        print("ERROR: manifest.json not found. Run generate_configs.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    # Check which configs already have results
    completed = set()
    failed_log = os.path.join(RESULTS_DIR, "failed.json")
    failed_configs = {}
    if os.path.isfile(failed_log):
        with open(failed_log) as f:
            failed_configs = json.load(f)

    for entry in manifest:
        result_file = os.path.join(RESULTS_DIR, entry["file"])
        if os.path.isfile(result_file):
            completed.add(entry["index"])

    todo = [e for e in manifest if e["index"] not in completed]
    print(f"\nTotal configs: {len(manifest)}")
    print(f"Already completed: {len(completed)}")
    print(f"Previously failed: {len(failed_configs)}")
    print(f"To process: {len(todo)}")

    if not todo:
        print("\nAll configurations already processed!")
        sys.exit(0)

    # Process
    n_success = 0
    n_fail = 0

    for i, entry in enumerate(todo):
        idx = entry["index"]
        config_file = entry["file"]
        config_type = entry["config_type"]
        config_path = os.path.join(CONFIG_DIR, config_file)
        result_path = os.path.join(RESULTS_DIR, config_file)
        calc_dir = os.path.join(WORK_DIR, f"calc_{idx:04d}")

        print(f"\n[{i+1}/{len(todo)}] {config_type} ({entry['n_atoms']} atoms, "
              f"elements: {entry['elements']})")

        try:
            energy, forces, stress = run_single(config_path, result_path, calc_dir)
            e_per_atom = energy / entry["n_atoms"]
            max_f = np.abs(forces).max()
            print(f"  E = {energy:.4f} eV ({e_per_atom:.4f} eV/atom), "
                  f"max|F| = {max_f:.4f} eV/A")
            n_success += 1

            # Clean scratch for this calc to save disk
            if os.path.exists(calc_dir):
                shutil.rmtree(calc_dir)

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            n_fail += 1
            failed_configs[str(idx)] = {
                "file": config_file,
                "config_type": config_type,
                "error": str(e),
            }
            # Save failed log incrementally
            with open(failed_log, "w") as f:
                json.dump(failed_configs, f, indent=2)

        # Progress summary every 50 configs
        if (i + 1) % 50 == 0:
            total_done = len(completed) + n_success + n_fail
            print(f"\n--- Progress: {total_done}/{len(manifest)} "
                  f"(+{n_success} ok, +{n_fail} fail) ---\n")

    print(f"\n{'=' * 60}")
    print(f"Results: {n_success} succeeded, {n_fail} failed")
    print(f"Total completed: {len(completed) + n_success}/{len(manifest)}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
