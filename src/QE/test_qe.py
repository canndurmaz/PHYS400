#!/usr/bin/env python3
"""Test script for Quantum Espresso + ASE integration.

Runs a simple SCF calculation on bulk FCC Aluminum and verifies
that energy, forces, and stress are computed correctly.
"""

import os
import shutil
import sys

from ase.build import bulk
from ase.calculators.espresso import Espresso, EspressoProfile

QE_BIN = "/home/kenobi/Workspaces/qe/bin/pw.x"
PSEUDO_DIR = "/home/kenobi/Workspaces/PHYS400/pseudopotentials"
WORK_DIR = os.path.join(os.path.dirname(__file__), "test_output")

# Known reference: Al FCC energy should be around -537 eV for these settings
ENERGY_RANGE = (-540.0, -530.0)  # eV, loose bounds


def make_calculator(directory):
    profile = EspressoProfile(command=QE_BIN, pseudo_dir=PSEUDO_DIR)
    return Espresso(
        profile=profile,
        pseudopotentials={"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
        input_data={
            "control": {
                "tprnfor": True,
                "tstress": True,
            },
            "system": {
                "ecutwfc": 30,
                "ecutrho": 240,
                "occupations": "smearing",
                "smearing": "mv",
                "degauss": 0.02,
            },
            "electrons": {
                "conv_thr": 1.0e-6,
            },
        },
        kpts=(4, 4, 4),
        directory=directory,
    )


def test_energy():
    """Test that SCF energy is in the expected range."""
    al = bulk("Al", "fcc", a=4.05)
    al.calc = make_calculator(os.path.join(WORK_DIR, "energy"))
    energy = al.get_potential_energy()
    assert ENERGY_RANGE[0] < energy < ENERGY_RANGE[1], (
        f"Energy {energy:.4f} eV outside expected range {ENERGY_RANGE}"
    )
    print(f"  Energy:  {energy:.4f} eV  [PASS]")
    return energy


def test_forces():
    """Test that forces are near zero for a perfect crystal."""
    al = bulk("Al", "fcc", a=4.05)
    al.calc = make_calculator(os.path.join(WORK_DIR, "forces"))
    forces = al.get_forces()
    max_force = abs(forces).max()
    assert max_force < 0.01, f"Max force {max_force:.6f} eV/A too large for perfect crystal"
    print(f"  Forces:  max |F| = {max_force:.6f} eV/A  [PASS]")
    return forces


def test_stress():
    """Test that stress tensor is computed and hydrostatic."""
    al = bulk("Al", "fcc", a=4.05)
    al.calc = make_calculator(os.path.join(WORK_DIR, "stress"))
    stress = al.get_stress()  # Voigt: xx, yy, zz, yz, xz, xy
    # For cubic crystal, diagonal components should be roughly equal
    diag = stress[:3]
    off_diag = stress[3:]
    assert abs(off_diag).max() < 0.001, f"Off-diagonal stress too large: {off_diag}"
    spread = diag.max() - diag.min()
    assert spread < 0.001, f"Diagonal stress spread {spread:.6f} too large for cubic cell"
    pressure_GPa = -sum(diag) / 3 / 0.00624  # eV/A^3 -> GPa
    print(f"  Stress:  P = {pressure_GPa:.2f} GPa, spread = {spread:.2e}  [PASS]")
    return stress


def main():
    print("=" * 50)
    print("Quantum Espresso + ASE Integration Test")
    print("=" * 50)

    # Check prerequisites
    if not os.path.isfile(QE_BIN):
        print(f"FAIL: pw.x not found at {QE_BIN}")
        sys.exit(1)

    pseudo = os.path.join(PSEUDO_DIR, "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    if not os.path.isfile(pseudo):
        print(f"FAIL: pseudopotential not found at {pseudo}")
        sys.exit(1)

    # Clean previous test output
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

    passed = 0
    failed = 0

    for name, test_fn in [
        ("SCF Energy", test_energy),
        ("Forces", test_forces),
        ("Stress", test_stress),
    ]:
        print(f"\n[{name}]")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    # Cleanup
    if failed == 0 and os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
        print("Test output cleaned up.")

    sys.exit(failed)


if __name__ == "__main__":
    main()
