#!/usr/bin/env python3
"""Compute elastic properties using the trained DeePMD potential.

Estimates Young's modulus (E) and Poisson's ratio (nu) via strain-stress
analysis, then compares with MEAM reference values from src/ML/results.json.
"""

import json
import os
import sys

import numpy as np
from ase.build import bulk

PROJECT_DIR = "/home/kenobi/Workspaces/PHYS400"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model.pb")


def compute_elastic_constants(atoms, delta=1e-3):
    """Estimate C11 and C12 via finite-difference strain-stress.

    Applies small axial strain in x, measures stress response.
    Returns Young's modulus E and Poisson's ratio nu (cubic approximation).
    """
    from ase.constraints import UnitCellFilter
    from ase.optimize import BFGS

    # Get equilibrium stress
    s0 = atoms.get_stress(voigt=True)  # [xx, yy, zz, yz, xz, xy] in eV/A^3

    # Apply +delta strain in x
    strained = atoms.copy()
    strained.calc = atoms.calc
    cell = strained.get_cell().copy()
    cell[0] *= (1.0 + delta)
    strained.set_cell(cell, scale_atoms=True)

    s1 = strained.get_stress(voigt=True)

    # C11 = d(sigma_xx) / d(epsilon_xx), C12 = d(sigma_yy) / d(epsilon_xx)
    # ASE stress has negative-pressure convention, so subtract
    # Convert from eV/A^3 to GPa: 1 eV/A^3 = 160.2176634 GPa
    eV_to_GPa = 160.2176634

    c11 = -(s1[0] - s0[0]) / delta * eV_to_GPa
    c12 = -(s1[1] - s0[1]) / delta * eV_to_GPa

    # Cubic elastic moduli
    E = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)
    nu = c12 / (c11 + c12)

    return {
        "C11_GPa": float(c11),
        "C12_GPa": float(c12),
        "E_GPa": float(E),
        "nu": float(nu),
    }


def run_elastic_test(elements, fractions, label="test"):
    """Create an alloy supercell and compute elastic properties."""
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "NNIP"))
    from calculator import DeepMDCalculator

    # Build supercell (FCC Al base)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    n_atoms = len(atoms)
    symbols = list(atoms.get_chemical_symbols())

    # Substitute atoms according to fractions
    np.random.seed(42)
    remaining = 1.0
    for elem, frac in zip(elements[1:], fractions[1:]):
        n_sub = int(round(frac * n_atoms))
        # Find Al atoms to replace
        al_indices = [i for i, s in enumerate(symbols) if s == elements[0]]
        if n_sub > 0 and al_indices:
            replace = np.random.choice(al_indices, min(n_sub, len(al_indices)), replace=False)
            for idx in replace:
                symbols[idx] = elem

    atoms.set_chemical_symbols(symbols)
    atoms.calc = DeepMDCalculator()

    print(f"\n{label}: {dict(zip(elements, fractions))}")
    print(f"  Atoms: {n_atoms}")
    print(f"  Composition: {dict(sorted(zip(*np.unique(symbols, return_counts=True))))}")

    result = compute_elastic_constants(atoms)
    print(f"  C11 = {result['C11_GPa']:.2f} GPa")
    print(f"  C12 = {result['C12_GPa']:.2f} GPa")
    print(f"  E   = {result['E_GPa']:.2f} GPa")
    print(f"  nu  = {result['nu']:.3f}")

    return result


def compare_with_meam():
    """Compare NNIP elastic properties with MEAM reference from results.json."""
    results_path = os.path.join(PROJECT_DIR, "src", "ML", "results.json")
    if not os.path.isfile(results_path):
        print("No MEAM reference data found")
        return

    with open(results_path) as f:
        meam_data = json.load(f)

    print("\n" + "=" * 60)
    print("Comparison: NNIP vs MEAM")
    print("=" * 60)

    for name, ref in meam_data.items():
        comp = ref["composition"]
        elements = list(comp.keys())
        fractions = list(comp.values())

        try:
            nnip = run_elastic_test(elements, fractions, label=name)
            print(f"  MEAM: E={ref['E_GPa']:.2f} GPa, nu={ref['nu']:.3f}")
            print(f"  NNIP: E={nnip['E_GPa']:.2f} GPa, nu={nnip['nu']:.3f}")
            e_diff = abs(nnip["E_GPa"] - ref["E_GPa"])
            nu_diff = abs(nnip["nu"] - ref["nu"])
            print(f"  Diff: dE={e_diff:.2f} GPa, dnu={nu_diff:.3f}")
        except Exception as e:
            print(f"  FAILED: {e}")


def main():
    print("=" * 60)
    print("Elastic Property Computation (NNIP)")
    print("=" * 60)

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Train and freeze model first (train.sh)")
        sys.exit(1)

    # Test on pure Al
    sys.path.insert(0, os.path.join(PROJECT_DIR, "src", "NNIP"))
    from calculator import DeepMDCalculator

    al = bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3))
    al.calc = DeepMDCalculator()

    print("\nPure Al (FCC):")
    result = compute_elastic_constants(al)
    print(f"  C11 = {result['C11_GPa']:.2f} GPa")
    print(f"  C12 = {result['C12_GPa']:.2f} GPa")
    print(f"  E   = {result['E_GPa']:.2f} GPa  (exp: ~70 GPa)")
    print(f"  nu  = {result['nu']:.3f}  (exp: ~0.35)")

    # Compare with MEAM
    compare_with_meam()


if __name__ == "__main__":
    main()
