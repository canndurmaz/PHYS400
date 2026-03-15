#!/usr/bin/env python3
"""ASE Calculator wrapper for the DeePMD-kit neural network potential.

Provides an ASE-compatible calculator that can be used as a drop-in
replacement for QE or LAMMPS calculators. Useful for:
- Quick testing without full LAMMPS rebuild
- Integration with ASE optimization/MD routines
- Comparison with DFT reference calculations
"""

import os

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

PROJECT_DIR = "/home/kenobi/Workspaces/PHYS400"
DEFAULT_MODEL = os.path.join(PROJECT_DIR, "models", "model.pb")


class DeepMDCalculator(Calculator):
    """ASE Calculator using a frozen DeePMD-kit model."""

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model_path=None, type_map=None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path or DEFAULT_MODEL
        self.type_map = type_map or [
            "Al", "Zn", "Mg", "Mn", "Cu", "Si", "Cr", "Fe", "Ti"
        ]
        self._dp = None

    def _load_model(self):
        if self._dp is None:
            from deepmd.infer import DeepPot
            self._dp = DeepPot(self.model_path)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._load_model()

        symbols = self.atoms.get_chemical_symbols()
        types = np.array([self.type_map.index(s) for s in symbols])
        coords = self.atoms.get_positions().reshape(1, -1)
        cell = self.atoms.get_cell().array.reshape(1, -1)

        e, f, v = self._dp.eval(coords, cell, types)

        self.results["energy"] = float(e[0, 0])
        self.results["forces"] = f[0].reshape(-1, 3)

        # Convert virial to ASE stress convention (Voigt, eV/A^3, negative pressure)
        if v is not None and v.size > 0:
            volume = self.atoms.get_volume()
            virial = v[0].reshape(3, 3)
            # ASE stress = -virial / volume, in Voigt order
            stress = -virial / volume
            self.results["stress"] = np.array([
                stress[0, 0], stress[1, 1], stress[2, 2],
                stress[1, 2], stress[0, 2], stress[0, 1],
            ])


def test():
    """Quick test of the calculator on a bulk Al cell."""
    from ase.build import bulk

    if not os.path.isfile(DEFAULT_MODEL):
        print(f"Model not found at {DEFAULT_MODEL}")
        print("Train and freeze model first (train.sh)")
        return

    al = bulk("Al", "fcc", a=4.05, cubic=True)
    al.calc = DeepMDCalculator()

    e = al.get_potential_energy()
    f = al.get_forces()
    s = al.get_stress()

    print(f"Energy: {e:.4f} eV ({e/len(al):.4f} eV/atom)")
    print(f"Max |F|: {np.abs(f).max():.6f} eV/A")
    print(f"Pressure: {-s[:3].mean() / 0.00624:.2f} GPa")


if __name__ == "__main__":
    test()
