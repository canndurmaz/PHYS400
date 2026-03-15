#!/usr/bin/env python3
"""Convert DFT results (extended XYZ) to DeePMD-kit raw format.

Reads from data/training/dft_results/ and data/validation/dft_results/
Outputs to data/training/deepmd/ and data/validation/deepmd/

DeePMD raw format:
  type.raw       - element type indices (per atom)
  set.000/
    energy.npy   - total energy per frame (nframes,)
    force.npy    - forces (nframes, natoms*3)
    coord.npy    - coordinates (nframes, natoms*3)
    box.npy      - cell vectors (nframes, 9)
    virial.npy   - virial tensor (nframes, 9) [optional]
"""

import os
import sys
from glob import glob

import numpy as np
from ase.io import read

PROJECT_DIR = "/home/kenobi/Workspaces/PHYS400"

TYPE_MAP = ["Al", "Zn", "Mg", "Mn", "Cu", "Si", "Cr", "Fe", "Ti"]


def convert_dataset(input_dir, output_dir, label="training"):
    """Convert a directory of extended XYZ files to DeePMD format."""
    files = sorted(glob(os.path.join(input_dir, "*.xyz")))
    if not files:
        print(f"  No XYZ files found in {input_dir}")
        return 0

    # Group by number of atoms (DeePMD requires same natoms per system)
    groups = {}
    for f in files:
        try:
            atoms = read(f, format="extxyz")
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}")
            continue

        # Check for required data
        energy = atoms.info.get("dft_energy", atoms.info.get("energy", None))
        forces = atoms.arrays.get("dft_forces", atoms.arrays.get("forces", None))

        if energy is None or forces is None:
            print(f"  WARNING: Missing energy/forces in {f}, skipping")
            continue

        n_atoms = len(atoms)
        if n_atoms not in groups:
            groups[n_atoms] = []
        groups[n_atoms].append(atoms)

    total = sum(len(v) for v in groups.values())
    print(f"  {label}: {total} configurations in {len(groups)} groups (by atom count)")

    # Write each group as a separate DeePMD system
    for n_atoms, configs in sorted(groups.items()):
        sys_dir = os.path.join(output_dir, f"sys_{n_atoms:03d}")
        set_dir = os.path.join(sys_dir, "set.000")
        os.makedirs(set_dir, exist_ok=True)

        n_frames = len(configs)
        energies = np.zeros(n_frames)
        forces_arr = np.zeros((n_frames, n_atoms * 3))
        coords_arr = np.zeros((n_frames, n_atoms * 3))
        boxes_arr = np.zeros((n_frames, 9))
        virials_arr = np.zeros((n_frames, 9))
        has_virial = False

        # Use first config to determine type mapping
        first_symbols = configs[0].get_chemical_symbols()
        type_indices = [TYPE_MAP.index(s) for s in first_symbols]

        for i, atoms in enumerate(configs):
            symbols = atoms.get_chemical_symbols()
            # Verify consistent atom types
            current_types = [TYPE_MAP.index(s) for s in symbols]
            if current_types != type_indices:
                # Sort atoms by type for consistency
                order = sorted(range(n_atoms), key=lambda x: current_types[x])
                atoms = atoms[order]
                symbols = atoms.get_chemical_symbols()
                current_types = [TYPE_MAP.index(s) for s in symbols]
                type_indices = current_types

            energies[i] = atoms.info.get("dft_energy", atoms.info.get("energy"))
            f = atoms.arrays.get("dft_forces", atoms.arrays.get("forces"))
            forces_arr[i] = f.flatten()
            coords_arr[i] = atoms.get_positions().flatten()
            boxes_arr[i] = atoms.get_cell().array.flatten()

            stress = atoms.info.get("dft_stress", atoms.info.get("stress", None))
            if stress is not None:
                # Convert stress (3x3) to virial = -stress * volume
                stress_arr = np.array(stress).reshape(3, 3)
                vol = atoms.get_volume()
                virial = -stress_arr * vol  # eV
                virials_arr[i] = virial.flatten()
                has_virial = True

        # Write type.raw
        with open(os.path.join(sys_dir, "type.raw"), "w") as f:
            for t in type_indices:
                f.write(f"{t}\n")

        # Write type_map.raw
        with open(os.path.join(sys_dir, "type_map.raw"), "w") as f:
            for elem in TYPE_MAP:
                f.write(f"{elem}\n")

        # Write numpy arrays
        np.save(os.path.join(set_dir, "energy.npy"), energies)
        np.save(os.path.join(set_dir, "force.npy"), forces_arr)
        np.save(os.path.join(set_dir, "coord.npy"), coords_arr)
        np.save(os.path.join(set_dir, "box.npy"), boxes_arr)
        if has_virial:
            np.save(os.path.join(set_dir, "virial.npy"), virials_arr)

        print(f"    sys_{n_atoms:03d}: {n_frames} frames, {n_atoms} atoms")

    return total


def main():
    print("=" * 60)
    print("Convert DFT Results to DeePMD Format")
    print("=" * 60)

    total = 0

    # Training data
    train_in = os.path.join(PROJECT_DIR, "data", "training", "dft_results")
    train_out = os.path.join(PROJECT_DIR, "data", "training", "deepmd")
    if os.path.isdir(train_in):
        print("\n[Training Data]")
        n = convert_dataset(train_in, train_out, "training")
        total += n
    else:
        print(f"Training data not found at {train_in}")
        print("Run run_dft.py first")

    # Validation data
    val_in = os.path.join(PROJECT_DIR, "data", "validation", "dft_results")
    val_out = os.path.join(PROJECT_DIR, "data", "validation", "deepmd")
    if os.path.isdir(val_in):
        print("\n[Validation Data]")
        n = convert_dataset(val_in, val_out, "validation")
        total += n

    print(f"\nTotal converted: {total} configurations")
    print("Ready for dp train")


if __name__ == "__main__":
    main()
