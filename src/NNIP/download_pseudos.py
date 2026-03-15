#!/usr/bin/env python3
"""Download and validate PBE PAW pseudopotentials for the 9-element Al alloy system.

Downloads from the QE pseudopotential library (pslibrary 1.0.0, PBE, PAW).
Validates each with a single-element SCF calculation on bulk crystal.

Elements: Al, Zn, Mg, Mn, Cu, Si, Cr, Fe, Ti
"""

import os
import shutil
import sys
import urllib.request

from ase.build import bulk
from ase.calculators.espresso import Espresso, EspressoProfile

QE_BIN = "/home/kenobi/Workspaces/qe/bin/pw.x"
PSEUDO_DIR = "/home/kenobi/Workspaces/PHYS400/pseudopotentials"
WORK_DIR = os.path.join(os.path.dirname(__file__), "pseudo_test_output")

BASE_URL = "https://pseudopotentials.quantum-espresso.org/upf_files"

# PBE PAW pseudopotentials from pslibrary
# Format: element -> (filename, bulk_structure, lattice_param_angstrom)
ELEMENTS = {
    "Al": ("Al.pbe-n-kjpaw_psl.1.0.0.UPF", "fcc", 4.05),
    "Zn": ("Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF", "hcp", 2.66),
    "Mg": ("Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF", "hcp", 3.21),
    "Mn": ("Mn.pbe-spn-kjpaw_psl.0.3.1.UPF", "bcc", 8.91),
    "Cu": ("Cu.pbe-dn-kjpaw_psl.1.0.0.UPF", "fcc", 3.61),
    "Si": ("Si.pbe-n-kjpaw_psl.1.0.0.UPF", "diamond", 5.43),
    "Cr": ("Cr.pbe-spn-kjpaw_psl.1.0.0.UPF", "bcc", 2.91),
    "Fe": ("Fe.pbe-spn-kjpaw_psl.1.0.0.UPF", "bcc", 2.87),
    "Ti": ("Ti.pbe-spn-kjpaw_psl.1.0.0.UPF", "hcp", 2.95),
}


def download_pseudopotentials():
    """Download missing pseudopotentials from the QE library."""
    os.makedirs(PSEUDO_DIR, exist_ok=True)
    downloaded = []
    skipped = []

    for elem, (filename, _, _) in ELEMENTS.items():
        filepath = os.path.join(PSEUDO_DIR, filename)
        if os.path.isfile(filepath):
            print(f"  {elem}: {filename} already exists, skipping")
            skipped.append(elem)
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"  {elem}: downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"       -> {size_kb:.0f} KB downloaded")
            downloaded.append(elem)
        except Exception as e:
            print(f"  FAIL: {elem}: {e}")
            # Try without the version suffix as fallback
            print(f"  URL was: {url}")

    return downloaded, skipped


def make_calculator(element, pseudo_file, directory):
    """Create a QE calculator for single-element validation."""
    profile = EspressoProfile(command=QE_BIN, pseudo_dir=PSEUDO_DIR)

    # Use higher ecutwfc for transition metals with semicore states
    ecutwfc = 40 if element in ("Mn", "Fe", "Cr", "Ti", "Cu", "Zn") else 30
    ecutrho = ecutwfc * 8

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
        },
    }

    # Add spin polarization for magnetic elements
    if element in ("Fe", "Cr", "Mn"):
        input_data["system"]["nspin"] = 2
        input_data["system"]["starting_magnetization(1)"] = 0.5

    return Espresso(
        profile=profile,
        pseudopotentials={element: pseudo_file},
        input_data=input_data,
        kpts=(6, 6, 6),
        directory=directory,
    )


def validate_element(element):
    """Run a single-element SCF test and check convergence."""
    filename, structure, a = ELEMENTS[element]
    calc_dir = os.path.join(WORK_DIR, element)

    # Build bulk crystal
    if structure == "diamond":
        atoms = bulk(element, "diamond", a=a)
    elif structure == "hcp":
        atoms = bulk(element, "hcp", a=a)
    else:
        atoms = bulk(element, structure, a=a)

    atoms.calc = make_calculator(element, filename, calc_dir)

    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = abs(forces).max()
        n_atoms = len(atoms)
        e_per_atom = energy / n_atoms

        print(f"  {element} ({structure}, a={a} A):")
        print(f"    Energy: {energy:.4f} eV ({e_per_atom:.4f} eV/atom)")
        print(f"    Max |F|: {max_force:.6f} eV/A")
        print(f"    Atoms: {n_atoms}")

        # Basic sanity checks
        if max_force > 0.5:
            print(f"    WARNING: Large forces — structure may not be at equilibrium")
        if abs(e_per_atom) < 1.0:
            print(f"    WARNING: Suspiciously small energy per atom")

        return True
    except Exception as e:
        print(f"  {element}: FAIL — {e}")
        return False


def main():
    print("=" * 60)
    print("Pseudopotential Download & Validation")
    print(f"Elements: {', '.join(ELEMENTS.keys())}")
    print("=" * 60)

    if not os.path.isfile(QE_BIN):
        print(f"ERROR: pw.x not found at {QE_BIN}")
        sys.exit(1)

    # Step 1: Download
    print("\n[1] Downloading pseudopotentials...")
    downloaded, skipped = download_pseudopotentials()
    print(f"\n    Downloaded: {len(downloaded)}, Already present: {len(skipped)}")

    # Check all files exist
    missing = []
    for elem, (filename, _, _) in ELEMENTS.items():
        if not os.path.isfile(os.path.join(PSEUDO_DIR, filename)):
            missing.append(elem)
    if missing:
        print(f"\nERROR: Missing pseudopotentials for: {', '.join(missing)}")
        print("Download these manually and re-run validation.")
        sys.exit(1)

    # Step 2: Validate
    print("\n[2] Validating with SCF calculations...")
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

    passed = 0
    failed = 0
    for element in ELEMENTS:
        print()
        ok = validate_element(element)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Validation: {passed} passed, {failed} failed out of {len(ELEMENTS)}")
    print("=" * 60)

    # Cleanup on success
    if failed == 0 and os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
        print("Test output cleaned up.")

    sys.exit(failed)


if __name__ == "__main__":
    main()
