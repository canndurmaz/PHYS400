#!/usr/bin/env python3
"""Parse magnetic moments from completed pw.x SCF outputs and inject them
into dft_results.json under elements.<sym>.magnetic.

For each magnetic element in MAGNETIC_ELEMENTS, walk the EOS-strain-0 (and
elastic baseline) scratch dirs, find espresso.pwo, and extract:
  - total magnetization (μB/cell)
  - absolute magnetization (μB/cell)
  - per-atom local moments
  - SCF convergence iterations

This is post-processing — it does NOT re-run any DFT. Run it after
generate_dft_reference completes.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH_DIR = os.path.join(HERE, "dft_scratch")
DFT_JSON = os.path.join(HERE, "dft_results.json")

# Match the last SCF iteration's reported magnetic state in a pw.x output.
_RE_TOTAL_MAG = re.compile(r"total magnetization\s*=\s*([-\d.]+)\s*Bohr")
_RE_ABS_MAG = re.compile(r"absolute magnetization\s*=\s*([\d.]+)\s*Bohr")
_RE_ATOM_MAG = re.compile(r"^\s*atom\s+(\d+)\s+\(R=[\d.]+\)\s+charge=\s*[-\d.]+\s+magn=\s*([-\d.]+)", re.M)
_RE_CONVERGED = re.compile(r"convergence has been achieved in\s+(\d+)\s+iterations")


def parse_pwo(path):
    """Return dict with magnetic data from the FINAL converged SCF iteration,
    or None if the file is missing/unconverged."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        text = f.read()

    # Take the last converged iteration. If the file has multiple SCF cycles
    # (e.g. a relaxation), the last block is the final state.
    conv_match = _RE_CONVERGED.findall(text)
    if not conv_match:
        return None
    n_iter = int(conv_match[-1])

    # Last total/abs magnetization printout — these are printed per SCF
    # iteration in the running SCF block.
    total_mag = float(_RE_TOTAL_MAG.findall(text)[-1]) if _RE_TOTAL_MAG.findall(text) else None
    abs_mag = float(_RE_ABS_MAG.findall(text)[-1]) if _RE_ABS_MAG.findall(text) else None

    # Per-atom moments: only printed when "report=1" is set OR after the last
    # SCF iteration in some pw.x versions. Grab the LAST block of atom-magn
    # lines that share a contiguous block, by parsing from the end backwards.
    atoms = _RE_ATOM_MAG.findall(text)
    per_atom = None
    if atoms:
        # Group by the largest contiguous trailing block (same atom 1, 2, ...)
        # parsed in file order — take the final occurrence of each atom index.
        latest = {}
        for idx, mag in atoms:
            latest[int(idx)] = float(mag)
        per_atom = [latest[k] for k in sorted(latest)]

    return {
        "total_mag_per_cell": total_mag,
        "absolute_mag_per_cell": abs_mag,
        "per_atom": per_atom,
        "scf_iterations": n_iter,
    }


def find_eos_pwo(element, strain_idx=3):
    """Return path to the central EOS strain point's pwo file (equilibrium-ish)."""
    return os.path.join(SCRATCH_DIR, element, f"eos_{strain_idx}", "espresso.pwo")


def find_elastic_pwo(element, kind="base"):
    """Return path to an elastic SCF's pwo (kind = 'base' or 'strain')."""
    return os.path.join(SCRATCH_DIR, element, f"elastic_{kind}", "espresso.pwo")


def main():
    if not os.path.isfile(DFT_JSON):
        print(f"ERROR: {DFT_JSON} not found", file=sys.stderr)
        return 1
    with open(DFT_JSON) as f:
        dft = json.load(f)

    # Hard-code the magnetic-element list to match dft_reference.py
    magnetic_elements = {"Fe", "Cr", "Mn", "Co", "Ni"}
    updated = []

    for sym in magnetic_elements:
        if sym not in dft.get("elements", {}):
            print(f"  {sym}: not in dft_results.json, skip", file=sys.stderr)
            continue

        eos_pwo = find_eos_pwo(sym, strain_idx=3)
        elastic_base = find_elastic_pwo(sym, "base")
        elastic_strain = find_elastic_pwo(sym, "strain")

        magnetic = {}
        eos_data = parse_pwo(eos_pwo)
        if eos_data:
            magnetic["eos_equilibrium"] = eos_data
        else:
            print(f"  {sym}: eos_3 pwo missing/unconverged", file=sys.stderr)

        eb = parse_pwo(elastic_base)
        if eb:
            magnetic["elastic_baseline"] = eb
        es = parse_pwo(elastic_strain)
        if es:
            magnetic["elastic_strained"] = es

        if magnetic:
            dft["elements"][sym]["magnetic"] = magnetic
            tot = magnetic.get("eos_equilibrium", {}).get("total_mag_per_cell", "?")
            abs_ = magnetic.get("eos_equilibrium", {}).get("absolute_mag_per_cell", "?")
            pa = magnetic.get("eos_equilibrium", {}).get("per_atom", [])
            per_atom_str = (f"per-atom range "
                            f"[{min(pa):+.2f}, {max(pa):+.2f}]" if pa else "")
            print(f"  {sym}: total={tot} μB, abs={abs_} μB, {per_atom_str}")
            updated.append(sym)

    if updated:
        with open(DFT_JSON, "w") as f:
            json.dump(dft, f, indent=2)
        print(f"\nUpdated {len(updated)} elements in {DFT_JSON}: {updated}")
    else:
        print("\nNo updates written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
