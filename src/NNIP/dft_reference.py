#!/usr/bin/env python3
"""DFT reference data generator using Quantum Espresso + ASE.

For each selected element:
  - EOS fit → equilibrium lattice constant, cohesive energy
  - Stress-strain → C11, C12 (cubic elements)

For binary pairs:
  - Formation energy in L1₂ (FCC) or B2 (BCC) structure

Parallelism:
  - EOS strain points run in parallel (7 pw.x at once per element)
  - Binary pair SCF runs in parallel (all pairs at once)
  - Controlled via --parallel N flag (default: 4 workers)

Output: dft_results.json
"""

import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from ase.build import bulk
from ase.eos import EquationOfState
from ase.calculators.espresso import Espresso, EspressoProfile

QE_BIN = "/home/kenobi/Workspaces/qe/bin/pw.x"
PSEUDO_DIR = "/home/kenobi/Workspaces/PHYS400/pseudopotentials"

# Element metadata: symbol → (lattice_type, approx_lattice_constant, pseudopotential_file)
ELEMENT_DATA = {
    "Al": ("fcc", 4.05, "Al.pbe-n-kjpaw_psl.1.0.0.UPF"),
    "Cu": ("fcc", 3.61, "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF"),
    "Fe": ("bcc", 2.87, "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Zn": ("hcp", 2.66, "Zn.pbe-dnl-kjpaw_psl.1.0.0.UPF"),
    "Mg": ("hcp", 3.21, "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF"),
    "Ti": ("hcp", 2.95, "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Cr": ("bcc", 2.91, "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Mn": ("bcc", 8.91, "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF"),  # alpha-Mn is complex; use BCC approx
    "Si": ("diamond", 5.43, "Si.pbe-n-kjpaw_psl.1.0.0.UPF"),
    "Au": ("fcc", 4.08, "Au.pz-rrkjus_aewfc.UPF"),
    "Mo": ("bcc", 3.15, "Mo-PBE.upf"),
}

# Elements requiring spin-polarized calculation
MAGNETIC_ELEMENTS = {"Fe", "Cr", "Mn"}

# Reference structure for binary pairs.
# Keys are sorted tuples of lattice types.
_BINARY_LATTICE = {
    ("bcc", "bcc"): "b2",
    ("bcc", "diamond"): "b2",
    ("bcc", "fcc"): "b2",
    ("bcc", "hcp"): "b2",
    ("diamond", "diamond"): "b2",
    ("diamond", "fcc"): "l12",
    ("diamond", "hcp"): "l12",
    ("fcc", "fcc"): "l12",
    ("fcc", "hcp"): "l12",
    ("hcp", "hcp"): "b2",
}


def _make_calculator(pseudopotentials, directory, magnetic=False):
    """Create an Espresso calculator for SCF."""
    profile = EspressoProfile(command=QE_BIN, pseudo_dir=PSEUDO_DIR)
    input_data = {
        "control": {"tprnfor": True, "tstress": True},
        "system": {
            "ecutwfc": 40,
            "ecutrho": 320,
            "occupations": "smearing",
            "smearing": "mv",
            "degauss": 0.02,
        },
        "electrons": {"conv_thr": 1.0e-6},
    }
    if magnetic:
        input_data["system"]["nspin"] = 2
    return Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=(6, 6, 6),
        directory=directory,
    )


# ── Single EOS point (top-level function for pickling) ───────────────────────

def _run_eos_point(symbol, pseudo, ase_lat, a, calc_dir, magnetic, strain_idx):
    """Run a single EOS strain point. Returns (strain_idx, volume, energy) or (strain_idx, None, None)."""
    try:
        atoms = bulk(symbol, ase_lat, a=a)
        atoms.calc = _make_calculator({symbol: pseudo}, calc_dir, magnetic=magnetic)
        t0 = time.time()
        e = atoms.get_potential_energy()
        v = atoms.get_volume()
        dt = time.time() - t0
        print(f"    [SCF] EOS point {strain_idx}: a={a:.4f}  E={e:.4f} eV  V={v:.3f} A^3  ({dt:.1f}s)", flush=True)
        return (strain_idx, v, e)
    except Exception as exc:
        print(f"    [SCF] EOS point {strain_idx}: a={a:.4f}  FAILED: {exc}", flush=True)
        return (strain_idx, None, None)


def _eos_fit(symbol, work_dir, n_workers=4):
    """Equation of state fit → (a0, E_coh, B) for one element.

    Runs 7 strain-point SCF calculations in parallel.
    """
    data = ELEMENT_DATA[symbol]
    lat, a_guess, pseudo = data
    magnetic = symbol in MAGNETIC_ELEMENTS

    print(f"  [EOS] Starting EOS fit for {symbol}", flush=True)
    print(f"  [EOS]   lattice={lat}, a_guess={a_guess} A, pseudo={pseudo}, magnetic={magnetic}", flush=True)
    print(f"  [EOS]   scratch dir: {work_dir}", flush=True)

    if lat in ("hcp", "hex"):
        ase_lat = "fcc"
        a_guess_ase = a_guess * math.sqrt(2)
        print(f"  [EOS]   HCP -> FCC proxy, a_ase={a_guess_ase:.4f} A", flush=True)
    elif lat == "diamond":
        ase_lat = "diamond"
        a_guess_ase = a_guess
    else:
        ase_lat = lat
        a_guess_ase = a_guess

    strains = np.linspace(0.96, 1.04, 7)
    eos_workers = min(n_workers, 7)
    print(f"  [EOS]   Running 7 SCF points in parallel ({eos_workers} workers)...", flush=True)

    t_eos_start = time.time()

    # Submit all 7 strain points in parallel
    eos_results = []
    with ProcessPoolExecutor(max_workers=eos_workers) as pool:
        futures = {}
        for i, s in enumerate(strains):
            a = a_guess_ase * s
            calc_dir = os.path.join(work_dir, f"eos_{i}")
            os.makedirs(calc_dir, exist_ok=True)
            fut = pool.submit(_run_eos_point, symbol, pseudo, ase_lat, a, calc_dir, magnetic, i)
            futures[fut] = i

        for fut in as_completed(futures):
            eos_results.append(fut.result())

    # Sort by strain index, filter successes
    eos_results.sort(key=lambda x: x[0])
    volumes = [v for _, v, e in eos_results if v is not None]
    energies = [e for _, v, e in eos_results if v is not None]

    t_eos = time.time() - t_eos_start
    print(f"  [EOS]   7 points finished in {t_eos:.1f}s ({len(volumes)} succeeded)", flush=True)

    if len(volumes) < 5:
        print(f"  [EOS]   WARNING: Only {len(volumes)} valid points, need 5. Using fallback.", flush=True)
        return a_guess, 3.0, 70.0

    print(f"  [EOS]   Fitting Birch-Murnaghan EOS...", flush=True)
    eos = EquationOfState(volumes, energies, "birchmurnaghan")
    v0, e0, B = eos.fit()
    print(f"  [EOS]   V0={v0:.3f} A^3, E0={e0:.4f} eV, B_raw={B:.6f} eV/A^3", flush=True)

    n_atoms = 4 if ase_lat == "fcc" else (2 if ase_lat == "bcc" else 8 if ase_lat == "diamond" else 4)
    a0 = (v0 / (n_atoms / 4.0)) ** (1.0 / 3.0) if ase_lat == "fcc" else (v0 * 2) ** (1.0 / 3.0)

    if lat in ("hcp", "hex"):
        a0 = a0 / math.sqrt(2)

    E_coh = -e0 / n_atoms
    B_GPa = B / 0.00624

    print(f"  [EOS] === Result: a0={a0:.4f} A, E_coh={E_coh:.4f} eV, B={B_GPa:.1f} GPa ===", flush=True)
    return a0, E_coh, B_GPa


def _elastic_constants(symbol, a0, work_dir):
    """C11, C12 via stress-strain for cubic elements."""
    data = ELEMENT_DATA[symbol]
    lat, _, pseudo = data
    magnetic = symbol in MAGNETIC_ELEMENTS

    if lat not in ("fcc", "bcc"):
        print(f"  [ELASTIC] Skipping {symbol} (lattice={lat}, not cubic)", flush=True)
        return None, None

    print(f"  [ELASTIC] Computing C11, C12 for {symbol} ({lat}, a0={a0:.4f} A)", flush=True)

    # Baseline stress
    print(f"  [ELASTIC] Step 1/2: Baseline stress...", flush=True)
    atoms = bulk(symbol, lat, a=a0)
    calc_dir = os.path.join(work_dir, "elastic_base")
    atoms.calc = _make_calculator({symbol: pseudo}, calc_dir, magnetic=magnetic)

    try:
        t0 = time.time()
        stress0 = atoms.get_stress()
        print(f"  [ELASTIC]   Baseline done in {time.time()-t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"  [ELASTIC]   Baseline FAILED: {exc}", flush=True)
        return None, None

    # Strained
    delta = 0.01
    print(f"  [ELASTIC] Step 2/2: Strained stress (delta={delta})...", flush=True)
    cell = atoms.get_cell().copy()
    cell[0, 0] *= (1 + delta)
    atoms_strained = atoms.copy()
    atoms_strained.set_cell(cell, scale_atoms=True)
    calc_dir_s = os.path.join(work_dir, "elastic_strain")
    atoms_strained.calc = _make_calculator({symbol: pseudo}, calc_dir_s, magnetic=magnetic)

    try:
        t0 = time.time()
        stress1 = atoms_strained.get_stress()
        print(f"  [ELASTIC]   Strained done in {time.time()-t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"  [ELASTIC]   Strained FAILED: {exc}", flush=True)
        return None, None

    to_GPa = 1.0 / 0.00624
    C11 = abs((stress1[0] - stress0[0]) / delta * to_GPa)
    C12 = abs((stress1[1] - stress0[1]) / delta * to_GPa)

    print(f"  [ELASTIC] === Result: C11={C11:.1f} GPa, C12={C12:.1f} GPa ===", flush=True)
    return C11, C12


# ── Single binary pair (top-level function for pickling) ─────────────────────

def _norm_lat(l):
    """Normalize lattice type for lookup."""
    l = l.lower()
    if l in ("hex",):
        return "hcp"
    if l in ("dia",):
        return "diamond"
    return l


def _run_binary_pair(sym_i, sym_j, work_dir, e_per_atom):
    """Run a single binary pair SCF. Returns (pair_key, E_form) or (pair_key, None)."""
    from ase import Atoms

    data_i = ELEMENT_DATA.get(sym_i)
    data_j = ELEMENT_DATA.get(sym_j)
    if not data_i or not data_j:
        return (f"{sym_i}-{sym_j}", None)

    lat_i, a_i, pseudo_i = data_i
    lat_j, a_j, pseudo_j = data_j

    key = tuple(sorted([_norm_lat(lat_i), _norm_lat(lat_j)]))
    ref_lat = _BINARY_LATTICE.get(key, "l12")
    a_mix = (a_i + a_j) / 2.0
    magnetic = sym_i in MAGNETIC_ELEMENTS or sym_j in MAGNETIC_ELEMENTS

    print(f"  [BINARY] {sym_i}-{sym_j}: {ref_lat}, a_mix={a_mix:.4f}, magnetic={magnetic}", flush=True)

    if ref_lat == "l12":
        a = a_mix
        positions = [
            [0, 0, 0], [0.5 * a, 0.5 * a, 0],
            [0.5 * a, 0, 0.5 * a], [0, 0.5 * a, 0.5 * a]
        ]
        symbols = [sym_i, sym_i, sym_i, sym_j]
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    elif ref_lat == "b2":
        a = a_mix
        positions = [[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]]
        symbols = [sym_i, sym_j]
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
    else:
        a = a_mix
        positions = [[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]]
        symbols = [sym_i, sym_j]
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    pseudopotentials = {sym_i: pseudo_i, sym_j: pseudo_j}
    calc_dir = os.path.join(work_dir, f"binary_{sym_i}_{sym_j}")
    os.makedirs(calc_dir, exist_ok=True)
    atoms.calc = _make_calculator(pseudopotentials, calc_dir, magnetic=magnetic)

    pair_key = f"{sym_i}-{sym_j}"
    try:
        t0 = time.time()
        e_total = atoms.get_potential_energy()
        dt = time.time() - t0
        n_atoms = len(atoms)
        e_mix = e_total / n_atoms
        n_i = symbols.count(sym_i)
        n_j = symbols.count(sym_j)
        e_ref = (n_i * e_per_atom.get(sym_i, 0) + n_j * e_per_atom.get(sym_j, 0)) / n_atoms
        e_form = e_mix - e_ref
        print(f"  [BINARY] {pair_key}: E_form={e_form:.4f} eV/atom ({dt:.1f}s)", flush=True)
        return (pair_key, e_form)
    except Exception as exc:
        print(f"  [BINARY] {pair_key}: FAILED ({exc})", flush=True)
        return (pair_key, None)


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_dft_reference(elements, output_path=None, work_dir=None, n_workers=4):
    """Run DFT calculations for selected elements and generate reference data.

    Args:
        elements: list of element symbols
        output_path: path for output JSON (default: dft_results.json next to this file)
        work_dir: scratch directory for QE calculations
        n_workers: max parallel pw.x processes (default: 4)

    Returns:
        dict with DFT reference data
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "dft_results.json")
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(__file__), "dft_scratch")
    os.makedirs(work_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"DFT Reference Generator", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Elements:        {elements}", flush=True)
    print(f"  Output:          {output_path}", flush=True)
    print(f"  Scratch:         {work_dir}", flush=True)
    print(f"  Parallel workers: {n_workers}", flush=True)
    print(f"  pw.x:            {QE_BIN} ({'found' if os.path.isfile(QE_BIN) else 'NOT FOUND'})", flush=True)

    valid = [e for e in elements if e in ELEMENT_DATA]
    skipped = [e for e in elements if e not in ELEMENT_DATA]
    if skipped:
        print(f"  Skipping (no metadata): {skipped}", flush=True)
    print(f"  Valid elements:  {valid}", flush=True)

    n_eos = len(valid) * 7
    n_elastic = sum(1 for e in valid if ELEMENT_DATA[e][0] in ("fcc", "bcc")) * 2
    n_binary = len(valid) * (len(valid) - 1) // 2
    print(f"\n  Planned: {n_eos} EOS + {n_elastic} elastic + {n_binary} binary = ~{n_eos+n_elastic+n_binary} SCF runs", flush=True)
    print(f"  EOS runs parallelized ({min(n_workers, 7)} at a time per element)", flush=True)
    print(f"  Binary pairs parallelized ({min(n_workers, n_binary)} at a time)", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {"elements": {}, "binary_pairs": {}}
    e_per_atom = {}
    t_total_start = time.time()

    # ── Stage 1: Single-element EOS + elastic constants ──────────────────
    # Elements run sequentially (elastic depends on EOS a0),
    # but EOS strain points within each element run in parallel.
    for idx, sym in enumerate(valid, 1):
        print(f"\n{'─'*60}", flush=True)
        print(f"  ELEMENT {idx}/{len(valid)}: {sym}", flush=True)
        print(f"{'─'*60}", flush=True)
        elem_dir = os.path.join(work_dir, sym)
        os.makedirs(elem_dir, exist_ok=True)

        t_elem_start = time.time()
        a0, E_coh, B = _eos_fit(sym, elem_dir, n_workers=n_workers)

        entry = {
            "a_lat": round(a0, 4),
            "E_coh": round(E_coh, 4),
            "B_GPa": round(B, 1),
            "lattice": ELEMENT_DATA[sym][0],
        }

        C11, C12 = _elastic_constants(sym, a0, elem_dir)
        if C11 is not None:
            entry["C11"] = round(C11, 1)
            entry["C12"] = round(C12, 1)

        results["elements"][sym] = entry
        print(f"  Element {sym} done in {time.time()-t_elem_start:.1f}s", flush=True)

        # Save intermediate
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [SAVE] Intermediate results -> {output_path}", flush=True)

        lat = ELEMENT_DATA[sym][0]
        n_atoms = 4 if lat == "fcc" else (2 if lat == "bcc" else 8 if lat == "diamond" else 2)
        e_per_atom[sym] = -E_coh

    # ── Stage 2: Binary pairs — ALL in parallel ──────────────────────────
    pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i+1, len(valid))]

    if pairs:
        print(f"\n{'─'*60}", flush=True)
        print(f"  BINARY PAIRS: {len(pairs)} total, running {min(n_workers, len(pairs))} in parallel", flush=True)
        print(f"{'─'*60}", flush=True)

        t_pairs_start = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for sym_i, sym_j in pairs:
                fut = pool.submit(_run_binary_pair, sym_i, sym_j, work_dir, e_per_atom)
                futures[fut] = (sym_i, sym_j)

            done_count = 0
            for fut in as_completed(futures):
                pair_key, e_form = fut.result()
                done_count += 1
                if e_form is not None:
                    results["binary_pairs"][pair_key] = {"E_form": round(e_form, 4)}
                print(f"  [{done_count}/{len(pairs)}] {pair_key}: {'OK' if e_form is not None else 'FAILED'}", flush=True)

                # Save after each completion
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

        t_pairs = time.time() - t_pairs_start
        print(f"  All {len(pairs)} pairs done in {t_pairs:.1f}s ({t_pairs/60:.1f} min)", flush=True)

    # ── Final summary ────────────────────────────────────────────────────
    t_total = time.time() - t_total_start
    print(f"\n{'='*60}", flush=True)
    print(f"DFT REFERENCE COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total wall time: {t_total:.1f}s ({t_total/60:.1f} min)", flush=True)
    print(f"  Elements: {list(results['elements'].keys())}", flush=True)
    print(f"  Pairs:    {list(results['binary_pairs'].keys())}", flush=True)
    print(f"  Output:   {output_path}", flush=True)

    print(f"\n  Per-element results:", flush=True)
    for sym, data in results["elements"].items():
        line = f"    {sym:3s}: a_lat={data['a_lat']:.4f} A, E_coh={data['E_coh']:.4f} eV, B={data['B_GPa']:.1f} GPa"
        if "C11" in data:
            line += f", C11={data['C11']:.1f}, C12={data['C12']:.1f} GPa"
        print(line, flush=True)

    if results["binary_pairs"]:
        print(f"\n  Binary pair results:", flush=True)
        for pair, data in results["binary_pairs"].items():
            print(f"    {pair}: E_form={data['E_form']:.4f} eV/atom", flush=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVE] Final results -> {output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFT Reference Generator")
    parser.add_argument("elements", nargs="+", help="Element symbols (e.g. Al Cu Fe)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--work-dir", default=None, help="Scratch directory")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel pw.x processes (default: 4)")
    args = parser.parse_args()

    generate_dft_reference(args.elements, args.output, args.work_dir, n_workers=args.parallel)
