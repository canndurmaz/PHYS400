#!/usr/bin/env python3
"""DFT reference data generator using Quantum Espresso + ASE.

For each selected element:
  - EOS fit → equilibrium lattice constant, cohesive energy
  - Stress-strain → C11, C12 (cubic elements)

For binary pairs:
  - Formation energy in L1₂ (FCC) or B2 (BCC) structure

Output: dft_results.json
"""

import json
import math
import os
import shutil
import sys
import time

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
# L1_2 (Cu3Au-type): natural for fcc-fcc pairs
# B2  (CsCl-type):   natural for bcc-bcc pairs
# B2  also used for fcc-bcc (most common intermetallic, e.g. FeAl, CuZn)
# L1_2 for fcc-hcp   (e.g. Al3Mg, Al3Ti — hcp elements often form L1_2 with fcc)
# B2  for bcc-hcp    (e.g. TiFe — B2 is the stable phase)
# L1_2 for fcc-diamond (e.g. Al3Si — diamond elements behave fcc-like in alloys)
# B2  for bcc-diamond  (e.g. FeSi — B2 is the stable phase)
# B2  for hcp-hcp      (e.g. MgZn — B2 or Laves, B2 is simpler)
# L1_2 for hcp-diamond (fallback)
# B2  for diamond-diamond (fallback)
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
    print(f"    [QE] Creating calculator in {directory}", flush=True)
    print(f"    [QE]   pseudopotentials: {pseudopotentials}", flush=True)
    print(f"    [QE]   magnetic (nspin=2): {magnetic}", flush=True)
    print(f"    [QE]   ecutwfc=40 Ry, ecutrho=320 Ry, kpts=(6,6,6)", flush=True)

    pseudo_file = list(pseudopotentials.values())[0]
    pseudo_path = os.path.join(PSEUDO_DIR, pseudo_file)
    if os.path.isfile(pseudo_path):
        print(f"    [QE]   pseudo file found: {pseudo_path}", flush=True)
    else:
        print(f"    [QE]   WARNING: pseudo file NOT found: {pseudo_path}", flush=True)

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


def _eos_fit(symbol, work_dir):
    """Equation of state fit → (a0, E_coh, B) for one element."""
    data = ELEMENT_DATA[symbol]
    lat, a_guess, pseudo = data
    magnetic = symbol in MAGNETIC_ELEMENTS

    print(f"  [EOS] Starting EOS fit for {symbol}", flush=True)
    print(f"  [EOS]   lattice type: {lat}", flush=True)
    print(f"  [EOS]   initial guess a = {a_guess} A", flush=True)
    print(f"  [EOS]   pseudopotential: {pseudo}", flush=True)
    print(f"  [EOS]   magnetic: {magnetic}", flush=True)
    print(f"  [EOS]   scratch dir: {work_dir}", flush=True)

    # For HCP, ASE bulk() uses a and c/a; use FCC proxy
    if lat in ("hcp", "hex"):
        ase_lat = "fcc"
        a_guess_ase = a_guess * math.sqrt(2)
        print(f"  [EOS]   HCP element -> using FCC proxy, a_ase = {a_guess_ase:.4f} A", flush=True)
    elif lat == "diamond":
        ase_lat = "diamond"
        a_guess_ase = a_guess
        print(f"  [EOS]   diamond lattice, a_ase = {a_guess_ase:.4f} A", flush=True)
    else:
        ase_lat = lat
        a_guess_ase = a_guess

    volumes = []
    energies = []
    strains = np.linspace(0.96, 1.04, 7)
    print(f"  [EOS]   Running 7 SCF calculations (strain 0.96 to 1.04)...", flush=True)

    t_eos_start = time.time()
    for i, s in enumerate(strains):
        a = a_guess_ase * s
        atoms = bulk(symbol, ase_lat, a=a)
        calc_dir = os.path.join(work_dir, f"eos_{i}")
        print(f"  [EOS]   Point {i+1}/7: strain={s:.3f}, a={a:.4f} A, dir={calc_dir}", flush=True)
        atoms.calc = _make_calculator(
            {symbol: pseudo}, calc_dir, magnetic=magnetic
        )
        try:
            t0 = time.time()
            print(f"    [SCF] Launching pw.x ...", flush=True)
            e = atoms.get_potential_energy()
            v = atoms.get_volume()
            dt = time.time() - t0
            volumes.append(v)
            energies.append(e)
            print(f"    [SCF] DONE in {dt:.1f}s  E={e:.4f} eV  V={v:.3f} A^3", flush=True)
        except Exception as exc:
            print(f"    [SCF] FAILED after {time.time()-t0:.1f}s: {exc}", flush=True)

    t_eos = time.time() - t_eos_start
    print(f"  [EOS]   All 7 points finished in {t_eos:.1f}s  ({len(volumes)} succeeded)", flush=True)

    if len(volumes) < 5:
        print(f"  [EOS]   WARNING: Only {len(volumes)} valid EOS points, need 5. Using fallback values.", flush=True)
        return a_guess, 3.0, 70.0

    print(f"  [EOS]   Fitting Birch-Murnaghan EOS...", flush=True)
    eos = EquationOfState(volumes, energies, "birchmurnaghan")
    v0, e0, B = eos.fit()
    print(f"  [EOS]   EOS fit: V0={v0:.3f} A^3, E0={e0:.4f} eV, B_raw={B:.6f} eV/A^3", flush=True)

    # Convert v0 back to lattice constant
    n_atoms = 4 if ase_lat == "fcc" else (2 if ase_lat == "bcc" else 8 if ase_lat == "diamond" else 4)
    a0 = (v0 / (n_atoms / 4.0)) ** (1.0 / 3.0) if ase_lat == "fcc" else (v0 * 2) ** (1.0 / 3.0)
    print(f"  [EOS]   Lattice constant from V0: a0={a0:.4f} A (n_atoms={n_atoms})", flush=True)

    # For HCP, convert back
    if lat in ("hcp", "hex"):
        a0_before = a0
        a0 = a0 / math.sqrt(2)
        print(f"  [EOS]   HCP conversion: a0 {a0_before:.4f} -> {a0:.4f} A", flush=True)

    # Cohesive energy per atom
    E_coh = -e0 / n_atoms
    print(f"  [EOS]   Cohesive energy: E_coh = -E0/{n_atoms} = {E_coh:.4f} eV/atom", flush=True)

    # Bulk modulus: eV/A^3 → GPa
    B_GPa = B / 0.00624
    print(f"  [EOS]   Bulk modulus: B = {B_GPa:.1f} GPa", flush=True)

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
    print(f"  [ELASTIC] Step 1/2: Baseline stress (unstrained)...", flush=True)
    ase_lat = lat
    atoms = bulk(symbol, ase_lat, a=a0)
    calc_dir = os.path.join(work_dir, "elastic_base")
    atoms.calc = _make_calculator({symbol: pseudo}, calc_dir, magnetic=magnetic)

    try:
        t0 = time.time()
        stress0 = atoms.get_stress()  # Voigt: xx, yy, zz, yz, xz, xy (eV/A^3)
        dt = time.time() - t0
        print(f"  [ELASTIC]   Baseline stress computed in {dt:.1f}s", flush=True)
        print(f"  [ELASTIC]   sigma0 = [{', '.join(f'{s:.6f}' for s in stress0)}] eV/A^3", flush=True)
    except Exception as exc:
        print(f"  [ELASTIC]   Baseline FAILED: {exc}", flush=True)
        return None, None

    # Apply small strain in x
    delta = 0.01
    print(f"  [ELASTIC] Step 2/2: Strained stress (delta={delta})...", flush=True)
    cell = atoms.get_cell().copy()
    cell[0, 0] *= (1 + delta)
    atoms_strained = atoms.copy()
    atoms_strained.set_cell(cell, scale_atoms=True)
    calc_dir_s = os.path.join(work_dir, "elastic_strain")
    atoms_strained.calc = _make_calculator(
        {symbol: pseudo}, calc_dir_s, magnetic=magnetic
    )

    try:
        t0 = time.time()
        stress1 = atoms_strained.get_stress()
        dt = time.time() - t0
        print(f"  [ELASTIC]   Strained stress computed in {dt:.1f}s", flush=True)
        print(f"  [ELASTIC]   sigma1 = [{', '.join(f'{s:.6f}' for s in stress1)}] eV/A^3", flush=True)
    except Exception as exc:
        print(f"  [ELASTIC]   Strained FAILED: {exc}", flush=True)
        return None, None

    # C11 = d(sigma_xx)/d(epsilon_xx), C12 = d(sigma_yy)/d(epsilon_xx)
    to_GPa = 1.0 / 0.00624
    C11 = (stress1[0] - stress0[0]) / delta * to_GPa
    C12 = (stress1[1] - stress0[1]) / delta * to_GPa

    print(f"  [ELASTIC]   d_sigma_xx = {stress1[0]-stress0[0]:.6f}, d_sigma_yy = {stress1[1]-stress0[1]:.6f} eV/A^3", flush=True)
    print(f"  [ELASTIC] === Result: C11={abs(C11):.1f} GPa, C12={abs(C12):.1f} GPa ===", flush=True)

    return abs(C11), abs(C12)


def _binary_formation_energy(sym_i, sym_j, work_dir, e_per_atom):
    """Formation energy of a binary compound (L1₂ or B2)."""
    data_i = ELEMENT_DATA.get(sym_i)
    data_j = ELEMENT_DATA.get(sym_j)
    if not data_i or not data_j:
        print(f"  [BINARY] Skipping {sym_i}-{sym_j}: missing element data", flush=True)
        return None

    lat_i, a_i, pseudo_i = data_i
    lat_j, a_j, pseudo_j = data_j

    # Choose reference structure (normalize hex→hcp, dia→diamond for lookup)
    def _norm_lat(l):
        l = l.lower()
        if l in ("hex",):
            return "hcp"
        if l in ("dia",):
            return "diamond"
        return l
    key = tuple(sorted([_norm_lat(lat_i), _norm_lat(lat_j)]))
    ref_lat = _BINARY_LATTICE.get(key, "l12")  # default L1_2 for any unknown combo

    # Average lattice constant
    a_mix = (a_i + a_j) / 2.0
    magnetic = sym_i in MAGNETIC_ELEMENTS or sym_j in MAGNETIC_ELEMENTS

    print(f"  [BINARY] Computing formation energy: {sym_i}({lat_i})-{sym_j}({lat_j})", flush=True)
    print(f"  [BINARY]   reference structure: {ref_lat}", flush=True)
    print(f"  [BINARY]   a_mix = ({a_i} + {a_j})/2 = {a_mix:.4f} A", flush=True)
    print(f"  [BINARY]   magnetic: {magnetic}", flush=True)
    print(f"  [BINARY]   pure E/atom references: {sym_i}={e_per_atom.get(sym_i, 'N/A')}, {sym_j}={e_per_atom.get(sym_j, 'N/A')}", flush=True)

    from ase import Atoms

    if ref_lat == "l12":
        a = a_mix * 1.0
        positions = [
            [0, 0, 0], [0.5 * a, 0.5 * a, 0],
            [0.5 * a, 0, 0.5 * a], [0, 0.5 * a, 0.5 * a]
        ]
        symbols = [sym_i, sym_i, sym_i, sym_j]
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        print(f"  [BINARY]   L1_2 structure: {sym_i}3{sym_j}, a={a:.4f} A, 4 atoms", flush=True)
    elif ref_lat == "b2":
        a = a_mix
        positions = [[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]]
        symbols = [sym_i, sym_j]
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        print(f"  [BINARY]   B2 structure: {sym_i}{sym_j}, a={a:.4f} A, 2 atoms", flush=True)
    else:
        a = a_mix
        positions = [[0, 0, 0], [0.5 * a, 0.5 * a, 0.5 * a]]
        symbols = [sym_i, sym_j]
        cell = [[a, 0, 0], [0, a, 0], [0, 0, a]]
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        print(f"  [BINARY]   B1 structure: {sym_i}{sym_j}, a={a:.4f} A, 2 atoms", flush=True)

    pseudopotentials = {sym_i: pseudo_i, sym_j: pseudo_j}
    calc_dir = os.path.join(work_dir, f"binary_{sym_i}_{sym_j}")
    print(f"  [BINARY]   scratch dir: {calc_dir}", flush=True)
    atoms.calc = _make_calculator(pseudopotentials, calc_dir, magnetic=magnetic)

    try:
        t0 = time.time()
        print(f"  [BINARY]   Launching pw.x for binary SCF...", flush=True)
        e_total = atoms.get_potential_energy()
        dt = time.time() - t0
        n_atoms = len(atoms)
        e_mix = e_total / n_atoms

        n_i = symbols.count(sym_i)
        n_j = symbols.count(sym_j)
        e_ref = (n_i * e_per_atom.get(sym_i, 0) + n_j * e_per_atom.get(sym_j, 0)) / n_atoms
        e_form = e_mix - e_ref
        print(f"  [BINARY]   SCF DONE in {dt:.1f}s", flush=True)
        print(f"  [BINARY]   E_total = {e_total:.4f} eV ({n_atoms} atoms)", flush=True)
        print(f"  [BINARY]   E_mix/atom = {e_mix:.4f} eV", flush=True)
        print(f"  [BINARY]   E_ref/atom = {e_ref:.4f} eV (weighted pure avg)", flush=True)
        print(f"  [BINARY] === Result: E_form = {e_form:.4f} eV/atom ===", flush=True)
        return e_form
    except Exception as exc:
        print(f"  [BINARY]   SCF FAILED after {time.time()-t0:.1f}s: {exc}", flush=True)
        return None


def generate_dft_reference(elements, output_path=None, work_dir=None):
    """Run DFT calculations for selected elements and generate reference data.

    Args:
        elements: list of element symbols
        output_path: path for output JSON (default: dft_results.json next to this file)
        work_dir: scratch directory for QE calculations

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
    print(f"  Requested elements: {elements}", flush=True)
    print(f"  Output file:        {output_path}", flush=True)
    print(f"  Scratch directory:  {work_dir}", flush=True)
    print(f"  QE binary:          {QE_BIN}", flush=True)
    print(f"  Pseudo directory:   {PSEUDO_DIR}", flush=True)

    # Check prerequisites
    if os.path.isfile(QE_BIN):
        print(f"  pw.x found:        YES", flush=True)
    else:
        print(f"  pw.x found:        NO  <-- DFT will fail!", flush=True)

    # Filter to elements we have data for
    valid = [e for e in elements if e in ELEMENT_DATA]
    skipped = [e for e in elements if e not in ELEMENT_DATA]
    if skipped:
        print(f"  Skipping (no DFT metadata): {skipped}", flush=True)
    print(f"  Valid elements:     {valid}", flush=True)

    # Count total work
    n_eos = len(valid) * 7       # 7 strain points per element
    n_elastic = sum(1 for e in valid if ELEMENT_DATA[e][0] in ("fcc", "bcc")) * 2  # 2 SCF per cubic
    n_binary = len(valid) * (len(valid) - 1) // 2
    n_total_scf = n_eos + n_elastic + n_binary
    print(f"\n  Planned DFT calculations:", flush=True)
    print(f"    EOS fits:         {len(valid)} elements x 7 strains = {n_eos} SCF runs", flush=True)
    print(f"    Elastic (cubic):  {n_elastic} SCF runs", flush=True)
    print(f"    Binary pairs:     {n_binary} SCF runs", flush=True)
    print(f"    TOTAL:            ~{n_total_scf} pw.x invocations", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {"elements": {}, "binary_pairs": {}}
    e_per_atom = {}
    t_total_start = time.time()

    # ── Stage 1: Single-element EOS + elastic constants ──────────────────
    for idx, sym in enumerate(valid, 1):
        print(f"\n{'─'*60}", flush=True)
        print(f"  ELEMENT {idx}/{len(valid)}: {sym}", flush=True)
        print(f"{'─'*60}", flush=True)
        elem_dir = os.path.join(work_dir, sym)
        os.makedirs(elem_dir, exist_ok=True)

        # EOS fit
        t_elem_start = time.time()
        a0, E_coh, B = _eos_fit(sym, elem_dir)
        t_eos_done = time.time()
        print(f"\n  Summary for {sym} EOS ({t_eos_done - t_elem_start:.1f}s):", flush=True)
        print(f"    a0    = {a0:.4f} A", flush=True)
        print(f"    E_coh = {E_coh:.4f} eV", flush=True)
        print(f"    B     = {B:.1f} GPa", flush=True)

        entry = {
            "a_lat": round(a0, 4),
            "E_coh": round(E_coh, 4),
            "B_GPa": round(B, 1),
            "lattice": ELEMENT_DATA[sym][0],
        }

        # Elastic constants for cubic elements
        C11, C12 = _elastic_constants(sym, a0, elem_dir)
        if C11 is not None:
            entry["C11"] = round(C11, 1)
            entry["C12"] = round(C12, 1)
            print(f"    C11   = {C11:.1f} GPa", flush=True)
            print(f"    C12   = {C12:.1f} GPa", flush=True)

        results["elements"][sym] = entry
        t_elem_done = time.time()
        print(f"  Element {sym} total time: {t_elem_done - t_elem_start:.1f}s", flush=True)

        # Save intermediate results after each element
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [SAVE] Intermediate results saved to {output_path}", flush=True)

        # Store per-atom energy for formation energy calculation
        data = ELEMENT_DATA[sym]
        lat = data[0]
        n_atoms = 4 if lat == "fcc" else (2 if lat == "bcc" else 8 if lat == "diamond" else 2)
        e_per_atom[sym] = -E_coh
        print(f"  [REF] Stored pure E/atom for {sym}: {e_per_atom[sym]:.4f} eV", flush=True)

    # ── Stage 2: Binary pair formation energies ──────────────────────────
    pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i+1, len(valid))]
    if pairs:
        print(f"\n{'─'*60}", flush=True)
        print(f"  BINARY PAIRS: {len(pairs)} total", flush=True)
        print(f"{'─'*60}", flush=True)

    for idx, (sym_i, sym_j) in enumerate(pairs, 1):
        print(f"\n  PAIR {idx}/{len(pairs)}: {sym_i}-{sym_j}", flush=True)
        t_pair_start = time.time()
        e_form = _binary_formation_energy(sym_i, sym_j, work_dir, e_per_atom)
        t_pair_done = time.time()
        if e_form is not None:
            pair_key = f"{sym_i}-{sym_j}"
            results["binary_pairs"][pair_key] = {
                "E_form": round(e_form, 4),
            }
            print(f"  Pair {sym_i}-{sym_j} done in {t_pair_done - t_pair_start:.1f}s: E_form={e_form:.4f} eV/atom", flush=True)
        else:
            print(f"  Pair {sym_i}-{sym_j} done in {t_pair_done - t_pair_start:.1f}s: FAILED", flush=True)

        # Save intermediate results after each pair
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [SAVE] Intermediate results saved to {output_path}", flush=True)

    # ── Final summary ────────────────────────────────────────────────────
    t_total = time.time() - t_total_start
    print(f"\n{'='*60}", flush=True)
    print(f"DFT REFERENCE COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total time:    {t_total:.1f}s ({t_total/60:.1f} min)", flush=True)
    print(f"  Elements done: {list(results['elements'].keys())}", flush=True)
    print(f"  Pairs done:    {list(results['binary_pairs'].keys())}", flush=True)
    print(f"  Output file:   {output_path}", flush=True)
    print(f"", flush=True)

    print(f"  Per-element results:", flush=True)
    for sym, data in results["elements"].items():
        line = f"    {sym:3s}: a_lat={data['a_lat']:.4f} A, E_coh={data['E_coh']:.4f} eV, B={data['B_GPa']:.1f} GPa"
        if "C11" in data:
            line += f", C11={data['C11']:.1f}, C12={data['C12']:.1f} GPa"
        print(line, flush=True)

    if results["binary_pairs"]:
        print(f"\n  Binary pair results:", flush=True)
        for pair, data in results["binary_pairs"].items():
            print(f"    {pair}: E_form={data['E_form']:.4f} eV/atom", flush=True)

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  [SAVE] Final results saved to {output_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFT Reference Generator")
    parser.add_argument("elements", nargs="+", help="Element symbols (e.g. Al Cu Fe)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--work-dir", default=None, help="Scratch directory")
    args = parser.parse_args()

    generate_dft_reference(args.elements, args.output, args.work_dir)
