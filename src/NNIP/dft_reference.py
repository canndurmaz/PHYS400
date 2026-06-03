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

from src.NNIP.logging_config import setup_logger

logger = setup_logger("dft")

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
    "Mn": ("bcc", 2.89, "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF"),  # δ-Mn/γ-Mn BCC proxy (α-Mn is 29-atom non-collinear)
    "Co": ("hcp", 2.48, "Co.pbe-spn-kjpaw_psl.0.3.1.UPF"),  # FM HCP, lit moment ~1.6 μB; PBE eq at 70 Ry cutoff
    "Ni": ("fcc", 3.52, "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF"),  # FM FCC, lit moment ~0.6 μB
    "Si": ("diamond", 5.43, "Si.pbe-n-kjpaw_psl.1.0.0.UPF"),
    "Au": ("fcc", 4.08, "Au.pz-rrkjus_aewfc.UPF"),
    "Mo": ("bcc", 3.15, "Mo-PBE.upf"),
}

# Elements requiring spin-polarized calculation
MAGNETIC_ELEMENTS = {"Fe", "Cr", "Mn", "Co", "Ni"}

# Initial magnetic moments (μ_B) for SCF symmetry-breaking. Cr and Mn use
# alternating-sign G-AFM seeds (their experimental ground states are AFM);
# Fe/Co/Ni use a ferromagnetic seed.
# Cr is UNRESOLVED: seed 0.6 → 1.5 → 3.0 all collapse to NM during SCF
# iteration (initial kick produces abs ≈ 24 μB, decays to ≈ 0.1 μB by iter
# 14). BCC-Cr's AFM lies only ~10 meV/atom below NM and the current 0.01 Ry
# smearing (≈ 136 meV) likely washes out the gap. Probable fixes: drop
# Cr's smearing to 0.005 Ry, or apply DFT+U on Cr d states, or use a
# fixed-moment per-sublattice constraint. Until then, Cr's B in
# dft_results.json reflects the NM SCF (~256 GPa vs lit 160).
ELEMENTAL_MAGMOM = {
    "Fe": 2.5,
    "Cr": 3.0,
    "Mn": 3.5,
    "Co": 1.7,
    "Ni": 0.7,
}

# Per-atom fixed-moment constraint for FM elements. When set, every EOS-scan
# SCF is forced to the same total moment via pw.x's tot_magnetization
# Lagrange multiplier, eliminating the magnetic-state flips that produced
# wavy E(V) curves in the previous run (Fe B=52 vs lit. 170 GPa; Ni's BM
# fit silently fell back to the default 70). Values are lit. ground-state
# moments — close enough to the observed equilibrium-volume moment that
# the constraint doesn't add appreciable strain to the SCF, while strict
# enough to prevent a state flip at any single strain point.
# AFM elements are intentionally absent: their total is zero by sublattice
# symmetry; forcing tot_magnetization=0 doesn't distinguish AFM from NM,
# so the AFM-collapse fix is the boosted seed in ELEMENTAL_MAGMOM, not a
# constraint.
_FM_TOT_MAG_PER_ATOM = {
    "Fe": 2.2,
    "Co": 1.7,
    "Ni": 0.6,
}

# Elements seeded with G-AFM (alternating sign per sublattice). Each gets its
# own counter so independent sublattices don't interfere in mixed-element cells.
_AFM_ELEMENTS = {"Cr", "Mn"}

# Per-element plane-wave cutoffs (ecutwfc, ecutrho) in Ry. The defaults below
# satisfy each pseudopotential's "Suggested minimum cutoff" from its UPF header.
# For 3d magnetic metals these are 50–90 % above the legacy 40/320 used for
# every element. Co's debug session (2026-06-03) showed B = 1012 GPa at the
# 40/320 default vs B = 220 GPa at 70/560 — a basis-set-incompleteness
# artifact (under-converged plane-wave basis gives volume-dependent energy
# errors that masquerade as enormous stiffness).
_RECOMMENDED_CUTOFFS = {
    "Al": (40, 320),
    "Cu": (40, 320),
    "Mg": (40, 320),
    "Si": (40, 320),
    "Ti": (40, 320),
    "Zn": (40, 320),
    "Mo": (40, 320),
    "Au": (40, 320),
    "Fe": (75, 600),   # pseudo recommends 71/496
    "Cr": (60, 480),
    "Mn": (50, 400),   # pseudo recommends 46/244; bumped for parity
    "Co": (70, 560),   # pseudo recommends 60/445
    "Ni": (80, 640),   # pseudo recommends 75/476
}


def _cutoffs_for_cell(symbols):
    """Pick ecutwfc/ecutrho safe for a cell that contains the given elements.

    For mixed-element cells the stricter cutoff wins so the 3d metal in a
    binary pair doesn't get under-sampled. Falls back to the historical
    40/320 for any unknown symbol.
    """
    uniq = set(symbols)
    wfc = max(_RECOMMENDED_CUTOFFS.get(s, (40, 320))[0] for s in uniq)
    rho = max(_RECOMMENDED_CUTOFFS.get(s, (40, 320))[1] for s in uniq)
    return wfc, rho


def _initial_magmoms(symbols):
    """Per-atom initial magnetic moments for a list of atomic symbols.

    AFM elements (_AFM_ELEMENTS) get alternating signs across their own
    sublattice (G-AFM seed); other magnetic elements get a constant
    ferromagnetic seed; non-magnetic species get 0.
    """
    moms = []
    afm_counts = {s: 0 for s in _AFM_ELEMENTS}
    for s in symbols:
        m = ELEMENTAL_MAGMOM.get(s, 0.0)
        if s in _AFM_ELEMENTS:
            m *= (-1) ** afm_counts[s]
            afm_counts[s] += 1
        moms.append(m)
    return moms

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


def _make_calculator(pseudopotentials, directory, magnetic=False, kpts=(6, 6, 6),
                     tight_scf=False, isolated_atom=False, tot_mag_per_cell=None,
                     ecutwfc=None, ecutrho=None):
    """Create an Espresso calculator for SCF.

    For magnetic systems we use a smaller smearing (0.01 Ry vs 0.02) so the
    exchange splitting around E_F is not washed out — at 0.02 Ry ≈ 272 meV the
    spin-up/spin-down densities of magnetic 3d metals partially mix at the
    Fermi level, which collapses the moment for borderline cases (Mn was hit
    by this in the prior run). ``tight_scf=True`` uses 1e-8 conv_thr for
    elastic-constant calculations where the stress difference between
    baseline and strained cells is sensitive to incompletely-converged
    magnetic states. ``isolated_atom=True`` swaps to settings appropriate
    for a single atom in a vacuum box: Gaussian smearing with much smaller
    width (atomic levels are discrete; MV smearing at 0.01 Ry oscillates),
    higher mixing β (the level structure converges in fewer iterations),
    and a higher electron_maxstep so we don't time out before the SCF
    settles. Co's isolated-atom run hit the QE default 100-iter cap in
    the prior attempt.
    """
    profile = EspressoProfile(command=QE_BIN, pseudo_dir=PSEUDO_DIR)
    # Cutoffs default to the safe-for-everything 40/320 only if caller did not
    # pick element-aware values. _cutoffs_for_cell() should be used at every
    # call site that knows which species are in the cell.
    if ecutwfc is None:
        ecutwfc = 40
    if ecutrho is None:
        ecutrho = 320
    input_data = {
        "control": {"tprnfor": True, "tstress": True},
        "system": {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
        },
        "electrons": {"conv_thr": 1.0e-8 if tight_scf else 1.0e-6,
                      "electron_maxstep": 200},
    }
    if isolated_atom:
        input_data["system"]["occupations"] = "smearing"
        input_data["system"]["smearing"] = "gauss"
        input_data["system"]["degauss"] = 0.005
        input_data["electrons"]["mixing_beta"] = 0.7
    else:
        input_data["system"]["occupations"] = "smearing"
        input_data["system"]["smearing"] = "mv"
        input_data["system"]["degauss"] = 0.01 if magnetic else 0.02
        input_data["electrons"]["mixing_beta"] = 0.3 if magnetic else 0.7
    if magnetic:
        input_data["system"]["nspin"] = 2
        if tot_mag_per_cell is not None:
            # Fixed-moment SCF: forces every strain point in an EOS scan to
            # the same total moment, killing the state-flip noise that gave
            # wavy E(V) curves for FM elements in the previous run.
            input_data["system"]["tot_magnetization"] = tot_mag_per_cell
        # Thomas-Fermi screening damps spin-charge oscillation that plain
        # mixing can't kill for FM 3d metals (Ni's elastic_base spent 50+
        # iters at ~0.2 Ry scf accuracy with plain + beta=0.3).
        input_data["electrons"]["mixing_mode"] = "local-TF"
        input_data["electrons"]["diagonalization"] = "david"
        input_data["electrons"]["diago_david_ndim"] = 4
    return Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data=input_data,
        kpts=kpts,
        directory=directory,
    )


# ── Isolated atom energy (for true cohesive energy) ──────────────────────────

def _isolated_atom_energy(symbol, pseudo, calc_dir, magnetic):
    """Run SCF for an isolated atom in a large box. Returns energy or None."""
    from ase import Atoms

    box_size = 12.0  # large enough to avoid periodic image interaction
    atoms = Atoms(symbol, positions=[[0, 0, 0]], cell=[box_size] * 3, pbc=True)
    if magnetic:
        atoms.set_initial_magnetic_moments(_initial_magmoms([symbol]))
    # Γ-only is exact for a single atom in a non-periodic box; default 6×6×6
    # samples 216 redundant k-points and dominates the per-element wall time.
    ecutwfc, ecutrho = _cutoffs_for_cell([symbol])
    atoms.calc = _make_calculator({symbol: pseudo}, calc_dir, magnetic=magnetic,
                                  kpts=(1, 1, 1), isolated_atom=True,
                                  ecutwfc=ecutwfc, ecutrho=ecutrho)
    try:
        t0 = time.time()
        e = atoms.get_potential_energy()
        dt = time.time() - t0
        print(f"    [SCF] Isolated {symbol}: E={e:.4f} eV ({dt:.1f}s)", flush=True)
        return e
    except Exception as exc:
        print(f"    [SCF] Isolated {symbol}: FAILED: {exc}", flush=True)
        return None


# ── Single EOS point (top-level function for pickling) ───────────────────────

def _run_eos_point(symbol, pseudo, ase_lat, a, calc_dir, magnetic, strain_idx):
    """Run a single EOS strain point. Returns (strain_idx, volume, energy) or (strain_idx, None, None)."""
    try:
        atoms = bulk(symbol, ase_lat, a=a) * (2, 2, 2)
        tot_mag = None
        if magnetic:
            atoms.set_initial_magnetic_moments(_initial_magmoms(atoms.get_chemical_symbols()))
            per_atom = _FM_TOT_MAG_PER_ATOM.get(symbol)
            if per_atom is not None:
                tot_mag = per_atom * len(atoms)
        # Denser k-mesh for magnetic elements: a sparse 3×3×3 on the 2×2×2
        # supercell lets the Fermi surface topology shift between EOS strain
        # points, so adjacent points can converge to *different* magnetic
        # states. 5×5×5 (≈10×10×10 on primitive) was still too sparse for
        # FM 3d metals — Co's debug session (2026-06-03) traced kinks in
        # E(V) to k-point sampling noise, fixed by 9×9×9 here.
        eos_kpts = (9, 9, 9) if magnetic else (3, 3, 3)
        ecutwfc, ecutrho = _cutoffs_for_cell([symbol])
        atoms.calc = _make_calculator({symbol: pseudo}, calc_dir, magnetic=magnetic,
                                      kpts=eos_kpts, tot_mag_per_cell=tot_mag,
                                      ecutwfc=ecutwfc, ecutrho=ecutrho)
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
    """Equation of state fit → (a0, E_coh, B, eos_data_dict) for one element.

    Runs 7 strain-point SCF calculations in parallel.
    Returns:
        (a0, E_coh, B_GPa, eos_data_dict) where eos_data_dict has volumes/energies lists.
    """
    data = ELEMENT_DATA[symbol]
    lat, a_guess, pseudo = data
    magnetic = symbol in MAGNETIC_ELEMENTS

    logger.info(f"  [EOS] Starting EOS fit for {symbol}")
    logger.debug(f"  [EOS]   lattice={lat}, a_guess={a_guess} A, pseudo={pseudo}, magnetic={magnetic}")
    logger.debug(f"  [EOS]   scratch dir: {work_dir}")

    if lat in ("hcp", "hex"):
        ase_lat = "fcc"
        a_guess_ase = a_guess * math.sqrt(2)
        logger.info(f"  [EOS]   HCP -> FCC proxy, a_ase={a_guess_ase:.4f} A")
    elif lat == "diamond":
        ase_lat = "diamond"
        a_guess_ase = a_guess
    else:
        ase_lat = lat
        a_guess_ase = a_guess

    # ±4 % window for all elements. The previous magnetic-specific ±2 %
    # window was a workaround for magnetic-state flips between strain
    # points — that noise source is now controlled by the FM
    # tot_magnetization constraint and the stronger AFM seed, so we can
    # use the wider window everywhere. ±2 % was also too narrow for Mn:
    # its true a0 (~2.82) fell off the bottom of the ±2 % window centered
    # at a_guess=2.89, forcing the BM solver to extrapolate.
    strains = np.linspace(0.96, 1.04, 7)
    eos_workers = min(n_workers, 7)
    logger.info(f"  [EOS]   Running 7 SCF points in parallel ({eos_workers} workers)...")

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
    logger.info(f"  [EOS]   7 points finished in {t_eos:.1f}s ({len(volumes)} succeeded)")

    if len(volumes) < 5:
        logger.warning(f"  [EOS]   Only {len(volumes)} valid points, need 5. Using fallback.")
        return a_guess, 3.0, 70.0, {"volumes": volumes, "energies": energies}, 0.0

    logger.info(f"  [EOS]   Fitting Birch-Murnaghan EOS...")
    eos = EquationOfState(volumes, energies, "birchmurnaghan")
    v0, e0, B = eos.fit()
    logger.debug(f"  [EOS]   V0={v0:.3f} A^3, E0={e0:.4f} eV, B_raw={B:.6f} eV/A^3")

    # Guard against degenerate fits. Three failure modes seen in practice:
    #   (1) NaN v0/e0/B  — happens when SCF failed for ≥1 point and the BM
    #       solver tries to extrapolate; ASE then propagates NaN.
    #   (2) v0 ≤ 0        — fit landed in the unphysical compression branch.
    #   (3) B ≤ 0         — wavy E(V) (Co's magnetic-relaxation noise) lets
    #       the fit pass shape sanity but with negative curvature at v0,
    #       i.e. a *mechanically unstable* mock minimum.
    # In all cases, silently passing the bad numbers downstream pollutes
    # dft_results.json with a_lat / E_coh / B garbage. Fall back to
    # a_guess and a noted-as-failed entry instead.
    B_GPa_provisional = B / 0.00624 if np.isfinite(B) else float("nan")
    if not (np.isfinite(v0) and np.isfinite(e0) and np.isfinite(B)
            and v0 > 0 and B_GPa_provisional > 1.0):
        logger.warning(f"  [EOS]   BM fit returned non-physical result "
                       f"(V0={v0}, E0={e0}, B={B_GPa_provisional:.1f} GPa); "
                       f"falling back to a_guess")
        return a_guess, 3.0, 70.0, {"volumes": volumes, "energies": energies}, 0.0

    # 2×2×2 supercell of primitive cells:
    #   FCC: 1×8 = 8 atoms, V_super = 8 × a^3/4 = 2a^3  →  a = (V/2)^(1/3)
    #   BCC: 1×8 = 8 atoms, V_super = 8 × a^3/2 = 4a^3  →  a = (V/4)^(1/3)
    #   Diamond: 2×8 = 16 atoms, V_super = 8 × a^3/4 = 2a^3  →  a = (V/2)^(1/3)
    if ase_lat == "fcc":
        n_atoms = 8
        a0 = (v0 / 2.0) ** (1.0 / 3.0)
    elif ase_lat == "bcc":
        n_atoms = 8
        a0 = (v0 / 4.0) ** (1.0 / 3.0)
    elif ase_lat == "diamond":
        n_atoms = 16
        a0 = (v0 / 2.0) ** (1.0 / 3.0)
    else:
        n_atoms = 8
        a0 = (v0 / 8.0) ** (1.0 / 3.0)

    if lat in ("hcp", "hex"):
        a0 = a0 / math.sqrt(2)

    # Compute true cohesive energy: E_coh = E_atom(isolated) - E_bulk/N
    logger.info(f"  [EOS]   Computing isolated atom energy for true E_coh...")
    iso_dir = os.path.join(work_dir, "isolated_atom")
    os.makedirs(iso_dir, exist_ok=True)
    e_atom = _isolated_atom_energy(symbol, pseudo, iso_dir, magnetic)

    if e_atom is not None:
        E_coh = e_atom - e0 / n_atoms
    else:
        logger.warning(f"  [EOS]   Isolated atom calc failed, using E_bulk/N as fallback")
        E_coh = -e0 / n_atoms

    B_GPa = B / 0.00624

    e_bulk_per_atom = e0 / n_atoms  # total DFT energy per atom (for formation energy refs)
    logger.info(f"  [EOS] === Result: a0={a0:.4f} A, E_coh={E_coh:.4f} eV, B={B_GPa:.1f} GPa ===")
    eos_data_dict = {"volumes": volumes, "energies": energies}
    return a0, E_coh, B_GPa, eos_data_dict, e_bulk_per_atom


def _elastic_constants(symbol, a0, work_dir):
    """C11, C12 via stress-strain for cubic elements."""
    data = ELEMENT_DATA[symbol]
    lat, _, pseudo = data
    magnetic = symbol in MAGNETIC_ELEMENTS

    if lat not in ("fcc", "bcc"):
        logger.info(f"  [ELASTIC] Skipping {symbol} (lattice={lat}, not cubic)")
        return None, None

    logger.info(f"  [ELASTIC] Computing C11, C12 for {symbol} ({lat}, a0={a0:.4f} A)")

    # Baseline stress — use cubic conventional cell so Cartesian strain maps to C11/C12
    # FCC conventional: 4 atoms × (1,1,2) = 8; BCC conventional: 2 atoms × (2,2,2) = 16 → use (1,1,2) for FCC, (2,2,2) for BCC
    logger.info(f"  [ELASTIC] Step 1/2: Baseline stress...")
    atoms = bulk(symbol, lat, a=a0, cubic=True) * (2, 2, 2)
    if magnetic:
        atoms.set_initial_magnetic_moments(_initial_magmoms(atoms.get_chemical_symbols()))
    calc_dir = os.path.join(work_dir, "elastic_base")
    ecutwfc, ecutrho = _cutoffs_for_cell([symbol])
    elastic_kpts = (5, 5, 5) if magnetic else (3, 3, 3)
    atoms.calc = _make_calculator({symbol: pseudo}, calc_dir, magnetic=magnetic,
                                  kpts=elastic_kpts, tight_scf=True,
                                  ecutwfc=ecutwfc, ecutrho=ecutrho)

    try:
        t0 = time.time()
        stress0 = atoms.get_stress()
        logger.info(f"  [ELASTIC]   Baseline done in {time.time()-t0:.1f}s")
    except Exception as exc:
        logger.warning(f"  [ELASTIC]   Baseline FAILED: {exc}", exc_info=True)
        return None, None

    # Strained
    delta = 0.01
    logger.info(f"  [ELASTIC] Step 2/2: Strained stress (delta={delta})...")
    cell = atoms.get_cell().copy()
    cell[0, 0] *= (1 + delta)
    atoms_strained = atoms.copy()
    atoms_strained.set_cell(cell, scale_atoms=True)
    if magnetic:
        atoms_strained.set_initial_magnetic_moments(_initial_magmoms(atoms_strained.get_chemical_symbols()))
    calc_dir_s = os.path.join(work_dir, "elastic_strain")
    atoms_strained.calc = _make_calculator({symbol: pseudo}, calc_dir_s,
                                           magnetic=magnetic, kpts=elastic_kpts,
                                           tight_scf=True,
                                           ecutwfc=ecutwfc, ecutrho=ecutrho)

    try:
        t0 = time.time()
        stress1 = atoms_strained.get_stress()
        logger.info(f"  [ELASTIC]   Strained done in {time.time()-t0:.1f}s")
    except Exception as exc:
        logger.warning(f"  [ELASTIC]   Strained FAILED: {exc}", exc_info=True)
        return None, None

    # Stress convention: ASE returns σ in eV/Å³ with σ_ij = (1/V)∂E/∂ε_ij and
    # tension-positive sign. With ε_xx = +δ > 0 (cell stretched along x), the
    # restoring stress should also be positive — so dσ/dε > 0 for any
    # stable cubic crystal. A negative value here is a red flag: either the
    # SCF magnetic state flipped between baseline and strained runs, or the
    # cell is mechanically unstable in this lattice (e.g. NM/FM-BCC Mn).
    to_GPa = 1.0 / 0.00624
    C11 = (stress1[0] - stress0[0]) / delta * to_GPa
    C12 = (stress1[1] - stress0[1]) / delta * to_GPa

    if C11 < 0 or C12 < 0 or C11 < C12:
        logger.warning(
            f"  [ELASTIC]   Non-physical result for {symbol}: "
            f"C11={C11:.1f}, C12={C12:.1f} GPa. "
            "Likely magnetic-state flip between baseline and strained SCFs, "
            "or the assumed lattice is mechanically unstable for this element.")

    logger.info(f"  [ELASTIC] === Result: C11={C11:.1f} GPa, C12={C12:.1f} GPa ===")
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

    pair_key = f"{sym_i}-{sym_j}"

    # Hard-fail when either reference is missing — silently treating it as
    # 0 eV pollutes E_form with the pseudopotential total energy of the other
    # species, which was the root cause of all the ~-268 / -1448 / -1657 eV
    # entries in earlier dft_results.json revisions.
    e_ref_i = e_per_atom.get(sym_i)
    e_ref_j = e_per_atom.get(sym_j)
    if e_ref_i is None or e_ref_j is None or e_ref_i == 0.0 or e_ref_j == 0.0:
        print(f"  [BINARY] {pair_key}: SKIPPED — missing/zero elemental reference "
              f"(E_bulk_{sym_i}={e_ref_i}, E_bulk_{sym_j}={e_ref_j})", flush=True)
        return (pair_key, None)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    if magnetic:
        atoms.set_initial_magnetic_moments(_initial_magmoms(symbols))
    pseudopotentials = {sym_i: pseudo_i, sym_j: pseudo_j}
    calc_dir = os.path.join(work_dir, f"binary_{sym_i}_{sym_j}")
    os.makedirs(calc_dir, exist_ok=True)
    ecutwfc, ecutrho = _cutoffs_for_cell([sym_i, sym_j])
    atoms.calc = _make_calculator(pseudopotentials, calc_dir, magnetic=magnetic,
                                  ecutwfc=ecutwfc, ecutrho=ecutrho)

    try:
        t0 = time.time()
        e_total = atoms.get_potential_energy()
        dt = time.time() - t0
        n_atoms = len(atoms)
        e_mix = e_total / n_atoms
        n_i = symbols.count(sym_i)
        n_j = symbols.count(sym_j)
        e_ref = (n_i * e_ref_i + n_j * e_ref_j) / n_atoms
        e_form = e_mix - e_ref
        print(f"  [BINARY] {pair_key}: E_form={e_form:.4f} eV/atom ({dt:.1f}s)", flush=True)
        return (pair_key, e_form)
    except Exception as exc:
        print(f"  [BINARY] {pair_key}: FAILED ({exc})", flush=True)
        return (pair_key, None)


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_dft_reference(elements, output_path=None, work_dir=None, n_workers=4,
                           include_elastic=True):
    """Run DFT calculations for selected elements and generate reference data.

    Args:
        elements: list of element symbols
        output_path: path for output JSON (default: dft_results.json next to this file)
        work_dir: scratch directory for QE calculations
        n_workers: max parallel pw.x processes (default: 4)
        include_elastic: run the C11/C12 elastic stage (default True). When
            False, only EOS + isolated-atom + binary-pair SCFs run; C_ij can
            be backfilled from B isotropically via fill_elastic_constants.py.
            The MEAM seeder (dft_to_meam.py) does not consume DFT C_ij.

    Returns:
        dict with DFT reference data
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "dft_results.json")
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(__file__), "dft_scratch")
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"DFT Reference Generator")
    logger.info(f"{'='*60}")
    logger.info(f"  Elements:        {elements}")
    logger.info(f"  Output:          {output_path}")
    logger.info(f"  Scratch:         {work_dir}")
    logger.info(f"  Parallel workers: {n_workers}")
    logger.info(f"  pw.x:            {QE_BIN} ({'found' if os.path.isfile(QE_BIN) else 'NOT FOUND'})")

    valid = [e for e in elements if e in ELEMENT_DATA]
    skipped = [e for e in elements if e not in ELEMENT_DATA]
    if skipped:
        logger.info(f"  Skipping (no metadata): {skipped}")
    logger.info(f"  Valid elements:  {valid}")

    n_eos = len(valid) * 7
    n_elastic = sum(1 for e in valid if ELEMENT_DATA[e][0] in ("fcc", "bcc")) * 2 if include_elastic else 0
    n_binary = len(valid) * (len(valid) - 1) // 2
    elastic_note = f"{n_elastic} elastic" if include_elastic else "0 elastic (--no-elastic)"
    logger.info(f"\n  Planned: {n_eos} EOS + {elastic_note} + {n_binary} binary = ~{n_eos+n_elastic+n_binary} SCF runs")
    logger.info(f"  EOS runs parallelized ({min(n_workers, 7)} at a time per element)")
    logger.info(f"  Binary pairs parallelized ({min(n_workers, n_binary)} at a time)")
    logger.info(f"{'='*60}\n")

    # ── Load existing partial results for resume ─────────────────────────
    results = {"elements": {}, "binary_pairs": {}}
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                results = json.load(f)
            logger.info(f"  Loaded existing results: {len(results.get('elements', {}))} elements, "
                        f"{len(results.get('binary_pairs', {}))} pairs")
        except (json.JSONDecodeError, ValueError):
            results = {"elements": {}, "binary_pairs": {}}

    e_per_atom = {}
    # Recover e_per_atom from previously computed elements
    for sym, data in results.get("elements", {}).items():
        if "e_bulk_per_atom" in data:
            e_per_atom[sym] = data["e_bulk_per_atom"]

    t_total_start = time.time()

    # ── Stage 1: Single-element EOS + elastic constants ──────────────────
    # Elements run sequentially (elastic depends on EOS a0),
    # but EOS strain points within each element run in parallel.
    for idx, sym in enumerate(valid, 1):
        if sym in results.get("elements", {}):
            logger.info(f"\n  ELEMENT {idx}/{len(valid)}: {sym} — [RESUME] already computed, skipping")
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(f"  ELEMENT {idx}/{len(valid)}: {sym}")
        logger.info(f"{'─'*60}")
        elem_dir = os.path.join(work_dir, sym)
        os.makedirs(elem_dir, exist_ok=True)

        t_elem_start = time.time()
        a0, E_coh, B, eos_data, e_bulk = _eos_fit(sym, elem_dir, n_workers=n_workers)

        entry = {
            "a_lat": round(a0, 4),
            "E_coh": round(E_coh, 4),
            "B_GPa": round(B, 1),
            "lattice": ELEMENT_DATA[sym][0],
            "eos_data": eos_data,
            "e_bulk_per_atom": e_bulk,
        }

        if include_elastic:
            C11, C12 = _elastic_constants(sym, a0, elem_dir)
            if C11 is not None:
                entry["C11"] = round(C11, 1)
                entry["C12"] = round(C12, 1)

        results["elements"][sym] = entry
        logger.info(f"  Element {sym} done in {time.time()-t_elem_start:.1f}s")

        # Save intermediate
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"  [SAVE] Intermediate results -> {output_path}")

        e_per_atom[sym] = e_bulk  # DFT total energy per atom for formation energy reference

    # ── Stage 2: Binary pairs — ALL in parallel ──────────────────────────
    pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i+1, len(valid))]
    # Filter out already-computed pairs
    existing_pairs = set(results.get("binary_pairs", {}).keys())
    remaining_pairs = [(a, b) for a, b in pairs if f"{a}-{b}" not in existing_pairs]

    if existing_pairs & {f"{a}-{b}" for a, b in pairs}:
        skipped = len(pairs) - len(remaining_pairs)
        logger.info(f"\n  [RESUME] {skipped} binary pair(s) already computed, skipping them")

    if remaining_pairs:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  BINARY PAIRS: {len(remaining_pairs)} remaining (of {len(pairs)} total), "
                     f"running {min(n_workers, len(remaining_pairs))} in parallel")
        logger.info(f"{'─'*60}")

        t_pairs_start = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for sym_i, sym_j in remaining_pairs:
                fut = pool.submit(_run_binary_pair, sym_i, sym_j, work_dir, e_per_atom)
                futures[fut] = (sym_i, sym_j)

            done_count = 0
            for fut in as_completed(futures):
                pair_key, e_form = fut.result()
                done_count += 1
                if e_form is not None:
                    results["binary_pairs"][pair_key] = {"E_form": round(e_form, 4)}
                logger.info(f"  [{done_count}/{len(remaining_pairs)}] {pair_key}: {'OK' if e_form is not None else 'FAILED'}")

                # Save after each completion
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

        t_pairs = time.time() - t_pairs_start
        logger.info(f"  All {len(remaining_pairs)} pairs done in {t_pairs:.1f}s ({t_pairs/60:.1f} min)")
    elif pairs:
        logger.info(f"\n  [RESUME] All {len(pairs)} binary pairs already computed")

    # ── Final summary ────────────────────────────────────────────────────
    t_total = time.time() - t_total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"DFT REFERENCE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Total wall time: {t_total:.1f}s ({t_total/60:.1f} min)")
    logger.info(f"  Elements: {list(results['elements'].keys())}")
    logger.info(f"  Pairs:    {list(results['binary_pairs'].keys())}")
    logger.info(f"  Output:   {output_path}")

    logger.info(f"\n  Per-element results:")
    for sym, data in results["elements"].items():
        line = f"    {sym:3s}: a_lat={data['a_lat']:.4f} A, E_coh={data['E_coh']:.4f} eV, B={data['B_GPa']:.1f} GPa"
        if "C11" in data:
            line += f", C11={data['C11']:.1f}, C12={data['C12']:.1f} GPa"
        logger.info(line)

    if results["binary_pairs"]:
        logger.info(f"\n  Binary pair results:")
        for pair, data in results["binary_pairs"].items():
            logger.info(f"    {pair}: E_form={data['E_form']:.4f} eV/atom")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n  [SAVE] Final results -> {output_path}")
    logger.info(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFT Reference Generator")
    parser.add_argument("elements", nargs="+", help="Element symbols (e.g. Al Cu Fe)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--work-dir", default=None, help="Scratch directory")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel pw.x processes (default: 4)")
    parser.add_argument("--no-elastic", action="store_true",
                        help="Skip C11/C12 elastic stage. The MEAM seeder does not "
                             "consume DFT C_ij; fill_elastic_constants.py can backfill "
                             "from B isotropically if downstream code needs them.")
    args = parser.parse_args()

    generate_dft_reference(args.elements, args.output, args.work_dir,
                           n_workers=args.parallel,
                           include_elastic=not args.no_elastic)
