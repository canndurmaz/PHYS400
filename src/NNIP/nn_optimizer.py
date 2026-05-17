#!/usr/bin/env python3
"""NN-based MEAM parameter optimizer.

Trains a neural network surrogate on (MEAM params → C11, C12) mappings
using src/ML/results.json as training targets. Each entry in results.json
provides a composition and its LAMMPS-computed (E_GPa, nu); the
elastic constants $C_{11}, C_{12}$ are reconstructed from those values
under the cubic-isotropic algebra.

This is mathematically equivalent to a (params → E, nu) regression but
better-conditioned: the ratio $\\nu = C_{12}/(C_{11}+C_{12})$ is
ill-conditioned in parameter space (small absolute errors in either
$C_{ij}$ blow up the relative error in $\\nu$), so training on the
underlying $C_{ij}$ produces a much smoother loss landscape and
considerably better results for Mg/Zn-rich compositions whose direct
$(E,\\nu)$ regression collapsed in the previous formulation. The loss
is Huber rather than MSE so that outlier samples (Mg/Zn perturbations
that fail mechanical-stability checks) cannot dominate the gradient.

Workflow:
1. Load training targets from results.json (convert E,nu -> C11,C12)
2. Sample MEAM parameter space (random perturbations) — parallel via MPI + OpenMP
3. For each sample: run LAMMPS for every composition → C11,C12 per entry
4. Train NN surrogate: Input(params) → Dense(20) → Dense(20) → Dense(10) → Output(N×2)
   with Huber loss
5. Inverse-optimize parameters through NN to match all C_ij targets
6. Validate with LAMMPS, derive (E, nu) analytically and save optimized files
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import numpy as np
import tensorflow as tf
from lammps import lammps

# ── MPI support (optional — activated when launched via mpirun) ──────────────
try:
    from mpi4py import MPI
    _MPI_COMM = MPI.COMM_WORLD
    _MPI_RANK = _MPI_COMM.Get_rank()
    _MPI_SIZE = _MPI_COMM.Get_size()
except ImportError:
    _MPI_COMM = None
    _MPI_RANK = 0
    _MPI_SIZE = 1

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress tkinter dialogs before importing config (which has module-level code)
os.environ.setdefault("TK_SILENT", "1")

from src.NNIP.meam_io import (
    parse_library, parse_params, params_to_vector, vector_to_files,
)
from src.NNIP.logging_config import setup_logger
from src.NNIP import nn_active_learning as nnal

logger = setup_logger("nn_optimizer")

DEFAULT_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "nn_checkpoint.json")
SAMPLING_MODES = ("random", "lhs", "sobol", "active")


# ── Cubic-isotropic algebra (E, nu) <-> (C11, C12) ───────────────────────────
# Filter ν close to the (1-2ν)=0 singularity to avoid noisy C_ij targets.
NU_FILTER_MAX = 0.48
# Acceptance threshold on C11 (replaces the previous "E > 10 GPa" check
# that gated whether a perturbed parameter sample is accepted into the
# training set).
C_REJECT_MIN = 30.0
# Huber transition point in normalised loss units.
HUBER_DELTA = 1.0


def _e_nu_to_cij(E, nu):
    """Cubic-isotropic: $(E,\\nu) \\to (C_{11}, C_{12})$. Vectorised."""
    s = E / ((1.0 - 2.0 * nu) * (1.0 + nu))
    return s * (1.0 - nu), s * nu


def _cij_to_e_nu(C11, C12):
    """Cubic-isotropic: $(C_{11}, C_{12}) \\to (E, \\nu)$. Safe at small sums."""
    s = C11 + C12
    s_safe = np.where(np.abs(s) < 1e-8, np.sign(s) * 1e-8 + 1e-12, s)
    E = (C11 - C12) * (C11 + 2.0 * C12) / s_safe
    nu = C12 / s_safe
    return E, nu


def _e_nu_pair_to_cij(E, nu):
    """Scalar (E, nu) -> (C11, C12). Returns 0,0 for unphysical inputs so
    that downstream rejection logic can treat them as failed samples."""
    if not (E > 0 and 0.0 < nu < NU_FILTER_MAX):
        return 0.0, 0.0
    arr_C11, arr_C12 = _e_nu_to_cij(np.array([E]), np.array([nu]))
    return float(arr_C11[0]), float(arr_C12[0])


# ── LAMMPS evaluation ─────────────────────────────────────────────────────────

def get_elastic_moduli(L, delta=1e-3):
    """Estimate Young's Modulus and Poisson's Ratio via axial strain."""
    try:
        t0 = time.time()
        L.command("run 0")
        s0_xx = -L.get_thermo("pxx") * 1e-4
        s0_yy = -L.get_thermo("pyy") * 1e-4
        L.command(f"change_box all x scale {1.0 + delta} remap units box")
        L.command("minimize 1e-8 1e-8 500 5000")
        dt1 = time.time() - t0
        s_plus_xx = -L.get_thermo("pxx") * 1e-4
        s_plus_yy = -L.get_thermo("pyy") * 1e-4
        c11 = (s_plus_xx - s0_xx) / delta
        c12 = (s_plus_yy - s0_yy) / delta
        L.command(f"change_box all x scale {1.0 / (1.0 + delta)} remap units box")
        L.command("minimize 1e-8 1e-8 500 5000")
        dt_total = time.time() - t0
        if dt_total > 10:
            logger.warning(f"    get_elastic_moduli slow: minimize1={dt1:.1f}s, total={dt_total:.1f}s")
        E = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)
        nu = c12 / (c11 + c12)
        return E, nu
    except Exception:
        logger.warning("get_elastic_moduli failed", exc_info=True)
        return 0.0, 0.5


def _run_lammps_composition(lib_path, params_path, composition):
    """Run LAMMPS for a single composition dict and return (E_GPa, nu).

    Uses element info directly from the library file rather than relying
    on element.py's POTENTIALS dict, so it works with any generated MEAM file.
    """
    comp = {sym: frac for sym, frac in composition.items() if frac > 0}

    # Read element info directly from the library file
    lib_data = parse_library(lib_path)
    lib_elements = list(lib_data.keys())  # preserves file order

    # Check all composition elements exist in this library
    missing = set(comp.keys()) - set(lib_elements)
    if missing:
        logger.warning(f"Elements {missing} not in library {lib_path}")
        return 0.0, 0.5

    # Active elements in library-file order (preserves index correspondence)
    active_syms = [sym for sym in lib_elements if sym in comp]

    # Weighted average lattice constant and dominant element for lattice type
    a_m = sum(comp[sym] * lib_data[sym]["params"]["alat"] for sym in active_syms)
    dominant = max(active_syms, key=lambda s: comp[s])
    lat_type = lib_data[dominant]["header"][0].lower()

    n_rep = max(1, round(40.0 / a_m))  # ~4nm box

    library_elements = " ".join(lib_elements)
    active_elements = " ".join(active_syms)

    n_atoms = n_rep ** 3 * (4 if lat_type == "fcc" else 2 if lat_type == "bcc" else 1)
    comp_str = "+".join(f"{s}{comp[s]:.2f}" for s in active_syms)
    logger.debug(f"    LAMMPS: {comp_str}, lat={lat_type}, a={a_m:.3f}, rep={n_rep}, ~{n_atoms} atoms")

    L = lammps(cmdargs=["-log", "none", "-screen", "none"])
    try:
        L.command("units metal")
        L.command("atom_style atomic")
        L.command("boundary p p p")
        L.command(f"lattice {lat_type} {a_m:.4f}")
        L.command(f"region box block 0 {n_rep} 0 {n_rep} 0 {n_rep}")
        L.command(f"create_box {len(active_syms)} box")
        L.command("create_atoms 1 box")

        remaining = 1.0
        for i, sym in enumerate(active_syms[1:], start=2):
            frac = comp[sym] / remaining
            L.command(f"set type 1 type/fraction {i} {frac:.6f} 12345")
            remaining -= comp[sym]

        L.command("pair_style meam")
        L.command(f"pair_coeff * * {lib_path} {library_elements} {params_path} {active_elements}")
        for i, sym in enumerate(active_syms, start=1):
            mass = lib_data[sym]["header"][3]
            L.command(f"mass {i} {mass}")

        t_min = time.time()
        L.command("minimize 1e-6 1e-8 100 1000")
        dt_min = time.time() - t_min
        if dt_min > 10:
            logger.warning(f"    LAMMPS minimize took {dt_min:.1f}s for {comp_str}")

        t_elastic = time.time()
        E, nu = get_elastic_moduli(L)
        dt_elastic = time.time() - t_elastic
        if dt_elastic > 10:
            logger.warning(f"    get_elastic_moduli took {dt_elastic:.1f}s for {comp_str}")

        return E, nu
    except Exception:
        logger.warning(f"_run_lammps_composition failed for {comp_str}", exc_info=True)
        return 0.0, 0.5
    finally:
        L.close()


def eval_all_entries(vec, names, base_lib, base_params, entries, tmp_dir):
    """Evaluate LAMMPS for all results.json entries. Returns list of (E, nu) pairs."""
    lib_path, params_path = vector_to_files(vec, names, base_lib, base_params, tmp_dir)
    results = []
    n = len(entries)
    for idx, (entry_name, entry_data) in enumerate(entries.items(), 1):
        comp = entry_data["composition"]
        comp_str = " ".join(f"{sym}:{frac:.3f}" for sym, frac in comp.items() if frac > 0)
        logger.info(f"    [{idx}/{n}] {entry_name} ({comp_str})")
        t0 = time.time()
        E, nu = _run_lammps_composition(lib_path, params_path, comp)
        dt = time.time() - t0
        status = "OK" if E > 10 else "FAILED"
        logger.info(f"           E={E:.2f} GPa, nu={nu:.3f} — {status} ({dt:.1f}s)")
        results.append((E, nu))
    return results


class _LammpsTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _LammpsTimeout("LAMMPS evaluation timed out")


# Per-sample wall-clock timeout in seconds (default 5 minutes per sample)
SAMPLE_TIMEOUT_SEC = int(os.environ.get("NNIP_SAMPLE_TIMEOUT", "300"))


def _eval_sample_worker(args):
    """Evaluate one parameter vector across all compositions. Process-safe.

    Each worker gets its own tmp directory to avoid file conflicts.
    OMP_NUM_THREADS is set per-worker for LAMMPS internal OpenMP parallelism.
    Times out after SAMPLE_TIMEOUT_SEC seconds to prevent hung LAMMPS runs.
    """
    vec, names, base_lib, base_params, entries, worker_dir, omp_threads = args
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.makedirs(worker_dir, exist_ok=True)

    # Set alarm-based timeout (Unix only)
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SAMPLE_TIMEOUT_SEC)
    try:
        results = eval_all_entries(vec, names, base_lib, base_params, entries, worker_dir)
        # Flatten as [C11_1, C12_1, C11_2, C12_2, ...]; unphysical (E, nu)
        # entries become (0, 0) so the dispatcher's rejection logic can
        # treat them as failed.
        flat = []
        for E, nu in results:
            C11, C12 = _e_nu_pair_to_cij(E, nu)
            flat.extend([C11, C12])
        return np.array(flat)
    except _LammpsTimeout:
        # Return zeros so caller treats it as a failed sample
        return np.zeros(len(entries) * 2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Training data loading ────────────────────────────────────────────────────

DEFAULT_TRAIN_PATH = os.path.join(project_root, "src", "ML", "results_train.json")
DEFAULT_VAL_PATH = os.path.join(project_root, "src", "ML", "results_val.json")


def load_targets(path):
    """Load alloy targets from a results-shaped JSON.

    Returns:
        dict: {config_name: {"composition": {...}, "E_GPa": float, "nu": float,
                             "C11_GPa": float (optional), "C12_GPa": float (optional)}}
    """
    with open(path) as f:
        return json.load(f)


# ── Phase-1 checkpoint (resume LAMMPS sampling after interruption) ───────────

def _config_hash(lib_path, params_path, opt_spec, entries):
    """SHA256 over inputs that determine whether cached samples are valid.

    A checkpoint's (vec, y) pairs only mean something for the exact MEAM
    files, opt_spec, and training targets they were generated against.
    """
    h = hashlib.sha256()
    for path in (lib_path, params_path):
        with open(path, "rb") as f:
            h.update(f.read())
    h.update(json.dumps(opt_spec, sort_keys=True).encode())
    h.update(json.dumps(entries, sort_keys=True).encode())
    return h.hexdigest()


def _load_checkpoint(path, config_hash):
    """Load checkpoint if compatible with current config; else return None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"  Failed to read checkpoint {path}: {e}")
        return None
    if data.get("config_hash") != config_hash:
        logger.warning(f"  Checkpoint hash mismatch — ignoring stale samples ({path})")
        return None
    return data


def _save_checkpoint(path, config_hash, param_names, X_samples, y_samples):
    """Atomic snapshot of the current accepted-sample set."""
    data = {
        "config_hash": config_hash,
        "param_names": list(param_names),
        "samples": [
            {"vec": [float(x) for x in v], "y": [float(x) for x in y]}
            for v, y in zip(X_samples, y_samples)
        ],
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ── Active-learning Phase 1 helper ───────────────────────────────────────────

def _evaluate_candidates_parallel(candidates, names, base_lib, base_params,
                                   entries, tmp_dir, omp_threads, n_parallel,
                                   n_entries, label="batch"):
    """Run LAMMPS on each candidate in parallel; return list of (vec, y) pairs
    for those that pass the C_REJECT_MIN acceptance gate.

    Mirrors the rejection logic used by the legacy one-shot dispatcher so
    sample quality is consistent across sampling modes.
    """
    accepted = []
    if not candidates:
        return accepted
    worker_args = [
        (vec, names, base_lib, base_params, entries,
         os.path.join(tmp_dir, f"al_{label}_{i}"), omp_threads)
        for i, vec in enumerate(candidates)
    ]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_parallel) as pool:
        futures = {pool.submit(_eval_sample_worker, args): vec
                   for args, vec in zip(worker_args, candidates)}
        for future in as_completed(futures):
            vec = futures[future]
            try:
                y = future.result()
            except Exception as exc:
                logger.info(f"  [active/{label}] worker failed: {exc}")
                continue
            timed_out = np.all(y == 0)
            valid = sum(1 for k in range(0, len(y), 2) if y[k] > C_REJECT_MIN)
            if timed_out:
                logger.info(f"  [active/{label}] TIMED OUT (valid={valid}/{n_entries})")
            elif valid >= n_entries // 2:
                accepted.append((vec, y))
                logger.info(f"  [active/{label}] accepted (valid={valid}/{n_entries})")
            else:
                logger.info(f"  [active/{label}] rejected (valid={valid}/{n_entries})")
    elapsed = time.time() - t0
    logger.info(f"  [active/{label}] batch done in {elapsed:.1f}s: "
                f"{len(accepted)}/{len(candidates)} accepted")
    return accepted


def _phase1_active_loop(initial_vec, names, base_lib, base_params, entries,
                         X_samples, y_samples, budget, batch_size,
                         seed_size, ensemble_size, pool_size, n_parallel,
                         omp_threads, tmp_dir, seed, checkpoint_fn,
                         n_entries):
    """Active-learning Phase 1: seed with Sobol, then iterate (ensemble
    train → variance score pool → diverse batch select → LAMMPS) until
    ``budget`` accepted samples are collected.

    Mutates ``X_samples`` / ``y_samples`` in place and calls
    ``checkpoint_fn`` after each accepted sample so a killed run resumes
    from the same iteration boundary.
    """
    logger.info(f"  [active] budget={budget}, seed={seed_size}, batch={batch_size}, "
                f"ensemble={ensemble_size}, pool={pool_size}")

    # ── Seed phase: Sobol bootstrap to populate the ensemble ─────────────
    seed_target = max(0, seed_size - len(X_samples))
    if seed_target > 0:
        logger.info(f"  [active/seed] generating {seed_target} Sobol seed samples")
        # Oversample 2x so a few rejected samples don't leave the seed undersized.
        seed_pool = list(nnal.generate_candidates(
            initial_vec, "sobol", seed_target * 2, seed=seed))
        accepted = _evaluate_candidates_parallel(
            seed_pool[:seed_target * 2], names, base_lib, base_params,
            entries, tmp_dir, omp_threads, n_parallel, n_entries,
            label="seed")
        for vec, y in accepted:
            if len(X_samples) >= budget:
                break
            X_samples.append(np.asarray(vec))
            y_samples.append(np.asarray(y))
            checkpoint_fn()
    else:
        logger.info(f"  [active/seed] resume: {len(X_samples)} samples already on disk, "
                    f"skipping seed")

    # ── Iteration loop ──────────────────────────────────────────────────
    iteration = 0
    while len(X_samples) < budget:
        iteration += 1
        remaining = budget - len(X_samples)
        this_batch = min(batch_size, remaining)
        logger.info(f"  [active/iter {iteration}] {len(X_samples)}/{budget} accepted; "
                    f"picking batch of {this_batch} via ensemble variance")
        t_pick = time.time()
        next_batch = nnal.pick_next_batch(
            np.asarray(initial_vec),
            np.asarray(X_samples), np.asarray(y_samples),
            batch_size=this_batch, pool_size=pool_size,
            ensemble_size=ensemble_size,
            seed=seed + iteration,
        )
        logger.info(f"  [active/iter {iteration}] batch picked in {time.time()-t_pick:.1f}s")
        accepted = _evaluate_candidates_parallel(
            [v for v in next_batch], names, base_lib, base_params,
            entries, tmp_dir, omp_threads, n_parallel, n_entries,
            label=f"iter{iteration}")
        if not accepted:
            logger.warning(f"  [active/iter {iteration}] no candidates accepted — "
                            f"continuing; if this persists the perturbation box may "
                            f"be too wide for the current potential")
            # Safety valve: if 5 consecutive iterations yield nothing, give up
            # rather than spin forever on a broken potential.
            if iteration >= 5 and len(X_samples) == seed_size:
                logger.error(f"  [active] aborting after 5 fruitless iterations; "
                              f"have {len(X_samples)} samples")
                break
            continue
        for vec, y in accepted:
            if len(X_samples) >= budget:
                break
            X_samples.append(np.asarray(vec))
            y_samples.append(np.asarray(y))
            checkpoint_fn()

    logger.info(f"  [active] phase complete: {len(X_samples)} samples in {iteration} iterations")


# ── NN Surrogate Optimization ────────────────────────────────────────────────

def optimize_nn(lib_path, params_path, opt_spec=None, n_perturbations=150,
                train_path=None, val_path=None, n_parallel=None,
                checkpoint_path=None, resume=True,
                sampling_mode="random", seed_size=20, batch_size=5,
                ensemble_size=5, pool_size=500, sampling_seed=0):
    """Multi-target NN optimization with a held-out validation set.

    Phases 1-3 fit and invert a surrogate against the ``train_path`` alloys.
    Phase 4 runs LAMMPS on the optimized potential against the disjoint
    ``val_path`` alloys and reports per-entry + aggregate generalisation
    error — never against the training set.

    Args:
        lib_path: path to MEAM library file
        params_path: path to MEAM params file
        opt_spec: dict specifying which parameters to optimize
        n_perturbations: total Phase-1 sample budget. Every accepted
                         (vec, y) pair counts against this number regardless
                         of sampling mode.
        train_path: path to training subset JSON (default:
                    src/ML/results_train.json — produced by
                    select_representatives.py)
        val_path: path to validation subset JSON (default:
                  src/ML/results_val.json — produced by
                  select_representatives.py). Must be disjoint from train.
        n_parallel: number of parallel workers for LAMMPS sampling.
                    Default: auto (cpu_count // 2, leaving threads for OpenMP).
                    When launched via mpirun, samples are distributed across
                    MPI ranks and workers run within each rank.
        checkpoint_path: Phase-1 sample checkpoint (default:
                         nn_checkpoint.json next to this script). Rewritten
                         atomically after every accepted sample so a killed
                         run can resume without re-evaluating LAMMPS.
        resume: if True, reuse a compatible checkpoint at startup.
        sampling_mode: how Phase-1 picks candidates. One of
                       ``"random"`` (legacy uniform-random ±10 % box),
                       ``"lhs"`` (Latin Hypercube — same budget, better
                       coverage), ``"sobol"`` (scrambled Sobol' — best
                       low-discrepancy fill), ``"active"`` (NN-ensemble
                       active learning with diverse batch selection;
                       typically needs ~3x fewer LAMMPS calls for the same
                       surrogate quality). Default ``"random"`` preserves
                       legacy behaviour.
        seed_size: active mode only — number of bootstrap samples drawn
                   via Sobol before the iteration loop starts (default 20).
        batch_size: active mode only — number of candidates selected per
                    iteration (default 5; should match ``n_parallel`` for
                    best throughput).
        ensemble_size: active mode only — number of NN surrogates in the
                       acquisition ensemble (default 5).
        pool_size: active mode only — size of the Sobol candidate pool
                   scored each iteration (default 500).
        sampling_seed: RNG seed for candidate generation and the
                       acquisition ensemble (default 0). Distinct from
                       ``--split-seed`` which controls representative
                       selection upstream.
    """
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, "
                         f"got {sampling_mode!r}")
    if sampling_mode == "active" and _MPI_SIZE > 1:
        raise RuntimeError("active sampling requires single-rank execution; "
                           "the acquisition step is not yet MPI-distributed. "
                           "Use --sampling random|lhs|sobol under mpirun.")
    if train_path is None:
        train_path = DEFAULT_TRAIN_PATH
    if val_path is None:
        val_path = DEFAULT_VAL_PATH
    entries = load_targets(train_path)
    val_entries = load_targets(val_path)
    n_entries = len(entries)
    n_val = len(val_entries)
    # Guard against accidental overlap (would silently turn val into train).
    overlap = set(entries.keys()) & set(val_entries.keys())
    if overlap:
        raise ValueError(f"train/val overlap detected ({len(overlap)} entries): "
                         f"{sorted(overlap)[:5]}...")

    # ── Parallelism config ───────────────────────────────────────────────
    n_cpus = mp.cpu_count()
    if n_parallel is None:
        n_parallel = max(1, n_cpus // 2)
    n_parallel = min(n_parallel, n_cpus)
    omp_threads = max(1, n_cpus // (n_parallel * _MPI_SIZE))
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    logger.info(f"\nMulti-Target NN Optimization (C11,C12 targets, Huber loss)")
    logger.info(f"Training entries:   {n_entries}  ({train_path})")
    logger.info(f"Validation entries: {n_val}  ({val_path})")
    logger.info(f"Parallelism: {_MPI_SIZE} MPI rank(s) x {n_parallel} workers x {omp_threads} OMP threads")

    def _target_cij(entry_data):
        """Return (C11, C12) for a target alloy. Prefer stored values
        from results.json; fall back to algebraic conversion only when
        not present (older datasets)."""
        if "C11_GPa" in entry_data and "C12_GPa" in entry_data:
            return float(entry_data["C11_GPa"]), float(entry_data["C12_GPa"])
        return _e_nu_pair_to_cij(entry_data["E_GPa"], entry_data["nu"])

    n_stored = 0
    for name, data in entries.items():
        E_t, nu_t = data["E_GPa"], data["nu"]
        C11_t, C12_t = _target_cij(data)
        if "C11_GPa" in data:
            n_stored += 1
        logger.info(f"  {name}: E={E_t:.2f} GPa, nu={nu_t:.3f}  =>  "
                    f"C11={C11_t:.2f}, C12={C12_t:.2f}")
    logger.info(f"  ({n_stored}/{n_entries} target alloys used stored C_ij; "
                f"{n_entries - n_stored} fell back to algebraic conversion)")

    # Build target vector: [C11_1, C12_1, C11_2, C12_2, ...]
    target_vec_enu = []   # kept for diagnostics / final reporting
    target_vec = []
    for entry_data in entries.values():
        E_t, nu_t = entry_data["E_GPa"], entry_data["nu"]
        target_vec_enu.extend([E_t, nu_t])
        C11_t, C12_t = _target_cij(entry_data)
        target_vec.extend([C11_t, C12_t])
    target_vec = np.array(target_vec)
    target_vec_enu = np.array(target_vec_enu)
    if np.any(target_vec == 0):
        logger.warning("  At least one target alloy has unphysical (E, nu); "
                       "its C_ij target was set to (0, 0).")

    base_lib = parse_library(lib_path)
    base_params = parse_params(params_path)

    if opt_spec is None:
        # Default: optimize all binary pair Ec, re, alpha
        n_elements = len(base_lib)
        par_keys = []
        for i in range(1, n_elements + 1):
            for j in range(i + 1, n_elements + 1):
                for pname in ("Ec", "re", "alpha"):
                    par_keys.append(f"{pname}({i},{j})")
        # Filter to keys that actually exist in the params file
        existing_keys = {k for k, _ in base_params}
        par_keys = [k for k in par_keys if k in existing_keys]
        opt_spec = {"params": par_keys}

    initial_vec, names = params_to_vector(base_lib, base_params, opt_spec)
    logger.info(f"Optimizing {len(names)} parameters: {names}")

    # 1. Sample parameter space (parallel)
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 1: Sampling {n_perturbations} points ({n_parallel} workers)")
    logger.info(f"  Per-sample timeout: {SAMPLE_TIMEOUT_SEC}s")
    logger.info(f"{'='*60}")
    t_phase1 = time.time()
    X_samples = []
    y_samples = []
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_nn")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Resume from checkpoint (Phase 1 only) ─────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    config_hash = _config_hash(lib_path, params_path, opt_spec, entries)
    if resume and _MPI_RANK == 0:
        loaded = _load_checkpoint(checkpoint_path, config_hash)
        if loaded is not None:
            for s in loaded["samples"]:
                X_samples.append(np.array(s["vec"]))
                y_samples.append(np.array(s["y"]))
            logger.info(f"  Resumed from checkpoint {checkpoint_path}: "
                        f"{len(X_samples)} samples loaded")
    if _MPI_SIZE > 1:
        X_samples = _MPI_COMM.bcast(X_samples, root=0)
        y_samples = _MPI_COMM.bcast(y_samples, root=0)

    def _checkpoint_now():
        if _MPI_RANK == 0:
            _save_checkpoint(checkpoint_path, config_hash, names,
                             X_samples, y_samples)

    def eval_to_flat(vec):
        """Run LAMMPS for all entries, return [C11_1, C12_1, C11_2, ...]."""
        results = eval_all_entries(vec, names, base_lib, base_params, entries, tmp_dir)
        flat = []
        for E, nu in results:
            C11, C12 = _e_nu_pair_to_cij(E, nu)
            flat.extend([C11, C12])
        return np.array(flat)

    # Baseline (always sequential — need it before anything else)
    if not X_samples:
        logger.info(f"  Evaluating baseline parameters (timeout={SAMPLE_TIMEOUT_SEC}s)...")
        t_baseline = time.time()
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(SAMPLE_TIMEOUT_SEC)
        try:
            y0 = eval_to_flat(initial_vec)
        except _LammpsTimeout:
            logger.error(f"  FATAL: Baseline evaluation timed out after {SAMPLE_TIMEOUT_SEC}s. "
                          f"The initial MEAM potential is too unstable for LAMMPS to minimize.")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            raise RuntimeError(f"Baseline LAMMPS evaluation timed out after {SAMPLE_TIMEOUT_SEC}s")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        dt_baseline = time.time() - t_baseline
        X_samples.append(initial_vec)
        y_samples.append(y0)
        _checkpoint_now()
        logger.info(f"  Baseline evaluated in {dt_baseline:.1f}s: {y0}")
    else:
        logger.info(f"  Skipping baseline — {len(X_samples)} samples already loaded from checkpoint")

    logger.info(f"  Sampling mode: {sampling_mode}")
    if sampling_mode == "active":
        _phase1_active_loop(
            initial_vec=initial_vec, names=names,
            base_lib=base_lib, base_params=base_params, entries=entries,
            X_samples=X_samples, y_samples=y_samples,
            budget=n_perturbations, batch_size=batch_size,
            seed_size=seed_size, ensemble_size=ensemble_size,
            pool_size=pool_size, n_parallel=n_parallel,
            omp_threads=omp_threads, tmp_dir=tmp_dir,
            seed=sampling_seed,
            checkpoint_fn=_checkpoint_now,
            n_entries=n_entries,
        )
        candidates = []  # downstream oneshot block is a no-op for active
    else:
        # Generate candidate perturbations for the *remaining* sample budget.
        # Oversample by 3x (random) or 1.5x (LHS/Sobol — better coverage
        # means fewer rejections).
        n_remaining = max(0, n_perturbations - len(X_samples))
        oversample = 3 if sampling_mode == "random" else 2
        n_candidates = n_remaining * oversample
        if n_candidates == 0:
            logger.info(f"  Already have {len(X_samples)}/{n_perturbations} samples from checkpoint — "
                        f"skipping perturbation phase")
            candidates = []
        else:
            candidates = list(nnal.generate_candidates(
                initial_vec, sampling_mode, n_candidates, seed=sampling_seed))
            logger.info(f"  Generated {n_candidates} candidate perturbations via {sampling_mode} "
                        f"(need {n_remaining} more samples)")

    if _MPI_SIZE > 1:
        # ── MPI mode: scatter candidates across ranks ────────────────────
        my_candidates = candidates[_MPI_RANK::_MPI_SIZE]
        logger.info(f"  MPI rank {_MPI_RANK}/{_MPI_SIZE}: evaluating {len(my_candidates)} candidates")
        my_results = []
        for i, vec in enumerate(my_candidates):
            worker_dir = os.path.join(tmp_dir, f"r{_MPI_RANK}_w{i}")
            y = _eval_sample_worker(
                (vec, names, base_lib, base_params, entries, worker_dir, omp_threads)
            )
            valid = sum(1 for k in range(0, len(y), 2) if y[k] > C_REJECT_MIN)
            if valid >= n_entries // 2:
                my_results.append((vec, y))
            status = "accepted" if valid >= n_entries // 2 else "rejected"
            logger.info(f"  Rank {_MPI_RANK}: [{i+1}/{len(my_candidates)}] {status} "
                        f"(OK {len(my_results)}/{n_perturbations} on this rank, valid={valid}/{n_entries})")
        # Gather to rank 0
        all_results = _MPI_COMM.gather(my_results, root=0)
        if _MPI_RANK == 0:
            for rank_results in all_results:
                for vec, y in rank_results:
                    if len(X_samples) >= n_perturbations:
                        break
                    X_samples.append(vec)
                    y_samples.append(y)
                    _checkpoint_now()
                    logger.info(f"  Sample {len(X_samples)}/{n_perturbations}")
        # Broadcast collected samples to all ranks
        X_samples = _MPI_COMM.bcast(X_samples, root=0)
        y_samples = _MPI_COMM.bcast(y_samples, root=0)
    elif n_parallel > 1:
        # ── Local multiprocessing mode ───────────────────────────────────
        worker_args = [
            (vec, names, base_lib, base_params, entries,
             os.path.join(tmp_dir, f"w{i}"), omp_threads)
            for i, vec in enumerate(candidates)
        ]
        n_evaluated = 0
        n_rejected = 0
        n_timed_out = 0
        t_sampling = time.time()
        with ProcessPoolExecutor(max_workers=n_parallel) as pool:
            futures = {}
            for i, args in enumerate(worker_args):
                futures[pool.submit(_eval_sample_worker, args)] = candidates[i]

            for future in as_completed(futures):
                if len(X_samples) >= n_perturbations:
                    # Cancel all remaining futures
                    n_cancelled = 0
                    for f in futures:
                        if f.cancel():
                            n_cancelled += 1
                    elapsed = time.time() - t_sampling
                    logger.info(f"  Enough samples collected. Cancelled {n_cancelled} remaining futures. "
                                f"Sampling took {elapsed:.1f}s ({elapsed/60:.1f}min)")
                    break
                n_evaluated += 1
                elapsed = time.time() - t_sampling
                try:
                    y = future.result()
                except Exception as e:
                    n_rejected += 1
                    logger.info(f"  [{n_evaluated}/{n_candidates}] Worker failed "
                                f"(OK {len(X_samples)}/{n_perturbations}, {elapsed:.0f}s elapsed): {e}")
                    continue
                vec = futures[future]
                valid = sum(1 for k in range(0, len(y), 2) if y[k] > C_REJECT_MIN)
                timed_out = np.all(y == 0)
                if timed_out:
                    n_timed_out += 1
                    n_rejected += 1
                    logger.info(f"  [{n_evaluated}/{n_candidates}] TIMED OUT "
                                f"(OK {len(X_samples)}/{n_perturbations}, {elapsed:.0f}s elapsed, "
                                f"{n_timed_out} timeouts so far)")
                elif valid >= n_entries // 2:
                    X_samples.append(vec)
                    y_samples.append(y)
                    _checkpoint_now()
                    rate = len(X_samples) / elapsed if elapsed > 0 else 0
                    eta = (n_perturbations - len(X_samples)) / rate if rate > 0 else float('inf')
                    logger.info(f"  [{n_evaluated}/{n_candidates}] OK {len(X_samples)}/{n_perturbations} "
                                f"accepted (valid={valid}/{n_entries}, {elapsed:.0f}s elapsed, "
                                f"rate={rate:.2f}/s, ETA={eta:.0f}s)")
                else:
                    n_rejected += 1
                    logger.info(f"  [{n_evaluated}/{n_candidates}] Rejected "
                                f"(OK {len(X_samples)}/{n_perturbations}, valid={valid}/{n_entries}, "
                                f"{elapsed:.0f}s elapsed, rejected={n_rejected})")
        elapsed = time.time() - t_sampling
        logger.info(f"  Parallel sampling done in {elapsed:.1f}s ({elapsed/60:.1f}min): "
                     f"{n_evaluated} evaluated, {n_rejected} rejected, {n_timed_out} timed out")
    else:
        # ── Sequential fallback ──────────────────────────────────────────
        t_seq = time.time()
        n_rejected_seq = 0
        for idx, vec in enumerate(candidates):
            if len(X_samples) >= n_perturbations:
                break
            t_sample = time.time()
            y = eval_to_flat(vec)
            dt_sample = time.time() - t_sample
            valid = sum(1 for k in range(0, len(y), 2) if y[k] > C_REJECT_MIN)
            elapsed = time.time() - t_seq
            if valid >= n_entries // 2:
                X_samples.append(vec)
                y_samples.append(y)
                _checkpoint_now()
                rate = len(X_samples) / elapsed if elapsed > 0 else 0
                logger.info(f"  [{idx+1}/{n_candidates}] OK {len(X_samples)}/{n_perturbations} accepted "
                            f"(valid={valid}/{n_entries}, {dt_sample:.1f}s, {elapsed:.0f}s elapsed)")
            else:
                n_rejected_seq += 1
                logger.info(f"  [{idx+1}/{n_candidates}] Rejected "
                            f"(OK {len(X_samples)}/{n_perturbations}, valid={valid}/{n_entries}, "
                            f"{dt_sample:.1f}s, rejected={n_rejected_seq}, {elapsed:.0f}s elapsed)")

    # Clean up worker tmp dirs
    for d in os.listdir(tmp_dir):
        p = os.path.join(tmp_dir, d)
        if os.path.isdir(p) and (d.startswith("w") or d.startswith("r")):
            shutil.rmtree(p, ignore_errors=True)

    dt_phase1 = time.time() - t_phase1
    logger.info(f"  Collected {len(X_samples)} valid samples")
    logger.info(f"  PHASE 1 total: {dt_phase1:.1f}s ({dt_phase1/60:.1f}min)")

    if len(X_samples) < 3:
        logger.error(f"  FATAL: Only {len(X_samples)} valid samples (need at least 3). "
                      f"Potential is likely too unstable for these compositions.")
        raise RuntimeError(f"Insufficient samples: {len(X_samples)} < 3")

    X_train = np.array(X_samples)
    y_train = np.array(y_samples)

    # 2. Train NN Surrogate
    t_phase2 = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 2: Training NN surrogate ({len(initial_vec)} -> {n_entries * 2})")
    logger.info(f"  Training data shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"{'='*60}")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(initial_vec),)),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(n_entries * 2),
    ])
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.Huber(delta=HUBER_DELTA))

    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    n_zero_x = np.sum(X_std == 0)
    n_zero_y = np.sum(y_std == 0)
    X_std[X_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    if n_zero_x or n_zero_y:
        logger.warning(f"  Zero-variance features: {n_zero_x} in X, {n_zero_y} in y (clamped to 1.0)")

    X_norm = (X_train - X_mean) / X_std
    y_norm = (y_train - y_mean) / y_std

    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._t_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get("loss", 999) if logs else 999
            if epoch % 1000 == 0:
                elapsed = time.time() - self._t_start
                logger.info(f"  Epoch {epoch}/10000: loss={loss:.6f} ({elapsed:.0f}s elapsed)")
            if loss < 0.005:
                elapsed = time.time() - self._t_start
                logger.info(f"  Target loss reached at epoch {epoch} (loss={loss:.6f}, {elapsed:.0f}s)")
                self.model.stop_training = True

    history = model.fit(X_norm, y_norm, epochs=10000, verbose=0, callbacks=[ProgressCallback()])
    training_loss_history = history.history.get("loss", [])
    final_loss = model.evaluate(X_norm, y_norm, verbose=0)
    n_epochs = len(training_loss_history)
    dt_phase2 = time.time() - t_phase2
    logger.info(f"  Training complete: {n_epochs} epochs, final loss: {final_loss:.4f}")
    logger.info(f"  PHASE 2 total: {dt_phase2:.1f}s ({dt_phase2/60:.1f}min)")

    # 3. Inverse optimization through NN
    t_phase3 = time.time()
    n_opt_steps = 3000
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 3: Inverse optimization through NN ({n_opt_steps} steps)")
    logger.info(f"{'='*60}")
    target_norm = (target_vec - y_mean) / y_std
    v_opt = tf.Variable(tf.zeros((1, len(initial_vec)), dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    opt_losses = []
    for step in range(n_opt_steps):
        with tf.GradientTape() as tape:
            pred_norm = model(v_opt)
            loss = tf.reduce_mean(tf.square(pred_norm - target_norm))
        grads = tape.gradient(loss, v_opt)
        optimizer.apply_gradients([(grads, v_opt)])
        # Constrain optimization to region near training samples (+/- 2.5 std devs)
        v_opt.assign(tf.clip_by_value(v_opt, -2.5, 2.5))
        opt_losses.append(float(loss.numpy()))
        if step % 500 == 0 or step == n_opt_steps - 1:
            elapsed = time.time() - t_phase3
            grad_norm = float(tf.norm(grads).numpy()) if grads is not None else 0.0
            logger.info(f"  Step {step}/{n_opt_steps}: loss={loss.numpy():.6f}, "
                        f"grad_norm={grad_norm:.6f}, {elapsed:.1f}s elapsed")

    dt_phase3 = time.time() - t_phase3
    logger.info(f"  PHASE 3 total: {dt_phase3:.1f}s ({dt_phase3/60:.1f}min)")

    # 4. Held-out validation: run LAMMPS on the 30% the surrogate never saw
    t_phase4 = time.time()
    optimized_vec = v_opt.numpy()[0] * X_std + X_mean
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 4: Held-out validation on {n_val} alloys")
    logger.info(f"  Source: {val_path}")
    logger.info(f"  Timeout: {SAMPLE_TIMEOUT_SEC}s")
    logger.info(f"{'='*60}")
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SAMPLE_TIMEOUT_SEC)
    try:
        val_pairs = eval_all_entries(optimized_vec, names, base_lib, base_params,
                                     val_entries, tmp_dir)
    except _LammpsTimeout:
        logger.error(f"  Validation timed out after {SAMPLE_TIMEOUT_SEC}s — "
                      f"optimized potential may be unstable")
        val_pairs = [(0.0, 0.5)] * n_val
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS (held-out — never seen during training)")
    logger.info("=" * 60)
    val_predictions = []
    E_errs, nu_errs = [], []
    for (name, data), (E_opt, nu_opt) in zip(val_entries.items(), val_pairs):
        E_target = data["E_GPa"]
        nu_target = data["nu"]
        C11_t, C12_t = _target_cij(data)
        C11_opt, C12_opt = _e_nu_pair_to_cij(E_opt, nu_opt)
        E_err = abs(E_opt - E_target) / max(abs(E_target), 1e-6) * 100
        nu_err = abs(nu_opt - nu_target) / max(abs(nu_target), 1e-6) * 100
        logger.info(f"  {name}:")
        logger.info(f"    C11: target={C11_t:.2f}  opt={C11_opt:.2f}")
        logger.info(f"    C12: target={C12_t:.2f}  opt={C12_opt:.2f}")
        logger.info(f"    E:   target={E_target:.2f}  opt={E_opt:.2f}  err={E_err:.1f}%")
        logger.info(f"    nu:  target={nu_target:.3f}  opt={nu_opt:.3f}  err={nu_err:.1f}%")
        val_predictions.append({
            "name": name,
            "C11_target": C11_t, "C12_target": C12_t,
            "C11_opt": C11_opt, "C12_opt": C12_opt,
            "E_target": E_target, "E_opt": E_opt, "E_err_pct": round(E_err, 2),
            "nu_target": nu_target, "nu_opt": nu_opt, "nu_err_pct": round(nu_err, 2),
        })
        if E_opt > 0:
            E_errs.append(E_err)
            nu_errs.append(nu_err)

    if E_errs:
        val_rmse_e = float(np.sqrt(np.mean(np.array(E_errs) ** 2)))
        val_rmse_nu = float(np.sqrt(np.mean(np.array(nu_errs) ** 2)))
        val_mean_e = float(np.mean(E_errs))
        val_mean_nu = float(np.mean(nu_errs))
        logger.info(f"\n  Aggregate over {len(E_errs)}/{n_val} physically-valid alloys:")
        logger.info(f"    E:  mean err {val_mean_e:.1f}%  RMSE {val_rmse_e:.1f}%")
        logger.info(f"    nu: mean err {val_mean_nu:.1f}%  RMSE {val_rmse_nu:.1f}%")
    else:
        val_rmse_e = val_rmse_nu = val_mean_e = val_mean_nu = float("nan")
        logger.warning(f"  No physically-valid val alloys — optimized potential may be broken")

    dt_phase4 = time.time() - t_phase4
    logger.info(f"  PHASE 4 total: {dt_phase4:.1f}s ({dt_phase4/60:.1f}min)")
    logger.info("=" * 60)

    # Write diagnostics JSON
    diag_path = os.path.join(os.path.dirname(__file__), "nn_diagnostics.json")
    diagnostics = {
        "training_loss_history": [float(x) for x in training_loss_history],
        "optimization_trajectory": opt_losses,
        "param_names": list(names),
        "initial_vec": initial_vec.tolist(),
        "optimized_vec": optimized_vec.tolist(),
        "train_target_vec_cij": target_vec.tolist(),
        "train_target_vec_enu": target_vec_enu.tolist(),
        "train_path": train_path,
        "val_path": val_path,
        "loss": "Huber",
        "huber_delta": HUBER_DELTA,
        "nu_filter_max": NU_FILTER_MAX,
        "C_reject_min": C_REJECT_MIN,
        "val_predictions": val_predictions,
        "val_metrics": {
            "rmse_E_pct": val_rmse_e,
            "rmse_nu_pct": val_rmse_nu,
            "mean_E_pct": val_mean_e,
            "mean_nu_pct": val_mean_nu,
            "n_valid": len(E_errs),
            "n_val": n_val,
        },
    }
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info(f"  Diagnostics saved to {diag_path}")

    # 5. Save optimized files
    out_dir = os.path.join(project_root, "EAM", "optimized")
    os.makedirs(out_dir, exist_ok=True)
    lib_out, par_out = vector_to_files(optimized_vec, names, base_lib, base_params, out_dir)

    lib_base = os.path.basename(lib_path)
    par_base = os.path.basename(params_path)
    final_lib = os.path.join(out_dir, f"optimized_{lib_base}")
    final_par = os.path.join(out_dir, f"optimized_{par_base}")
    os.replace(lib_out, final_lib)
    os.replace(par_out, final_par)

    dt_total = time.time() - t_phase1
    logger.info(f"\nOptimized files saved to {out_dir}")
    logger.info(f"  Library: {final_lib}")
    logger.info(f"  Params:  {final_par}")
    logger.info(f"\nNN Optimization total: {dt_total:.1f}s ({dt_total/60:.1f}min)")
    logger.info(f"  Phase 1 (sampling):     {dt_phase1:.1f}s")
    logger.info(f"  Phase 2 (NN training):  {dt_phase2:.1f}s")
    logger.info(f"  Phase 3 (optimization): {dt_phase3:.1f}s")
    logger.info(f"  Phase 4 (validation):   {dt_phase4:.1f}s")

    return final_lib, final_par


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Target NN MEAM Optimizer")
    parser.add_argument("--library", required=True, help="Path to MEAM library file")
    parser.add_argument("--params", required=True, help="Path to MEAM params file")
    parser.add_argument("--perturbations", type=int, default=150,
                        help="Total Phase-1 sample budget (default: 150). "
                             "Used by all sampling modes — the legacy flag name is "
                             "kept for backwards compatibility.")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Number of parallel workers (default: auto = cpu_count/2)")
    parser.add_argument("--train", default=None,
                        help=f"Training-subset JSON (default: {DEFAULT_TRAIN_PATH})")
    parser.add_argument("--val", default=None,
                        help=f"Validation-subset JSON (default: {DEFAULT_VAL_PATH})")
    parser.add_argument("--checkpoint", default=None,
                        help="Phase-1 sample checkpoint path "
                             "(default: nn_checkpoint.json next to this script)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoint and start sampling from scratch")
    parser.add_argument("--sampling", choices=SAMPLING_MODES, default="random",
                        help="Phase-1 sampling strategy. "
                             "'random' (default) is the legacy uniform-random ±10%% box; "
                             "'lhs' uses Latin Hypercube; 'sobol' uses scrambled Sobol'; "
                             "'active' runs ensemble-variance active learning "
                             "(typically ~3x fewer LAMMPS calls for the same surrogate quality).")
    parser.add_argument("--seed-size", type=int, default=20,
                        help="Active mode only: bootstrap samples drawn before iteration starts (default: 20)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Active mode only: candidates selected per iteration (default: 5; "
                             "should match --parallel for best throughput)")
    parser.add_argument("--ensemble-size", type=int, default=5,
                        help="Active mode only: NNs in the acquisition ensemble (default: 5)")
    parser.add_argument("--pool-size", type=int, default=500,
                        help="Active mode only: Sobol candidate pool scored each iteration (default: 500)")
    parser.add_argument("--sampling-seed", type=int, default=0,
                        help="Seed for candidate generation and acquisition ensemble (default: 0)")
    args = parser.parse_args()

    os.environ["TK_SILENT"] = "1"
    optimize_nn(args.library, args.params, n_perturbations=args.perturbations,
                train_path=args.train, val_path=args.val,
                n_parallel=args.parallel,
                checkpoint_path=args.checkpoint, resume=not args.no_resume,
                sampling_mode=args.sampling, seed_size=args.seed_size,
                batch_size=args.batch_size, ensemble_size=args.ensemble_size,
                pool_size=args.pool_size, sampling_seed=args.sampling_seed)
