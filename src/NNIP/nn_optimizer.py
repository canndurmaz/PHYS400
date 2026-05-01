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

logger = setup_logger("nn_optimizer")


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

def load_training_targets(results_path=None):
    """Load training targets from src/ML/results.json.

    Returns:
        dict: {config_name: {"composition": {...}, "E_GPa": float, "nu": float}}
    """
    if results_path is None:
        results_path = os.path.join(project_root, "src", "ML", "results.json")
    with open(results_path) as f:
        return json.load(f)


# ── NN Surrogate Optimization ────────────────────────────────────────────────

def optimize_nn(lib_path, params_path, opt_spec=None, n_samples=50,
                results_path=None, n_parallel=None):
    """Multi-target NN optimization using results.json training data.

    Args:
        lib_path: path to MEAM library file
        params_path: path to MEAM params file
        opt_spec: dict specifying which parameters to optimize
        n_samples: number of parameter space samples
        results_path: path to results.json (default: src/ML/results.json)
        n_parallel: number of parallel workers for LAMMPS sampling.
                    Default: auto (cpu_count // 2, leaving threads for OpenMP).
                    When launched via mpirun, samples are distributed across
                    MPI ranks and workers run within each rank.
    """
    entries = load_training_targets(results_path)
    n_entries = len(entries)

    # ── Parallelism config ───────────────────────────────────────────────
    n_cpus = mp.cpu_count()
    if n_parallel is None:
        n_parallel = max(1, n_cpus // 2)
    n_parallel = min(n_parallel, n_cpus)
    omp_threads = max(1, n_cpus // (n_parallel * _MPI_SIZE))
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)

    logger.info(f"\nMulti-Target NN Optimization (C11,C12 targets, Huber loss)")
    logger.info(f"Training entries: {n_entries}")
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
    logger.info(f"PHASE 1: Sampling {n_samples} points ({n_parallel} workers)")
    logger.info(f"  Per-sample timeout: {SAMPLE_TIMEOUT_SEC}s")
    logger.info(f"{'='*60}")
    t_phase1 = time.time()
    X_samples = []
    y_samples = []
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_nn")
    os.makedirs(tmp_dir, exist_ok=True)

    def eval_to_flat(vec):
        """Run LAMMPS for all entries, return [C11_1, C12_1, C11_2, ...]."""
        results = eval_all_entries(vec, names, base_lib, base_params, entries, tmp_dir)
        flat = []
        for E, nu in results:
            C11, C12 = _e_nu_pair_to_cij(E, nu)
            flat.extend([C11, C12])
        return np.array(flat)

    # Baseline (always sequential — need it before anything else)
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
    logger.info(f"  Baseline evaluated in {dt_baseline:.1f}s: {y0}")

    # Generate all candidate perturbations upfront
    n_candidates = n_samples * 3
    candidates = []
    for _ in range(n_candidates):
        pert = 1.0 + (np.random.rand(len(initial_vec)) - 0.5) * 0.2
        candidates.append(initial_vec * pert)
    logger.info(f"  Generated {n_candidates} candidate perturbations")

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
            status = "accepted" if valid >= n_entries // 2 else "rejected"
            logger.info(f"  Rank {_MPI_RANK}: [{i+1}/{len(my_candidates)}] {status} (valid={valid}/{n_entries})")
            if valid >= n_entries // 2:
                my_results.append((vec, y))
        # Gather to rank 0
        all_results = _MPI_COMM.gather(my_results, root=0)
        if _MPI_RANK == 0:
            for rank_results in all_results:
                for vec, y in rank_results:
                    if len(X_samples) >= n_samples:
                        break
                    X_samples.append(vec)
                    y_samples.append(y)
                    logger.info(f"  Sample {len(X_samples)}/{n_samples}")
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
                if len(X_samples) >= n_samples:
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
                    logger.info(f"  [{n_evaluated}/{n_candidates}] Worker failed ({elapsed:.0f}s elapsed): {e}")
                    continue
                vec = futures[future]
                valid = sum(1 for k in range(0, len(y), 2) if y[k] > C_REJECT_MIN)
                timed_out = np.all(y == 0)
                if timed_out:
                    n_timed_out += 1
                    n_rejected += 1
                    logger.info(f"  [{n_evaluated}/{n_candidates}] TIMED OUT ({elapsed:.0f}s elapsed, "
                                f"{n_timed_out} timeouts so far)")
                elif valid >= n_entries // 2:
                    X_samples.append(vec)
                    y_samples.append(y)
                    rate = len(X_samples) / elapsed if elapsed > 0 else 0
                    eta = (n_samples - len(X_samples)) / rate if rate > 0 else float('inf')
                    logger.info(f"  [{n_evaluated}/{n_candidates}] Sample {len(X_samples)}/{n_samples} "
                                f"accepted (valid={valid}/{n_entries}, {elapsed:.0f}s elapsed, "
                                f"rate={rate:.2f}/s, ETA={eta:.0f}s)")
                else:
                    n_rejected += 1
                    if n_evaluated % 10 == 0 or n_evaluated <= 5:
                        logger.info(f"  [{n_evaluated}/{n_candidates}] Rejected (valid={valid}/{n_entries}, "
                                    f"{elapsed:.0f}s elapsed, accepted={len(X_samples)}/{n_samples}, "
                                    f"rejected={n_rejected})")
        elapsed = time.time() - t_sampling
        logger.info(f"  Parallel sampling done in {elapsed:.1f}s ({elapsed/60:.1f}min): "
                     f"{n_evaluated} evaluated, {n_rejected} rejected, {n_timed_out} timed out")
    else:
        # ── Sequential fallback ──────────────────────────────────────────
        t_seq = time.time()
        n_rejected_seq = 0
        for idx, vec in enumerate(candidates):
            if len(X_samples) >= n_samples:
                break
            t_sample = time.time()
            y = eval_to_flat(vec)
            dt_sample = time.time() - t_sample
            valid = sum(1 for k in range(0, len(y), 2) if y[k] > C_REJECT_MIN)
            elapsed = time.time() - t_seq
            if valid >= n_entries // 2:
                X_samples.append(vec)
                y_samples.append(y)
                rate = len(X_samples) / elapsed if elapsed > 0 else 0
                logger.info(f"  [{idx+1}/{n_candidates}] Sample {len(X_samples)}/{n_samples} accepted "
                            f"(valid={valid}/{n_entries}, {dt_sample:.1f}s, {elapsed:.0f}s elapsed)")
            else:
                n_rejected_seq += 1
                if idx < 5 or idx % 10 == 0:
                    logger.info(f"  [{idx+1}/{n_candidates}] Rejected (valid={valid}/{n_entries}, "
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

    # 4. Validate with LAMMPS
    t_phase4 = time.time()
    optimized_vec = v_opt.numpy()[0] * X_std + X_mean
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 4: Validating optimized parameters with LAMMPS")
    logger.info(f"  Timeout: {SAMPLE_TIMEOUT_SEC}s")
    logger.info(f"{'='*60}")
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(SAMPLE_TIMEOUT_SEC)
    try:
        y_final = eval_to_flat(optimized_vec)
    except _LammpsTimeout:
        logger.error(f"  Validation timed out after {SAMPLE_TIMEOUT_SEC}s — "
                      f"optimized potential may be unstable")
        y_final = np.zeros(n_entries * 2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    entry_names = list(entries.keys())
    final_predictions = []
    for k, name in enumerate(entry_names):
        E_target = entries[name]["E_GPa"]
        nu_target = entries[name]["nu"]
        # ``y_final`` is now [C11_1, C12_1, C11_2, C12_2, ...]; derive the
        # reportable (E, nu) analytically.
        C11_opt = float(y_final[k * 2])
        C12_opt = float(y_final[k * 2 + 1])
        E_arr, nu_arr = _cij_to_e_nu(np.array([C11_opt]), np.array([C12_opt]))
        E_opt = float(E_arr[0])
        nu_opt = float(nu_arr[0])
        E_err = abs(E_opt - E_target) / max(abs(E_target), 1e-6) * 100
        nu_err = abs(nu_opt - nu_target) / max(abs(nu_target), 1e-6) * 100
        logger.info(f"  {name}:")
        logger.info(f"    C11: opt={C11_opt:.2f}  C12: opt={C12_opt:.2f}")
        logger.info(f"    E:  target={E_target:.2f}  opt={E_opt:.2f}  err={E_err:.1f}%")
        logger.info(f"    nu: target={nu_target:.3f}  opt={nu_opt:.3f}  err={nu_err:.1f}%")
        final_predictions.append({
            "name": name,
            "C11_opt": C11_opt, "C12_opt": C12_opt,
            "E_opt": E_opt, "nu_opt": nu_opt,
        })
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
        "target_vec_cij": target_vec.tolist(),
        "target_vec_enu": target_vec_enu.tolist(),
        "loss": "Huber",
        "huber_delta": HUBER_DELTA,
        "nu_filter_max": NU_FILTER_MAX,
        "C_reject_min": C_REJECT_MIN,
        "final_predictions": final_predictions,
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
    parser.add_argument("--samples", type=int, default=30, help="Number of initial samples")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Number of parallel workers (default: auto = cpu_count/2)")
    parser.add_argument("--results", default=None, help="Path to results.json")
    args = parser.parse_args()

    os.environ["TK_SILENT"] = "1"
    optimize_nn(args.library, args.params, n_samples=args.samples,
                results_path=args.results, n_parallel=args.parallel)
