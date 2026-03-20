#!/usr/bin/env python3
"""NN-based MEAM parameter optimizer.

Trains a neural network surrogate on (MEAM params → E, nu) mappings
using src/ML/results.json as training targets. Each entry in results.json
provides a composition and its LAMMPS-computed (E_GPa, nu).

Workflow:
1. Load training targets from results.json
2. Sample MEAM parameter space (random perturbations) — parallel via MPI + OpenMP
3. For each sample: run LAMMPS for every composition → (E, nu) per entry
4. Train NN surrogate: Input(params) → Dense(20) → Dense(20) → Dense(10) → Output(N×2)
5. Inverse-optimize parameters through NN to match all targets
6. Validate with LAMMPS and save optimized files
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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


# ── LAMMPS evaluation ─────────────────────────────────────────────────────────

def get_elastic_moduli(L, delta=1e-3):
    """Estimate Young's Modulus and Poisson's Ratio via axial strain."""
    try:
        L.command("run 0")
        s0_xx = -L.get_thermo("pxx") * 1e-4
        s0_yy = -L.get_thermo("pyy") * 1e-4
        L.command(f"change_box all x scale {1.0 + delta} remap units box")
        L.command("minimize 1e-10 1e-10 1000 10000")
        s_plus_xx = -L.get_thermo("pxx") * 1e-4
        s_plus_yy = -L.get_thermo("pyy") * 1e-4
        c11 = (s_plus_xx - s0_xx) / delta
        c12 = (s_plus_yy - s0_yy) / delta
        L.command(f"change_box all x scale {1.0 / (1.0 + delta)} remap units box")
        L.command("minimize 1e-10 1e-10 1000 10000")
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

        L.command("minimize 1e-6 1e-8 100 1000")
        E, nu = get_elastic_moduli(L)
        return E, nu
    except Exception:
        logger.warning("_run_lammps_composition failed", exc_info=True)
        return 0.0, 0.5
    finally:
        L.close()


def eval_all_entries(vec, names, base_lib, base_params, entries, tmp_dir):
    """Evaluate LAMMPS for all results.json entries. Returns list of (E, nu) pairs."""
    lib_path, params_path = vector_to_files(vec, names, base_lib, base_params, tmp_dir)
    results = []
    for entry_name, entry_data in entries.items():
        E, nu = _run_lammps_composition(lib_path, params_path, entry_data["composition"])
        results.append((E, nu))
    return results


def _eval_sample_worker(args):
    """Evaluate one parameter vector across all compositions. Process-safe.

    Each worker gets its own tmp directory to avoid file conflicts.
    OMP_NUM_THREADS is set per-worker for LAMMPS internal OpenMP parallelism.
    """
    vec, names, base_lib, base_params, entries, worker_dir, omp_threads = args
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.makedirs(worker_dir, exist_ok=True)
    results = eval_all_entries(vec, names, base_lib, base_params, entries, worker_dir)
    flat = []
    for E, nu in results:
        flat.extend([E, nu])
    return np.array(flat)


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

    logger.info(f"\nMulti-Target NN Optimization")
    logger.info(f"Training entries: {n_entries}")
    logger.info(f"Parallelism: {_MPI_SIZE} MPI rank(s) x {n_parallel} workers x {omp_threads} OMP threads")
    for name, data in entries.items():
        logger.info(f"  {name}: E={data['E_GPa']:.2f} GPa, nu={data['nu']:.3f}")

    # Build target vector: [E_1, nu_1, E_2, nu_2, ...]
    target_vec = []
    for entry_data in entries.values():
        target_vec.extend([entry_data["E_GPa"], entry_data["nu"]])
    target_vec = np.array(target_vec)

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
    logger.info(f"\nSampling {n_samples} points ({n_parallel} workers)...")
    X_samples = []
    y_samples = []
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp_nn")
    os.makedirs(tmp_dir, exist_ok=True)

    def eval_to_flat(vec):
        results = eval_all_entries(vec, names, base_lib, base_params, entries, tmp_dir)
        flat = []
        for E, nu in results:
            flat.extend([E, nu])
        return np.array(flat)

    # Baseline (always sequential — need it before anything else)
    y0 = eval_to_flat(initial_vec)
    X_samples.append(initial_vec)
    y_samples.append(y0)
    logger.info(f"  Baseline: {y0}")

    # Generate all candidate perturbations upfront
    n_candidates = n_samples * 3
    candidates = []
    for _ in range(n_candidates):
        pert = 1.0 + (np.random.rand(len(initial_vec)) - 0.5) * 0.2
        candidates.append(initial_vec * pert)

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
            valid = sum(1 for k in range(0, len(y), 2) if y[k] > 10.0)
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
        with ProcessPoolExecutor(max_workers=n_parallel) as pool:
            futures = {}
            for i, args in enumerate(worker_args):
                futures[pool.submit(_eval_sample_worker, args)] = candidates[i]

            for future in as_completed(futures):
                if len(X_samples) >= n_samples:
                    break
                n_evaluated += 1
                try:
                    y = future.result()
                except Exception as e:
                    n_rejected += 1
                    logger.info(f"  [{n_evaluated}/{n_candidates}] Worker failed: {e}")
                    continue
                vec = futures[future]
                valid = sum(1 for k in range(0, len(y), 2) if y[k] > 10.0)
                if valid >= n_entries // 2:
                    X_samples.append(vec)
                    y_samples.append(y)
                    logger.info(f"  [{n_evaluated}/{n_candidates}] Sample {len(X_samples)}/{n_samples} accepted (valid_entries={valid}/{n_entries})")
                else:
                    n_rejected += 1
                    logger.info(f"  [{n_evaluated}/{n_candidates}] Rejected (valid_entries={valid}/{n_entries})")
        logger.info(f"  Parallel sampling done: {n_evaluated} evaluated, {n_rejected} rejected")
    else:
        # ── Sequential fallback ──────────────────────────────────────────
        for idx, vec in enumerate(candidates):
            if len(X_samples) >= n_samples:
                break
            y = eval_to_flat(vec)
            valid = sum(1 for k in range(0, len(y), 2) if y[k] > 10.0)
            if valid >= n_entries // 2:
                X_samples.append(vec)
                y_samples.append(y)
                logger.info(f"  [{idx+1}/{n_candidates}] Sample {len(X_samples)}/{n_samples} accepted (valid_entries={valid}/{n_entries})")
            else:
                logger.info(f"  [{idx+1}/{n_candidates}] Rejected (valid_entries={valid}/{n_entries})")

    # Clean up worker tmp dirs
    for d in os.listdir(tmp_dir):
        p = os.path.join(tmp_dir, d)
        if os.path.isdir(p) and (d.startswith("w") or d.startswith("r")):
            shutil.rmtree(p, ignore_errors=True)

    logger.info(f"  Collected {len(X_samples)} valid samples")
    X_train = np.array(X_samples)
    y_train = np.array(y_samples)

    # 2. Train NN Surrogate
    logger.info(f"\nTraining NN surrogate ({len(initial_vec)} -> {n_entries * 2})...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(initial_vec),)),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(n_entries * 2),
    ])
    model.compile(optimizer="adam", loss="mse")

    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    X_norm = (X_train - X_mean) / X_std
    y_norm = (y_train - y_mean) / y_std

    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get("loss", 999) if logs else 999
            if epoch % 1000 == 0:
                logger.info(f"  Epoch {epoch}/10000: loss={loss:.6f}")
            if loss < 0.005:
                logger.info(f"  Target loss reached at epoch {epoch} (loss={loss:.6f})")
                self.model.stop_training = True

    history = model.fit(X_norm, y_norm, epochs=10000, verbose=0, callbacks=[ProgressCallback()])
    training_loss_history = history.history.get("loss", [])
    final_loss = model.evaluate(X_norm, y_norm, verbose=0)
    n_epochs = len(training_loss_history)
    logger.info(f"  Training complete: {n_epochs} epochs, final loss: {final_loss:.4f}")

    # 3. Inverse optimization through NN
    n_opt_steps = 3000
    logger.info(f"\nOptimizing through NN surrogate ({n_opt_steps} steps)...")
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
            logger.info(f"  Step {step}/{n_opt_steps}: loss={loss.numpy():.6f}")

    # 4. Validate with LAMMPS
    optimized_vec = v_opt.numpy()[0] * X_std + X_mean
    logger.info("\nValidating optimized parameters with LAMMPS...")
    y_final = eval_to_flat(optimized_vec)

    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    entry_names = list(entries.keys())
    final_predictions = []
    for k, name in enumerate(entry_names):
        E_target = entries[name]["E_GPa"]
        nu_target = entries[name]["nu"]
        E_opt = y_final[k * 2]
        nu_opt = y_final[k * 2 + 1]
        E_err = abs(E_opt - E_target) / max(abs(E_target), 1e-6) * 100
        nu_err = abs(nu_opt - nu_target) / max(abs(nu_target), 1e-6) * 100
        logger.info(f"  {name}:")
        logger.info(f"    E:  target={E_target:.2f}  opt={E_opt:.2f}  err={E_err:.1f}%")
        logger.info(f"    nu: target={nu_target:.3f}  opt={nu_opt:.3f}  err={nu_err:.1f}%")
        final_predictions.append({"name": name, "E_opt": float(E_opt), "nu_opt": float(nu_opt)})
    logger.info("=" * 60)

    # Write diagnostics JSON
    diag_path = os.path.join(os.path.dirname(__file__), "nn_diagnostics.json")
    diagnostics = {
        "training_loss_history": [float(x) for x in training_loss_history],
        "optimization_trajectory": opt_losses,
        "param_names": list(names),
        "initial_vec": initial_vec.tolist(),
        "optimized_vec": optimized_vec.tolist(),
        "target_vec": target_vec.tolist(),
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

    logger.info(f"\nOptimized files saved to {out_dir}")
    logger.info(f"  Library: {final_lib}")
    logger.info(f"  Params:  {final_par}")

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
