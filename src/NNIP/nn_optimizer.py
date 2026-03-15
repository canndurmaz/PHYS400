#!/usr/bin/env python3
"""NN-based MEAM parameter optimizer.

Trains a neural network surrogate on (MEAM params → E, nu) mappings
using src/ML/results.json as training targets. Each entry in results.json
provides a composition and its LAMMPS-computed (E_GPa, nu).

Workflow:
1. Load training targets from results.json
2. Sample MEAM parameter space (random perturbations)
3. For each sample: run LAMMPS for every composition → (E, nu) per entry
4. Train NN surrogate: Input(params) → Dense(64) → Dense(64) → Dense(32) → Output(N×2)
5. Inverse-optimize parameters through NN to match all targets
6. Validate with LAMMPS and save optimized files
"""

import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf
from lammps import lammps

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress tkinter dialogs before importing config (which has module-level code)
os.environ.setdefault("TK_SILENT", "1")

from src.NNIP.meam_io import (
    parse_library, parse_params, params_to_vector, vector_to_files,
)
from src.MD.config import find_potential
from src.MD.element import ELEMENTS


# ── LAMMPS evaluation ─────────────────────────────────────────────────────────

def get_elastic_moduli(L, delta=1e-3):
    """Estimate Young's Modulus and Poisson's Ratio via axial strain."""
    try:
        L.command("run 0")
        s0_xx = -L.get_thermo("pxx") * 1e-4
        s0_yy = -L.get_thermo("pyy") * 1e-4
        L.command(f"change_box all x scale {1.0 + delta} remap units box")
        L.command("minimize 1e-10 1e-10 100 1000")
        s_plus_xx = -L.get_thermo("pxx") * 1e-4
        s_plus_yy = -L.get_thermo("pyy") * 1e-4
        c11 = (s_plus_xx - s0_xx) / delta
        c12 = (s_plus_yy - s0_yy) / delta
        L.command(f"change_box all x scale {1.0 / (1.0 + delta)} remap units box")
        L.command("minimize 1e-10 1e-10 100 1000")
        E = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)
        nu = c12 / (c11 + c12)
        return E, nu
    except Exception:
        return 0.0, 0.5


def _run_lammps_composition(lib_path, params_path, composition):
    """Run LAMMPS for a single composition dict and return (E_GPa, nu)."""
    comp = {sym: frac for sym, frac in composition.items() if frac > 0}
    symbols = sorted(comp.keys())

    try:
        pot = find_potential(symbols)
    except RuntimeError:
        return 0.0, 0.5

    sel = sorted(
        [ELEMENTS[sym] for sym in comp],
        key=lambda e: e.meam_index,
    )
    a_m = sum(comp[e.symbol] * e.lattice_constant for e in sel)
    n_rep = max(1, round(40.0 / a_m))  # ~4nm box

    library_elements = " ".join(e.symbol for e in pot["elements"])
    active_elements = " ".join(e.symbol for e in sel)

    L = lammps(cmdargs=["-log", "none", "-screen", "none"])
    try:
        L.command("units metal")
        L.command("atom_style atomic")
        L.command("boundary p p p")
        L.command(f"lattice {sel[0].lattice_type} {a_m:.4f}")
        L.command(f"region box block 0 {n_rep} 0 {n_rep} 0 {n_rep}")
        L.command(f"create_box {len(sel)} box")
        L.command("create_atoms 1 box")

        remaining = 1.0
        for i, elem in enumerate(sel[1:], start=2):
            frac = comp[elem.symbol] / remaining
            L.command(f"set type 1 type/fraction {i} {frac:.6f} 12345")
            remaining -= comp[elem.symbol]

        L.command("pair_style meam")
        L.command(f"pair_coeff * * {lib_path} {library_elements} {params_path} {active_elements}")
        for i, elem in enumerate(sel, start=1):
            L.command(f"mass {i} {elem.mass}")

        L.command("minimize 1e-6 1e-8 100 1000")
        E, nu = get_elastic_moduli(L)
        return E, nu
    except Exception:
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

def optimize_nn(lib_path, params_path, opt_spec=None, n_samples=30, results_path=None):
    """Multi-target NN optimization using results.json training data.

    Args:
        lib_path: path to MEAM library file
        params_path: path to MEAM params file
        opt_spec: dict specifying which parameters to optimize
        n_samples: number of parameter space samples
        results_path: path to results.json (default: src/ML/results.json)
    """
    entries = load_training_targets(results_path)
    n_entries = len(entries)

    print(f"\nMulti-Target NN Optimization")
    print(f"Training entries: {n_entries}")
    for name, data in entries.items():
        print(f"  {name}: E={data['E_GPa']:.2f} GPa, nu={data['nu']:.3f}")

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
    print(f"Optimizing {len(names)} parameters: {names}")

    # 1. Sample parameter space
    print(f"\nSampling {n_samples} points...")
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

    # Baseline
    y0 = eval_to_flat(initial_vec)
    X_samples.append(initial_vec)
    y_samples.append(y0)
    print(f"  Baseline: {y0}")

    for i in range(n_samples * 3):
        pert = 1.0 + (np.random.rand(len(initial_vec)) - 0.5) * 0.2
        vec_pert = initial_vec * pert
        y = eval_to_flat(vec_pert)
        # Accept if at least some entries have valid E
        valid = sum(1 for k in range(0, len(y), 2) if abs(y[k]) > 1.0)
        if valid >= n_entries // 2:
            X_samples.append(vec_pert)
            y_samples.append(y)
            print(f"  Sample {len(X_samples)}/{n_samples}: valid_entries={valid}/{n_entries}")
        if len(X_samples) >= n_samples:
            break

    X_train = np.array(X_samples)
    y_train = np.array(y_samples)

    # 2. Train NN Surrogate
    print(f"\nTraining NN surrogate ({len(initial_vec)} → {n_entries * 2})...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(initial_vec),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_entries * 2),
    ])
    model.compile(optimizer="adam", loss="mse")

    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    X_norm = (X_train - X_mean) / X_std
    y_norm = (y_train - y_mean) / y_std

    class StopAtLoss(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs and logs.get("loss", 999) < 5.0:
                print(f"\n  Target loss reached at epoch {epoch}")
                self.model.stop_training = True

    model.fit(X_norm, y_norm, epochs=10000, verbose=0, callbacks=[StopAtLoss()])
    final_loss = model.evaluate(X_norm, y_norm, verbose=0)
    print(f"  Final training loss: {final_loss:.4f}")

    # 3. Inverse optimization through NN
    print("\nOptimizing through NN surrogate...")
    target_norm = (target_vec - y_mean) / y_std
    v_opt = tf.Variable(tf.zeros((1, len(initial_vec)), dtype=tf.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for step in range(3000):
        with tf.GradientTape() as tape:
            pred_norm = model(v_opt)
            loss = tf.reduce_mean(tf.square(pred_norm - target_norm))
        grads = tape.gradient(loss, v_opt)
        optimizer.apply_gradients([(grads, v_opt)])
        if step % 500 == 0:
            print(f"  Step {step}: loss={loss.numpy():.6f}")

    # 4. Validate with LAMMPS
    optimized_vec = v_opt.numpy()[0] * X_std + X_mean
    print("\nValidating optimized parameters with LAMMPS...")
    y_final = eval_to_flat(optimized_vec)

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    entry_names = list(entries.keys())
    for k, name in enumerate(entry_names):
        E_target = entries[name]["E_GPa"]
        nu_target = entries[name]["nu"]
        E_opt = y_final[k * 2]
        nu_opt = y_final[k * 2 + 1]
        E_err = abs(E_opt - E_target) / max(abs(E_target), 1e-6) * 100
        nu_err = abs(nu_opt - nu_target) / max(abs(nu_target), 1e-6) * 100
        print(f"  {name}:")
        print(f"    E:  target={E_target:.2f}  opt={E_opt:.2f}  err={E_err:.1f}%")
        print(f"    nu: target={nu_target:.3f}  opt={nu_opt:.3f}  err={nu_err:.1f}%")
    print("=" * 60)

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

    print(f"\nOptimized files saved to {out_dir}")
    print(f"  Library: {final_lib}")
    print(f"  Params:  {final_par}")

    return final_lib, final_par


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Target NN MEAM Optimizer")
    parser.add_argument("--library", required=True, help="Path to MEAM library file")
    parser.add_argument("--params", required=True, help="Path to MEAM params file")
    parser.add_argument("--samples", type=int, default=30, help="Number of initial samples")
    parser.add_argument("--results", default=None, help="Path to results.json")
    args = parser.parse_args()

    os.environ["TK_SILENT"] = "1"
    optimize_nn(args.library, args.params, n_samples=args.samples, results_path=args.results)
