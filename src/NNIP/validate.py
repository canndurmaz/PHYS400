#!/usr/bin/env python3
"""Validate the trained DeePMD model against held-out DFT data.

Produces:
- Energy parity plot (predicted vs DFT)
- Force component parity plot
- MAE and RMSE statistics
- Per-element breakdown
"""

import json
import os
import sys

import numpy as np

PROJECT_DIR = "/home/kenobi/Workspaces/PHYS400"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model.pb")
VALIDATION_DIR = os.path.join(PROJECT_DIR, "data", "validation", "dft_results")
RESULTS_DIR = os.path.join(PROJECT_DIR, "models", "validation_results")


def load_validation_data():
    """Load DFT results from validation set."""
    from ase.io import read
    from glob import glob

    files = sorted(glob(os.path.join(VALIDATION_DIR, "*.xyz")))
    if not files:
        print(f"ERROR: No validation data found in {VALIDATION_DIR}")
        sys.exit(1)

    configs = []
    for f in files:
        atoms = read(f, format="extxyz")
        configs.append(atoms)

    print(f"Loaded {len(configs)} validation configurations")
    return configs


def evaluate_model(configs):
    """Run DeePMD model on validation configs and compare with DFT."""
    try:
        from deepmd.infer import DeepPot
    except ImportError:
        print("ERROR: deepmd-kit not installed. Run install_deepmd.sh first.")
        sys.exit(1)

    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    dp = DeepPot(MODEL_PATH)
    type_map = dp.get_type_map()
    print(f"Model type map: {type_map}")

    results = {
        "dft_energies": [],
        "nn_energies": [],
        "dft_forces": [],
        "nn_forces": [],
        "n_atoms": [],
        "config_types": [],
    }

    for atoms in configs:
        n = len(atoms)
        symbols = atoms.get_chemical_symbols()
        types = np.array([type_map.index(s) for s in symbols])
        coords = atoms.get_positions().reshape(1, -1)
        cell = atoms.get_cell().array.reshape(1, -1)

        e, f, v = dp.eval(coords, cell, types)

        dft_e = atoms.info.get("dft_energy", atoms.info.get("energy", None))
        dft_f = atoms.arrays.get("dft_forces", atoms.arrays.get("forces", None))

        if dft_e is None or dft_f is None:
            continue

        results["dft_energies"].append(dft_e / n)  # per atom
        results["nn_energies"].append(e[0, 0] / n)
        results["dft_forces"].append(dft_f.flatten())
        results["nn_forces"].append(f[0].flatten())
        results["n_atoms"].append(n)
        results["config_types"].append(
            atoms.info.get("config_type", "unknown")
        )

    return results


def compute_metrics(results):
    """Compute MAE, RMSE for energy and forces."""
    dft_e = np.array(results["dft_energies"])
    nn_e = np.array(results["nn_energies"])
    e_errors = nn_e - dft_e

    dft_f = np.concatenate(results["dft_forces"])
    nn_f = np.concatenate(results["nn_forces"])
    f_errors = nn_f - dft_f

    metrics = {
        "energy": {
            "MAE_meV_per_atom": float(np.mean(np.abs(e_errors)) * 1000),
            "RMSE_meV_per_atom": float(np.sqrt(np.mean(e_errors**2)) * 1000),
            "max_error_meV_per_atom": float(np.max(np.abs(e_errors)) * 1000),
        },
        "forces": {
            "MAE_meV_per_A": float(np.mean(np.abs(f_errors)) * 1000),
            "RMSE_meV_per_A": float(np.sqrt(np.mean(f_errors**2)) * 1000),
            "max_error_meV_per_A": float(np.max(np.abs(f_errors)) * 1000),
        },
        "n_configs": len(dft_e),
        "n_force_components": len(dft_f),
    }

    return metrics


def plot_parity(results, metrics):
    """Create parity plots for energy and forces."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dft_e = np.array(results["dft_energies"])
    nn_e = np.array(results["nn_energies"])

    # Energy parity plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(dft_e, nn_e, s=10, alpha=0.6)
    lims = [min(dft_e.min(), nn_e.min()), max(dft_e.max(), nn_e.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT Energy (eV/atom)")
    ax.set_ylabel("NN Energy (eV/atom)")
    ax.set_title(f"Energy Parity — MAE: {metrics['energy']['MAE_meV_per_atom']:.2f} meV/atom")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "energy_parity.png"), dpi=150)
    plt.close(fig)

    # Force parity plot
    dft_f = np.concatenate(results["dft_forces"])
    nn_f = np.concatenate(results["nn_forces"])

    fig, ax = plt.subplots(figsize=(6, 6))
    # Subsample if too many points
    n_pts = len(dft_f)
    if n_pts > 10000:
        idx = np.random.choice(n_pts, 10000, replace=False)
        dft_f_plot, nn_f_plot = dft_f[idx], nn_f[idx]
    else:
        dft_f_plot, nn_f_plot = dft_f, nn_f

    ax.scatter(dft_f_plot, nn_f_plot, s=2, alpha=0.3)
    lims = [min(dft_f.min(), nn_f.min()), max(dft_f.max(), nn_f.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("DFT Force (eV/Å)")
    ax.set_ylabel("NN Force (eV/Å)")
    ax.set_title(f"Force Parity — MAE: {metrics['forces']['MAE_meV_per_A']:.2f} meV/Å")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "force_parity.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {RESULTS_DIR}")


def main():
    print("=" * 60)
    print("DeePMD Model Validation")
    print("=" * 60)

    configs = load_validation_data()
    results = evaluate_model(configs)

    if not results["dft_energies"]:
        print("ERROR: No valid comparisons found")
        sys.exit(1)

    metrics = compute_metrics(results)

    print(f"\n--- Metrics ---")
    print(f"Energy MAE:  {metrics['energy']['MAE_meV_per_atom']:.2f} meV/atom "
          f"(target: < 5 meV/atom)")
    print(f"Energy RMSE: {metrics['energy']['RMSE_meV_per_atom']:.2f} meV/atom")
    print(f"Force MAE:   {metrics['forces']['MAE_meV_per_A']:.2f} meV/Å "
          f"(target: < 100 meV/Å)")
    print(f"Force RMSE:  {metrics['forces']['RMSE_meV_per_A']:.2f} meV/Å")
    print(f"Configs:     {metrics['n_configs']}")
    print(f"Force comps: {metrics['n_force_components']}")

    # Check targets
    e_pass = metrics["energy"]["MAE_meV_per_atom"] < 5.0
    f_pass = metrics["forces"]["MAE_meV_per_A"] < 100.0
    print(f"\nEnergy target: {'PASS' if e_pass else 'FAIL'}")
    print(f"Force target:  {'PASS' if f_pass else 'FAIL'}")

    plot_parity(results, metrics)

    # Save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
