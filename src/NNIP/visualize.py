#!/usr/bin/env python3
"""Visualization functions for NNIP pipeline diagnostics.

Generates plots from pipeline artifacts:
  - NN training loss curve
  - Optimization trajectory
  - Parity plot (predicted vs target)
  - EOS curves from DFT results

Usage:
    python -m src.NNIP.visualize              # generate all available plots
    python -m src.NNIP.visualize --no-show    # save only, don't display
"""

import json
import os

NNIP_DIR = os.path.dirname(__file__)
PLOTS_DIR = os.path.join(NNIP_DIR, "plots")


def _ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_loss(diag_path=None, output_path=None):
    """Plot NN surrogate training loss curve.

    Args:
        diag_path: path to nn_diagnostics.json
        output_path: output PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if diag_path is None:
        diag_path = os.path.join(NNIP_DIR, "nn_diagnostics.json")
    if output_path is None:
        _ensure_plots_dir()
        output_path = os.path.join(PLOTS_DIR, "nn_training_loss.png")

    diag = _load_json(diag_path)
    if diag is None or "training_loss_history" not in diag:
        print(f"  [VIZ] Skipping training loss plot — no data at {diag_path}")
        return None

    losses = diag["training_loss_history"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(losses, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("NN Surrogate Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [VIZ] Training loss plot -> {output_path}")
    return output_path


def plot_optimization_trajectory(diag_path=None, output_path=None):
    """Plot inverse optimization loss trajectory.

    Args:
        diag_path: path to nn_diagnostics.json
        output_path: output PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if diag_path is None:
        diag_path = os.path.join(NNIP_DIR, "nn_diagnostics.json")
    if output_path is None:
        _ensure_plots_dir()
        output_path = os.path.join(PLOTS_DIR, "nn_opt_trajectory.png")

    diag = _load_json(diag_path)
    if diag is None or "optimization_trajectory" not in diag:
        print(f"  [VIZ] Skipping optimization trajectory plot — no data at {diag_path}")
        return None

    losses = diag["optimization_trajectory"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(losses, linewidth=1.2, color="tab:orange")
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Loss")
    ax.set_title("NN Inverse Optimization Trajectory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [VIZ] Optimization trajectory plot -> {output_path}")
    return output_path


def plot_parity(verification_results, output_path=None):
    """Plot parity diagram: optimized vs target for E and nu.

    Args:
        verification_results: dict {name: {E_opt, E_target, nu_opt, nu_target, ...}}
        output_path: output PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not verification_results:
        print("  [VIZ] Skipping parity plot — no verification results")
        return None

    if output_path is None:
        _ensure_plots_dir()
        output_path = os.path.join(PLOTS_DIR, "parity.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Young's modulus parity
    ax = axes[0]
    E_targets = [v["E_target"] for v in verification_results.values()]
    E_opts = [v["E_opt"] for v in verification_results.values()]
    names = list(verification_results.keys())

    ax.scatter(E_targets, E_opts, s=60, zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (E_targets[i], E_opts[i]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    lims = [min(E_targets + E_opts) * 0.9, max(E_targets + E_opts) * 1.1]
    if lims[0] == lims[1]:
        lims = [lims[0] - 1, lims[1] + 1]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Target E (GPa)")
    ax.set_ylabel("Optimized E (GPa)")
    ax.set_title("Young's Modulus Parity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    # Poisson's ratio parity
    ax = axes[1]
    nu_targets = [v["nu_target"] for v in verification_results.values()]
    nu_opts = [v["nu_opt"] for v in verification_results.values()]

    ax.scatter(nu_targets, nu_opts, s=60, color="tab:orange", zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (nu_targets[i], nu_opts[i]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    lims = [min(nu_targets + nu_opts) * 0.9, max(nu_targets + nu_opts) * 1.1]
    if lims[0] == lims[1]:
        lims = [lims[0] - 0.05, lims[1] + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Target nu")
    ax.set_ylabel("Optimized nu")
    ax.set_title("Poisson's Ratio Parity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [VIZ] Parity plot -> {output_path}")
    return output_path


def plot_eos_curves(dft_path=None, output_path=None):
    """Plot EOS curves (volume vs energy) for each element with EOS data.

    Args:
        dft_path: path to dft_results.json
        output_path: output PNG path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dft_path is None:
        dft_path = os.path.join(NNIP_DIR, "dft_results.json")
    if output_path is None:
        _ensure_plots_dir()
        output_path = os.path.join(PLOTS_DIR, "eos_curves.png")

    data = _load_json(dft_path)
    if data is None:
        print(f"  [VIZ] Skipping EOS plot — no data at {dft_path}")
        return None

    # Collect elements that have eos_data
    eos_elements = {}
    for sym, edata in data.get("elements", {}).items():
        if "eos_data" in edata:
            eos_elements[sym] = edata["eos_data"]

    if not eos_elements:
        print("  [VIZ] Skipping EOS plot — no eos_data in dft_results.json")
        return None

    n = len(eos_elements)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (sym, eos) in enumerate(eos_elements.items()):
        ax = axes[idx // cols][idx % cols]
        vols = eos["volumes"]
        energies = eos["energies"]
        ax.plot(vols, energies, "o-", markersize=5)
        ax.set_xlabel("Volume (A^3)")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"{sym} EOS")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [VIZ] EOS curves plot -> {output_path}")
    return output_path


def plot_all(verification_results=None):
    """Generate all available plots.

    Args:
        verification_results: optional dict from pipeline verification stage
    """
    print("\n[Visualization] Generating plots...")
    plot_training_loss()
    plot_optimization_trajectory()
    if verification_results:
        plot_parity(verification_results)
    plot_eos_curves()
    print("[Visualization] Done.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NNIP Pipeline Visualization")
    parser.add_argument("--diag", default=None, help="Path to nn_diagnostics.json")
    parser.add_argument("--dft", default=None, help="Path to dft_results.json")
    args = parser.parse_args()

    print("\n[Visualization] Standalone mode")
    plot_training_loss(diag_path=args.diag)
    plot_optimization_trajectory(diag_path=args.diag)
    plot_eos_curves(dft_path=args.dft)
    print("[Visualization] Done.\n")
