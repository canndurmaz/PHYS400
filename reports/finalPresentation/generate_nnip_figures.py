#!/usr/bin/env python3
"""Generate NNIP-specific figures for the final presentation.

Reads:
    src/NNIP/nn_diagnostics.json
    src/NNIP/pipeline_summary.json

Produces (under figures/):
    nnip_training_loss.png        -- 10k-step training-loss curve
    nnip_verification_errors.png  -- error histogram on the 30 held-out alloys
                                     with a separate "rejected" bar for the
                                     unphysical NNIP predictions
    nnip_stage_timings.png        -- per-stage wall-clock (log scale)

Run from anywhere; paths are resolved relative to this script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NNIP = REPO / "src" / "NNIP"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NU_FILTER_MAX = 0.48
C_REJECT_MIN = 30.0


def _load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def plot_training_loss(diag: dict, out: Path) -> None:
    history = diag.get("training_loss_history", [])
    if not history:
        print(f"[training_loss] no history in diagnostics, skipping")
        return
    steps = np.arange(1, len(history) + 1)
    loss = np.asarray(history, dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(steps, loss, color="#8C1414", lw=1.1, label="NN surrogate Huber loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Huber loss (scaled $C_{ij}$)")
    ax.set_title(
        f"NN-surrogate training loss "
        f"({len(history):,} steps, $\\delta={diag.get('huber_delta', 1.0)}$)"
    )
    ax.grid(alpha=0.3, linestyle=":")

    # Annotate start and end
    ax.annotate(
        f"start: {loss[0]:.3f}",
        xy=(steps[0], loss[0]),
        xytext=(steps[len(steps) // 12], loss[0] + 0.02),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#444"),
    )
    ax.annotate(
        f"end: {loss[-1]:.3f}",
        xy=(steps[-1], loss[-1]),
        xytext=(steps[-1] * 0.55, loss[-1] - 0.04),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#444"),
    )
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[training_loss] wrote {out}")


def _classify_unphysical_eu(E, nu) -> bool:
    """Apply the same physicality filter used by Stage 2 (LAMMPS) to the
    NNIP-predicted (E, nu) of the held-out alloys, so that the verification
    error histogram is on the same playing field as the training data:
    E > 0, 0 <= nu < 0.48 (singularity guard for 1-2nu).
    """
    if E is None or nu is None:
        return True
    try:
        if not (math.isfinite(E) and math.isfinite(nu)):
            return True
    except TypeError:
        return True
    if E <= 0:
        return True
    if nu < 0 or nu >= NU_FILTER_MAX:
        return True
    return False


def plot_verification_errors(diag: dict, summary: dict, out: Path) -> None:
    # pipeline_summary holds the *current* verification numbers; val_predictions
    # inside nn_diagnostics.json can lag (older run). Prefer the summary block.
    verification = summary.get("verification") or {}
    if not verification:
        print("[verification_errors] no pipeline_summary verification block, skipping")
        return

    physical, rejected = [], []
    for name, row in verification.items():
        row2 = dict(row, name=name)
        if _classify_unphysical_eu(row.get("E_opt"), row.get("nu_opt")):
            rejected.append(row2)
        else:
            physical.append(row2)

    e_errs = np.asarray([p["E_err_pct"] for p in physical], dtype=float)
    nu_errs = np.asarray([p["nu_err_pct"] for p in physical], dtype=float)
    n_total = len(verification)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # --- E error panel ------------------------------------------------------
    ax = axes[0]
    if e_errs.size:
        bins = np.linspace(0, max(100, e_errs.max() * 1.05), 16)
        ax.hist(e_errs, bins=bins, color="#8C1414", alpha=0.85,
                edgecolor="white", label=f"Physical preds (N={e_errs.size})")
    ax.axvline(20.0, color="#444", ls="--", lw=1,
               label="20% target")
    ax.set_xlabel(r"$|E_\mathrm{NNIP} - E_\mathrm{MD}| / E_\mathrm{MD}$ (%)")
    ax.set_ylabel("Count")
    ax.set_title(r"Young's modulus error")
    ax.grid(alpha=0.3, linestyle=":")

    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    # --- nu error panel -----------------------------------------------------
    ax = axes[1]
    if nu_errs.size:
        bins = np.linspace(0, max(100, nu_errs.max() * 1.05), 16)
        ax.hist(nu_errs, bins=bins, color="#1f4e79", alpha=0.85,
                edgecolor="white", label=f"Physical preds (N={nu_errs.size})")
    ax.axvline(20.0, color="#444", ls="--", lw=1,
               label="20% target")
    ax.set_xlabel(r"$|\nu_\mathrm{NNIP} - \nu_\mathrm{MD}| / \nu_\mathrm{MD}$ (%)")
    ax.set_ylabel("Count")
    ax.set_title(r"Poisson ratio error")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)

    n_pass_E = int(np.sum(e_errs <= 20.0))
    n_pass_nu = int(np.sum(nu_errs <= 20.0))
    fig.suptitle(
        f"NNIP Stage-5 verification on {n_total} held-out alloys  "
        f"$\\,\\vert\\,$  E within 20%: {n_pass_E}/{n_total}  "
        f"$\\,\\vert\\,$  "
        rf"$\nu$ within 20%: {n_pass_nu}/{n_total}  "
        f"$\\,\\vert\\,$  unphysical NNIP outputs: {len(rejected)}/{n_total}",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[verification_errors] wrote {out} "
          f"(physical={len(physical)}, rejected={len(rejected)})")


def plot_stage_timings(summary: dict, out: Path) -> None:
    timings = summary.get("stage_timings_sec", {})
    if not timings:
        print("[stage_timings] no stage_timings_sec, skipping")
        return

    display_order = [
        ("element_discovery", "Element discovery"),
        ("dft_reference",     "DFT reference (Stage 3)"),
        ("meam_init",         "MEAM init (Stage 4)"),
        ("nn_optimization",   "NN optimization (Stage 5)"),
        ("verification",      "Verification (Stage 5)"),
        ("visualization",     "Visualization"),
    ]
    labels, values = [], []
    for key, lab in display_order:
        v = float(timings.get(key, 0.0))
        if v <= 0:
            v = 0.0
        labels.append(lab)
        values.append(v)
    total = float(timings.get("total", sum(values))) or 1.0

    # Floor for log scale
    floor = 0.5  # seconds
    plot_vals = [max(v, floor) for v in values]

    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    y = np.arange(len(labels))
    colors = ["#bdbdbd", "#1f4e79", "#444444", "#8C1414", "#a85d5d", "#cccccc"]
    ax.barh(y, plot_vals, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(left=floor * 0.5)
    ax.set_xlabel("Wall-clock (seconds, log scale)")
    ax.grid(alpha=0.3, linestyle=":", axis="x")
    ax.set_title(
        f"Pipeline stage timings --- total wall-clock {total / 3600:.1f} h "
        f"({total:,.0f} s)"
    )

    for yi, v, pv in zip(y, values, plot_vals):
        if v <= 0:
            txt = "skipped (resumed)"
        elif v >= 3600:
            txt = f"{v:,.0f} s  ({v / 3600:.1f} h, {100 * v / total:.1f}%)"
        elif v >= 60:
            txt = f"{v:,.0f} s  ({v / 60:.1f} min, {100 * v / total:.1f}%)"
        else:
            txt = f"{v:.2f} s  ({100 * v / total:.2f}%)"
        ax.text(pv * 1.15, yi, txt, va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[stage_timings] wrote {out}")


def main() -> None:
    diag_path = NNIP / "nn_diagnostics.json"
    summary_path = NNIP / "pipeline_summary.json"
    if not diag_path.exists():
        raise SystemExit(f"missing {diag_path}")
    if not summary_path.exists():
        raise SystemExit(f"missing {summary_path}")

    diag = _load_json(diag_path)
    summary = _load_json(summary_path)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "text.usetex": False,  # keep portable; LaTeX-style escapes handled by mathtext
        "figure.dpi": 160,
    })

    plot_training_loss(diag,    FIG_DIR / "nnip_training_loss.png")
    plot_verification_errors(diag, summary, FIG_DIR / "nnip_verification_errors.png")
    plot_stage_timings(summary, FIG_DIR / "nnip_stage_timings.png")


if __name__ == "__main__":
    main()
