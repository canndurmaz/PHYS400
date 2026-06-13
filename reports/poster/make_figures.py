#!/usr/bin/env python3
"""Regenerate the A0 poster's result figures in a single, poster-tuned theme.

Unlike the presentation generators (``src/stats/dataset_stats.py`` and
``reports/finalPresentation/generate_nnip_figures.py``), the poster needs plots
that stay legible at arm's length on an A0 sheet and that sit in a colour family
*distinct from* the METU red the poster reserves for headings and result boxes.
So every figure here uses oversized fonts, large markers, and a teal/categorical
(non-red) palette.

Reads (paths resolved relative to the repo root):
    src/ML/results.json            -- LAMMPS elastic database (E, nu, C_ij)
    src/NNIP/nn_diagnostics.json   -- surrogate training-loss history
    src/NNIP/pipeline_summary.json -- Stage-5 verification on held-out alloys

Writes (into reports/poster/figures/):
    E_nu_scatter.png          -- E-nu cloud, coloured by dominant element
    element_frequency.png     -- element coverage bar chart
    dataset_distributions.png -- E and nu histograms over the corpus
    nnip_training_loss.png    -- surrogate Huber-loss curve
    nnip_parity.png           -- NNIP verification parity (E_opt vs E_target, nu)

Run:  python reports/poster/make_figures.py   (inside the `phys` venv)

The small physicality / unphysical filters below intentionally mirror the
canonical ones in dataset_stats.py and generate_nnip_figures.py so the poster's
numbers stay consistent with the rest of the project.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- paths -----------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS_JSON = REPO / "src" / "ML" / "results.json"
NNIP = REPO / "src" / "NNIP"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- canonical constants (mirror dataset_stats.py / generate_nnip_figures.py)
ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Mo", "Ni", "Si", "Ti", "Zn"]
PHYSICAL_E_RANGE = (0.0, 400.0)
PHYSICAL_NU_RANGE = (0.0, 0.5)
NU_FILTER_MAX = 0.48
NAMED = {  # commercial Al alloys to flag in the scatter
    "AL2024_noMg.json": "Al 2024",
    "AL2219_simple.json": "Al 2219",
    "AL5052_simple.json": "Al 5052",
    "AL7075_simple.json": "Al 7075",
}

# --- palette: red is reserved for the poster; plots live in teal + categorical
TEAL = "#0E7C7B"
TEAL_DK = "#08494A"
INK = "#1C1A19"
GREY = "#8A8A8A"
GOOD = "#1B9E77"   # within-tolerance accent (green-teal)
# 12 perceptually distinct, print-safe hues that avoid the METU red.
ELEM_COLORS = {
    "Al": "#1f77b4", "Co": "#ff7f0e", "Cr": "#2ca02c", "Cu": "#9467bd",
    "Fe": "#8c564b", "Mg": "#e377c2", "Mn": "#17becf", "Mo": "#bcbd22",
    "Ni": "#7f7f7f", "Si": "#393b79", "Ti": "#637939", "Zn": "#17375e",
}

# --- poster theme: oversized type for A0 legibility ------------------------
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 18,
    "axes.titlesize": 21,
    "axes.titleweight": "bold",
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 1.3,
    "axes.edgecolor": INK,
    "grid.color": "#C9C9C9",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _physical(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    return ((E > PHYSICAL_E_RANGE[0]) & (E < PHYSICAL_E_RANGE[1])
            & (nu > PHYSICAL_NU_RANGE[0]) & (nu < PHYSICAL_NU_RANGE[1]))


def _unphysical_eu(E, nu) -> bool:
    """NNIP output is unphysical if E<=0 or nu outside [0, 0.48)."""
    try:
        if E is None or nu is None or not (math.isfinite(E) and math.isfinite(nu)):
            return True
    except TypeError:
        return True
    return E <= 0 or nu < 0 or nu >= NU_FILTER_MAX


# ---------------------------------------------------------------------------
def fig_e_nu_scatter(R: dict) -> None:
    keys = list(R.keys())
    E = np.array([R[k]["E_GPa"] for k in keys], dtype=float)
    nu = np.array([R[k]["nu"] for k in keys], dtype=float)
    dom = np.array([
        max(R[k].get("composition", {}).items(), key=lambda kv: kv[1])[0]
        if R[k].get("composition") else "?"
        for k in keys
    ])
    m = _physical(E, nu)
    Ef, nuf, domf = E[m], nu[m], dom[m]
    keysf = np.array(keys)[m]

    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    # Al densest -> draw last so it stays on top; rare dominants stay visible.
    order = [e for e in ELEMENTS if e != "Al"] + ["Al"]
    for el in order:
        sel = domf == el
        if sel.any():
            ax.scatter(Ef[sel], nuf[sel], s=22, alpha=0.45,
                       color=ELEM_COLORS[el], edgecolors="none",
                       label=f"{el}  ({int(sel.sum())})")

    # Flag the named commercial Al alloys with a star (they cluster tightly,
    # so a single legend entry reads better than four overlapping labels).
    star_x = [Ef[np.where(keysf == k)[0][0]] for k in NAMED if (keysf == k).any()]
    star_y = [nuf[np.where(keysf == k)[0][0]] for k in NAMED if (keysf == k).any()]
    if star_x:
        ax.scatter(star_x, star_y, s=360, marker="*", facecolor="#FFD23F",
                   edgecolor=INK, linewidth=1.5, zorder=6)

    ax.set_xlabel(r"Young's modulus  $E$  (GPa)")
    ax.set_ylabel(r"Poisson's ratio  $\nu$")
    ax.set_title(f"Elastic database — {int(m.sum()):,} alloys")
    ax.grid(linestyle=":", alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    if star_x:
        handles.append(Line2D([0], [0], marker="*", color="none",
                              markerfacecolor="#FFD23F", markeredgecolor=INK,
                              markersize=18, label="Commercial Al alloys"))
        labels.append("Commercial Al alloys")
    # Legend below the axes so the scatter itself fills the full width.
    leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                    ncol=4, title="Dominant element (count)",
                    title_fontsize=15, markerscale=2.6, columnspacing=1.2,
                    handletextpad=0.3, borderaxespad=0.2, framealpha=0.95)
    leg.get_title().set_fontweight("bold")
    fig.savefig(FIG_DIR / "E_nu_scatter.png")
    plt.close(fig)
    print(f"[E_nu_scatter] {int(m.sum())} physical alloys")


def fig_element_frequency(R: dict) -> None:
    freq = Counter()
    for v in R.values():
        for el in v.get("composition", {}):
            freq[el] += 1
    counts = [freq.get(el, 0) for el in ELEMENTS]

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    bars = ax.bar(ELEMENTS, counts, color=TEAL, edgecolor=TEAL_DK, linewidth=1.4)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=13, color=INK)
    ax.set_xlabel("Element")
    ax.set_ylabel("Configurations")
    ax.set_title(f"Element coverage — {len(R):,} configurations")
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "element_frequency.png")
    plt.close(fig)


def fig_dataset_distributions(R: dict) -> None:
    E = np.array([v["E_GPa"] for v in R.values()], dtype=float)
    nu = np.array([v["nu"] for v in R.values()], dtype=float)
    m = _physical(E, nu)
    E, nu = E[m], nu[m]

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 6.0))
    ax[0].hist(E, bins=60, color=TEAL, edgecolor=TEAL_DK, linewidth=0.4)
    ax[0].axvspan(60, 75, color=GOOD, alpha=0.18, label="Al-alloy band")
    ax[0].set_xlabel(r"Young's modulus $E$ (GPa)")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"$E$  —  median {np.median(E):.0f} GPa")
    ax[0].legend(fontsize=13)
    ax[1].hist(nu, bins=60, color=TEAL, edgecolor=TEAL_DK, linewidth=0.4)
    ax[1].set_xlabel(r"Poisson's ratio $\nu$")
    ax[1].set_ylabel("count")
    ax[1].set_title(rf"$\nu$  —  median {np.median(nu):.3f}")
    for a in ax:
        a.grid(axis="y", linestyle=":", alpha=0.6)
        a.set_axisbelow(True)
    fig.savefig(FIG_DIR / "dataset_distributions.png")
    plt.close(fig)


def fig_training_loss(diag: dict) -> None:
    history = np.asarray(diag.get("training_loss_history", []), dtype=float)
    if history.size == 0:
        print("[training_loss] no history, skipping")
        return
    steps = np.arange(1, history.size + 1)

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.plot(steps, history, color=TEAL, lw=2.6, solid_capstyle="round")
    ax.scatter([steps[0], steps[-1]], [history[0], history[-1]],
               s=90, color=TEAL_DK, zorder=5)
    ax.annotate(f"start  {history[0]:.2f}", (steps[0], history[0]),
                textcoords="offset points", xytext=(34, 6),
                fontsize=16, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))
    ax.annotate(f"end  {history[-1]:.2f}", (steps[-1], history[-1]),
                textcoords="offset points", xytext=(-40, 34),
                fontsize=16, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Huber loss (scaled $C_{ij}$)")
    ax.set_title(f"Surrogate training — {history.size:,} steps")
    ax.grid(linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    fig.savefig(FIG_DIR / "nnip_training_loss.png")
    plt.close(fig)


def _parity_panel(ax, target, opt, collapsed_t, collapsed_y, *,
                  label, unit, lo, hi, band=0.20):
    """One parity panel: physical points vs 1:1 line + +/-20% band."""
    lim_lo, lim_hi = lo, hi
    xs = np.array([lim_lo, lim_hi])
    ax.fill_between(xs, xs * (1 - band), xs * (1 + band),
                    color=TEAL, alpha=0.13, label=f"$\\pm${int(band*100)}% band")
    ax.plot(xs, xs, color=INK, ls="--", lw=1.8, label="exact (1:1)")
    # collapsed / unphysical outputs (E_opt=0 or nu_opt->0.5), shown honestly
    if len(collapsed_t):
        ax.scatter(collapsed_t, collapsed_y, s=70, marker="x",
                   color=GREY, linewidth=2.0, alpha=0.8,
                   label=f"collapsed ({len(collapsed_t)})")
    n_in = int(np.sum(np.abs(opt - target) <= band * np.abs(target)))
    ax.scatter(target, opt, s=130, color=TEAL, edgecolor=TEAL_DK,
               linewidth=1.2, alpha=0.9, zorder=5,
               label=f"physical, in band: {n_in}/{len(target)}")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"MD reference {label} {unit}")
    ax.set_ylabel(f"NNIP optimized {label} {unit}")
    ax.set_title(f"{label}")
    ax.grid(linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=13.5)


def fig_nnip_parity(summary: dict) -> None:
    V = summary.get("verification", {})
    if not V:
        print("[nnip_parity] no verification block, skipping")
        return
    phys, coll = [], []
    for row in V.values():
        (coll if _unphysical_eu(row.get("E_opt"), row.get("nu_opt")) else phys).append(row)

    Et = np.array([r["E_target"] for r in phys]); Eo = np.array([r["E_opt"] for r in phys])
    nt = np.array([r["nu_target"] for r in phys]); no = np.array([r["nu_opt"] for r in phys])
    cEt = np.array([r["E_target"] for r in coll]); cEo = np.array([r["E_opt"] for r in coll])
    cnt = np.array([r["nu_target"] for r in coll]); cno = np.array([r["nu_opt"] for r in coll])

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.2))
    _parity_panel(axes[0], Et, Eo, cEt, cEo,
                  label=r"Young's modulus $E$", unit="(GPa)", lo=0, hi=480)
    _parity_panel(axes[1], nt, no, cnt, cno,
                  label=r"Poisson's ratio $\nu$", unit="", lo=0.0, hi=0.52)
    fig.suptitle(
        f"NNIP verification on {len(V)} unseen alloys — "
        f"{len(phys)} physically stable",
        fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / "nnip_parity.png")
    plt.close(fig)
    print(f"[nnip_parity] physical={len(phys)} collapsed={len(coll)}")


def main() -> None:
    R = _load(RESULTS_JSON)
    print(f"[make_figures] {len(R)} records from {RESULTS_JSON.name}")
    fig_e_nu_scatter(R)
    fig_element_frequency(R)
    fig_dataset_distributions(R)
    fig_training_loss(_load(NNIP / "nn_diagnostics.json"))
    fig_nnip_parity(_load(NNIP / "pipeline_summary.json"))
    print(f"[make_figures] wrote figures into {FIG_DIR}")


if __name__ == "__main__":
    main()
