"""Stage 1 + Stage 2 report assets.

Reads ``src/ML/results.json`` (the LAMMPS-evaluated elastic database
produced by Stage~2) and emits:

* ``figures/element_frequency.png`` -- element-coverage bar chart
* ``figures/dataset_distributions.png`` -- $E$/$\\nu$ histograms
* ``figures/E_nu_scatter.png`` -- $E$--$\\nu$ scatter, named alloys highlighted
* ``sections/_auto_md_stats.tex`` -- summary statistics over the
  physicality-filtered dataset
"""

from __future__ import annotations

import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (
    RESULTS_JSON, ensure_dirs, figure_path, write_latex_table,
)

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Mo", "Ni", "Si", "Ti", "Zn"]
NAMED = {
    "AL2024_noMg.json": "Al 2024",
    "AL2219_simple.json": "Al 2219",
    "AL5052_simple.json": "Al 5052",
    "AL7075_simple.json": "Al 7075",
}
PHYSICAL_E_RANGE = (0.0, 400.0)
PHYSICAL_NU_RANGE = (0.0, 0.5)


def _load() -> dict:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _physicality_mask(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    return ((E > PHYSICAL_E_RANGE[0]) & (E < PHYSICAL_E_RANGE[1])
            & (nu > PHYSICAL_NU_RANGE[0]) & (nu < PHYSICAL_NU_RANGE[1]))


def plot_element_frequency(R: dict) -> None:
    freq = Counter()
    for v in R.values():
        for el in v.get("composition", {}):
            freq[el] += 1
    counts = [freq.get(el, 0) for el in ELEMENTS]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(ELEMENTS, counts, color="#3a6ea5", edgecolor="black")
    ax.set_xlabel("Element")
    ax.set_ylabel("Number of configurations")
    ax.set_title(f"Element coverage across {len(R)} configurations")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figure_path("element_frequency.png"), dpi=160)
    plt.close(fig)


def plot_distributions_and_scatter(R: dict) -> dict:
    E = np.array([v["E_GPa"] for v in R.values()], dtype=float)
    nu = np.array([v["nu"] for v in R.values()], dtype=float)
    # Dominant element per record (the one with the largest mass fraction).
    dominants = []
    for v in R.values():
        comp = v.get("composition", {})
        if comp:
            dom = max(comp.items(), key=lambda kv: kv[1])[0]
        else:
            dom = "?"
        dominants.append(dom)
    dominants = np.array(dominants)

    mask = _physicality_mask(E, nu)
    Ef, nuf, df = E[mask], nu[mask], dominants[mask]

    # Histograms
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2))
    axs[0].hist(Ef, bins=60, color="#a85a3a", edgecolor="black", alpha=0.85)
    axs[0].axvspan(60, 75, color="green", alpha=0.15, label="Al-alloy band")
    axs[0].set_xlabel("Young's modulus $E$ (GPa)")
    axs[0].set_ylabel("Count")
    axs[0].set_title(f"LAMMPS dataset: $E$ (N={len(Ef)})")
    axs[0].legend(fontsize=8)
    axs[0].grid(linestyle=":", alpha=0.5)
    axs[1].hist(nuf, bins=60, color="#3a8a4a", edgecolor="black", alpha=0.85)
    axs[1].set_xlabel(r"Poisson ratio $\nu$")
    axs[1].set_ylabel("Count")
    axs[1].set_title(r"LAMMPS dataset: $\nu$")
    axs[1].grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figure_path("dataset_distributions.png"), dpi=160)
    plt.close(fig)

    # Scatter, coloured by dominant element. The fixed ELEMENTS order
    # ensures the colour assignment is stable across reruns regardless of
    # dictionary iteration order.
    cmap = plt.get_cmap("tab20")
    color_map = {el: cmap(i % 20) for i, el in enumerate(ELEMENTS)}
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    # Plot Al last so the dense Al-dominant cloud sits on top and remains
    # visible; rare dominants (e.g., Si) plotted first stay distinguishable.
    plot_order = [el for el in ELEMENTS if el != "Al"] + ["Al"]
    for el in plot_order:
        sel = df == el
        if not sel.any():
            continue
        ax.scatter(Ef[sel], nuf[sel], s=6, alpha=0.4,
                   color=color_map[el], label=f"{el} ({sel.sum()})",
                   edgecolors="none")
    ax.set_xlabel("Young's modulus $E$ (GPa)")
    ax.set_ylabel(r"Poisson ratio $\nu$")
    ax.set_title(r"$E$--$\nu$ scatter, coloured by dominant element")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2,
              title="Dominant (count)", title_fontsize=8,
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(figure_path("E_nu_scatter.png"), dpi=160)
    plt.close(fig)

    return {
        "n_total": len(R),
        "n_physical": int(mask.sum()),
        "E": dict(mean=float(Ef.mean()), median=float(np.median(Ef)),
                  std=float(Ef.std()), min=float(Ef.min()), max=float(Ef.max())),
        "nu": dict(mean=float(nuf.mean()), median=float(np.median(nuf)),
                   std=float(nuf.std()), min=float(nuf.min()), max=float(nuf.max())),
    }


def write_md_stats_table(stats: dict) -> None:
    rows = [
        [
            "$E$ (GPa)", stats["n_physical"],
            f"{stats['E']['mean']:.3f}", f"{stats['E']['median']:.3f}",
            f"{stats['E']['std']:.3f}", f"{stats['E']['min']:.3f}",
            f"{stats['E']['max']:.3f}",
        ],
        [
            r"$\nu$", stats["n_physical"],
            f"{stats['nu']['mean']:.3f}", f"{stats['nu']['median']:.3f}",
            f"{stats['nu']['std']:.3f}", f"{stats['nu']['min']:.3f}",
            f"{stats['nu']['max']:.3f}",
        ],
    ]
    write_latex_table(
        out_name="_auto_md_stats.tex",
        rows=rows,
        headers=["Quantity", "N", "Mean", "Median", "Std.", "Min", "Max"],
        caption=(f"Summary statistics of the LAMMPS-evaluated elastic "
                 f"properties over the physicality-filtered dataset "
                 f"($0<E<400$\\,GPa, $0<\\nu<0.5$). Auto-generated from "
                 f"{stats['n_total']} configuration records."),
        label="tab:md_stats",
    )


def main() -> None:
    ensure_dirs()
    R = _load()
    print(f"[dataset_stats] loaded {len(R)} records from {RESULTS_JSON}")
    plot_element_frequency(R)
    stats = plot_distributions_and_scatter(R)
    write_md_stats_table(stats)
    print(f"[dataset_stats] done (N_physical={stats['n_physical']})")


if __name__ == "__main__":
    main()
