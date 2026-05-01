"""Stage 3 (DFT) report assets.

Reads ``src/NNIP/dft_results.json`` and emits:

* ``figures/formation_energy_heatmap.png`` -- symmetric heatmap of the
  binary $E_\\mathrm{form}(i,j)$ values.
* ``sections/_auto_dft_elements.tex`` -- elemental reference table
  (lattice, $a_0$, $E_\\mathrm{coh}$, $B$, $C_{11}$, $C_{12}$).
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import (
    DFT_RESULTS_JSON, ensure_dirs, figure_path, write_latex_table,
)


def _load() -> dict:
    with open(DFT_RESULTS_JSON) as f:
        return json.load(f)


def plot_formation_heatmap(D: dict) -> None:
    els = list(D["elements"].keys())
    n = len(els)
    M = np.full((n, n), np.nan)
    for k, v in D["binary_pairs"].items():
        a, b = k.split("-")
        if a in els and b in els:
            i, j = els.index(a), els.index(b)
            M[i, j] = v["E_form"]
            M[j, i] = v["E_form"]
    # Larger figure + smaller in-cell text so the .3f labels fit in every
    # cell even when the figure is rendered at a modest \includegraphics
    # width in the report.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(els)
    ax.set_yticks(range(n)); ax.set_yticklabels(els)
    for i in range(n):
        for j in range(n):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if abs(M[i, j]) < vmax * 0.55 else "white")
    ax.set_title(r"DFT binary formation energy $E_\mathrm{form}$ (eV/atom)")
    fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
    fig.tight_layout()
    fig.savefig(figure_path("formation_energy_heatmap.png"), dpi=200)
    plt.close(fig)


def write_elements_table(D: dict) -> None:
    rows = []
    note = None
    for el, v in D["elements"].items():
        a0 = v.get("a_lat")
        Ec = v.get("E_coh")
        B = v.get("B_GPa")
        lat = v.get("lattice", "?").upper()
        C11 = v.get("C11")
        C12 = v.get("C12")
        # Flag elements whose EOS scan did not converge (empty eos_data)
        eos = v.get("eos_data") or {}
        eos_empty = not eos.get("volumes") or not eos.get("energies")
        flag = r"$^\dagger$" if eos_empty else ""
        if eos_empty:
            note = (r"$^\dagger$~EOS scan did not converge; tabulated values "
                    r"retained, donor-library entries used at merge time.")
        rows.append([
            f"{el}{flag}", lat,
            f"{a0:.3f}" if a0 is not None else "---",
            f"{Ec:.3f}" if Ec is not None else "---",
            f"{B:.3f}" if B is not None else "---",
            f"{C11:.3f}" if C11 is not None else "---",
            f"{C12:.3f}" if C12 is not None else "---",
        ])
    write_latex_table(
        out_name="_auto_dft_elements.tex",
        rows=rows,
        headers=["Element", "Lattice", r"$a_0$ (\AA)",
                 r"$E_\mathrm{coh}$", r"$B$",
                 r"$C_{11}$", r"$C_{12}$"],
        caption=(r"Elemental reference data extracted from DFT. "
                 r"$E_\mathrm{coh}$ is in eV/atom and $B,C_{ij}$ in GPa."),
        label="tab:dft_elements",
        note=note,
    )


def main() -> None:
    ensure_dirs()
    D = _load()
    print(f"[dft_stats] loaded {len(D['elements'])} elements, "
          f"{len(D['binary_pairs'])} binary pairs from {DFT_RESULTS_JSON}")
    plot_formation_heatmap(D)
    write_elements_table(D)
    print("[dft_stats] done")


if __name__ == "__main__":
    main()
