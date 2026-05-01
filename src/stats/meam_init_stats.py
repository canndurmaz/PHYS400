"""Stage 4 (MEAM initialization) report assets.

Computes the per-element MEAM initialization parameters (volume
$\\Omega$ and Rose shape parameter $\\alpha$) from the DFT reference and
emits a LaTeX table at ``sections/_auto_meam_init.tex``.
"""

from __future__ import annotations

import json
import math

from _common import DFT_RESULTS_JSON, ensure_dirs, write_latex_table

# 1 GPa -> eV/A^3
_GPA_TO_EVA3 = 1.0 / 160.21766208


def _omega(a0: float, lattice: str) -> float:
    lat = lattice.lower()
    if lat == "fcc":
        return a0**3 / 4.0
    if lat == "bcc":
        return a0**3 / 2.0
    if lat == "hcp":
        return a0**3 * math.sqrt(2.0)
    if lat == "diamond":
        return a0**3 / 8.0
    return a0**3 / 4.0


def main() -> None:
    ensure_dirs()
    with open(DFT_RESULTS_JSON) as f:
        D = json.load(f)

    rows = []
    note_marker = None
    for el, v in D["elements"].items():
        a0 = v.get("a_lat")
        Ec = v.get("E_coh")
        B = v.get("B_GPa")
        lat = v.get("lattice", "fcc")
        if None in (a0, Ec, B):
            continue
        Om = _omega(a0, lat)
        alpha = math.sqrt(9.0 * (B * _GPA_TO_EVA3) * Om / Ec)
        # Heuristic for the failed-EOS case: implausible volume or alpha
        flag = ""
        if Om > 100.0 or alpha > 15.0:
            flag = r"$^\dagger$"
            note_marker = (r"$^\dagger$~EOS scan did not converge; values "
                           r"overwritten by the donor library at merge time.")
        rows.append([
            f"{el}{flag}", lat.upper(),
            f"{a0:.3f}", f"{Ec:.3f}",
            f"{Om:.3f}", f"{alpha:.3f}",
        ])

    write_latex_table(
        out_name="_auto_meam_init.tex",
        rows=rows,
        headers=["Element", "Lattice", r"$a_0$ (\AA)",
                 r"$E_\mathrm{coh}$ (eV)",
                 r"$\Omega$ (\AA$^3$)", r"$\alpha$"],
        caption=(r"MEAM single-element initialization values computed from "
                 r"the DFT reference of Stage~3 by Eq.~\eqref{eq:alpha}."),
        label="tab:meam_init",
        note=note_marker,
    )
    print("[meam_init_stats] done")


if __name__ == "__main__":
    main()
