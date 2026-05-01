r"""One-shot: fill ``C11_GPa`` and ``C12_GPa`` in
``src/ML/results.json`` for every entry that has $E_\\mathrm{GPa}$ and
$\\nu$ but no $C_{ij}$ yet.

Use the cubic-isotropic algebra
\\[
   C_{11} = \\frac{E\\,(1-\\nu)}{(1-2\\nu)(1+\\nu)},
   \\qquad
   C_{12} = \\frac{E\\,\\nu}{(1-2\\nu)(1+\\nu)}
\\]
(bijective for $\\nu < 0.5$). Entries with non-physical $E$/$\\nu$ or
$\\nu \\geq 0.49$ (singularity at $\\nu = 0.5$) are skipped, and any
entry that already carries ``C11_GPa``/``C12_GPa`` is left untouched
so the script is safe to re-run.
"""

from __future__ import annotations

import json
import os
import sys

RESULTS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "ML", "results.json"))

# Strict upper bound to keep (1-2nu) bounded away from zero.
NU_FILTER_MAX = 0.49


def _cij_from_e_nu(E: float, nu: float) -> tuple[float, float]:
    s = E / ((1.0 - 2.0 * nu) * (1.0 + nu))
    return s * (1.0 - nu), s * nu


def fill(path: str = RESULTS_PATH, dry_run: bool = False) -> dict:
    with open(path) as f:
        results = json.load(f)

    n_total = len(results)
    n_filled = 0
    n_already = 0
    n_skipped = 0
    skipped_examples = []

    for name, data in results.items():
        if "C11_GPa" in data and "C12_GPa" in data:
            n_already += 1
            continue
        E = float(data.get("E_GPa", 0.0))
        nu = float(data.get("nu", 0.0))
        if E <= 0 or nu <= 0 or nu >= NU_FILTER_MAX:
            n_skipped += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(f"{name} (E={E}, nu={nu})")
            continue
        C11, C12 = _cij_from_e_nu(E, nu)
        if not (C11 > 0 and C12 >= 0 and C11 >= C12):
            # mechanical instability — skip rather than store nonsense
            n_skipped += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(
                    f"{name} (gave C11={C11:.1f} < C12={C12:.1f})")
            continue
        if not dry_run:
            data["C11_GPa"] = round(C11, 2)
            data["C12_GPa"] = round(C12, 2)
        n_filled += 1

    if not dry_run and n_filled > 0:
        with open(path, "w") as f:
            json.dump(results, f, indent=4)

    print(f"Path: {path}")
    print(f"Total entries:           {n_total}")
    print(f"Filled (E,nu -> C11,C12): {n_filled}")
    print(f"Already had C_ij:        {n_already}")
    print(f"Skipped (non-physical):  {n_skipped}")
    if skipped_examples:
        print(f"  examples: {skipped_examples}")
    if dry_run:
        print("Dry run: no file was modified.")
    return {"total": n_total, "filled": n_filled,
            "already": n_already, "skipped": n_skipped}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=RESULTS_PATH,
                        help=f"Path to results.json (default: {RESULTS_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args()
    fill(args.path, dry_run=args.dry_run)
