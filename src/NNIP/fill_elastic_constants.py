r"""Fill missing or unphysical $C_{11}, C_{12}$ entries in
``dft_results.json`` using the recorded bulk modulus and a per-element
Poisson-ratio default.

Why this exists
---------------
``dft_reference.py:_elastic_constants`` only computes $C_{11}, C_{12}$
for cubic (FCC/BCC) elements; HCP and diamond elements (Si, Ti, Zn, Mg)
are skipped, leaving ``None``. A handful of cubic entries also come out
clearly broken (e.g.\ Fe with ``C11 < C12``, indicating mechanical
instability and a miss-converged DFT run). For Stage~4's MEAM seeding
those entries still need *some* value, so this script reconstructs them
from the bulk modulus $B$ and an assumed Poisson ratio under the
cubic-isotropic relation
\\[
   C_{11} = \\frac{3B(1-\\nu)}{1+\\nu},
   \\qquad
   C_{12} = \\frac{3B\\nu}{1+\\nu}.
\\]
The result is internally consistent ($\\,C_{11}+2C_{12} = 3B$) and stays
within the cubic-isotropy assumption already used by Stage~5.

The script is idempotent: running it again is a no-op when every
element already has physical $C_{ij}$.
"""

from __future__ import annotations

import json
import os
import sys

# Per-element Poisson ratio defaults (PBE / literature averages).
# Anything not listed uses ``DEFAULT_NU``.
ASSUMED_NU = {
    "Al": 0.33,
    "Cu": 0.34,
    "Si": 0.22,
    "Ti": 0.36,
    "Zn": 0.25,
    "Cr": 0.21,
    "Fe": 0.29,
    "Mg": 0.29,
    "Mn": 0.30,
    "Mo": 0.31,
    "Ni": 0.31,
    "Co": 0.31,
}
DEFAULT_NU = 0.30


def _cij_from_B(B: float, nu: float) -> tuple[float, float]:
    """Cubic-isotropic $(B,\\nu) \\to (C_{11}, C_{12})$."""
    C11 = 3.0 * B * (1.0 - nu) / (1.0 + nu)
    C12 = 3.0 * B * nu / (1.0 + nu)
    return C11, C12


def _is_unphysical(c11, c12) -> bool:
    """Treat ``None``, mechanically unstable, or implausibly small entries
    as unphysical and replace them on the next ``--force`` pass."""
    if c11 is None or c12 is None:
        return True
    if not (c11 > 0 and c12 >= 0):
        return True
    if c11 < c12:                         # Cauchy / stability violation
        return True
    if c11 < 30.0:                        # smaller than any real metal
        return True
    return False


def fill(path: str, dry_run: bool = False, only_missing: bool = True) -> int:
    """Fill C_ij entries in ``path``. Returns the number of entries updated."""
    with open(path) as f:
        data = json.load(f)
    elements = data.get("elements", {})
    n_updated = 0
    for sym, rec in elements.items():
        B = rec.get("B_GPa")
        c11, c12 = rec.get("C11"), rec.get("C12")
        if B is None or B <= 0:
            print(f"  {sym}: SKIP (no usable B_GPa={B})")
            continue
        needs_fill = (c11 is None or c12 is None) if only_missing else _is_unphysical(c11, c12)
        if not needs_fill:
            continue
        nu = ASSUMED_NU.get(sym, DEFAULT_NU)
        new_c11, new_c12 = _cij_from_B(B, nu)
        old = f"C11={c11}, C12={c12}"
        new = f"C11={new_c11:.3f}, C12={new_c12:.3f}"
        print(f"  {sym}: {old}  ->  {new}  (B={B:.2f}, nu={nu:.2f})")
        if not dry_run:
            rec["C11"] = round(new_c11, 3)
            rec["C12"] = round(new_c12, 3)
        n_updated += 1
    if not dry_run and n_updated:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    print(f"\n  Updated {n_updated} entr{'y' if n_updated == 1 else 'ies'}"
          f"{' (dry run)' if dry_run else ''}.")
    return n_updated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fill missing/unphysical C11,C12 in dft_results.json")
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "dft_results.json")
    parser.add_argument("--path", default=default_path,
                        help=f"Path to dft_results.json (default: {default_path})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--force", action="store_true",
                        help="Also replace unphysical (e.g. C11<C12) entries, "
                             "not only missing ones")
    args = parser.parse_args()
    print(f"Filling C_ij in {args.path}"
          + (" (force mode: also fixes unphysical entries)" if args.force else ""))
    fill(args.path, dry_run=args.dry_run, only_missing=not args.force)
