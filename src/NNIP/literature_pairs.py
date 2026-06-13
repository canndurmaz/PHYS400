#!/usr/bin/env python3
"""Extract literature-known binary cross-term values from previous MEAM files.

Walks a list of previous `.meam` + `library_*.meam` pairs, parses each one's
element ordering, and collects `(Ec, re, alpha)` for every binary pair (i,j).
Multi-source pairs are resolved by `_resolve_conflict()` below.

The output `literature_pairs.json` is the single source of truth for which
binary pairs the optimizer should treat as *frozen* (already known from prior
work) vs. *DFT-seeded and free to optimize*.

CLI:
    python -m src.NNIP.literature_pairs           # regenerate the JSON
    python -m src.NNIP.literature_pairs --print   # also dump the table
"""

import itertools
import json
import os
import re
import sys
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Canonical 12-element ordering (matches CLAUDE.md invariant).
ELEMS = ["Al", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Mo", "Ni", "Si", "Ti", "Zn"]

# Default list of previous (.meam, library_*.meam) pairs to draw from.
# Order is informational only — conflicts are resolved by `_resolve_conflict()`,
# not by file order.
DEFAULT_PREVIOUS = [
    ("EAM/AlMgZn.meam",            "EAM/library_AlMgZn.meam"),
    ("EAM/AlSiCuMgFE.meam",        "EAM/library_AlSiCuMgFE.meam"),
    ("EAM/AlCuSiTiZn.meam",        "EAM/library_AlCuSiTiZn.meam"),
    ("EAM/CuMo.meam",              "EAM/library_CuMo.meam"),
    ("EAM/FeMnNiTiCuCrCoAl.meam",  "EAM/library_FeMnNiTiCuCrCoAl.meam"),
]

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "literature_pairs.json")


# ── Parsers ──────────────────────────────────────────────────────────────────

_HEADER_PAT = re.compile(r"\s*'([A-Z][a-z]?)'\s+'\w+'")
_PARAM_PAT = re.compile(
    r"^\s*(Ec|re|alpha)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([-+\d.eE]+)"
)


def _parse_lib_order(libpath):
    """Return the element ordering encoded by a `library_*.meam` file."""
    order = []
    with open(libpath) as f:
        for line in f:
            m = _HEADER_PAT.match(line)
            if m:
                order.append(m.group(1))
    return order


def _parse_pair_params(parpath):
    """Extract {(name, i, j): value} for every Ec/re/alpha binary entry."""
    out = {}
    with open(parpath) as f:
        for line in f:
            m = _PARAM_PAT.match(line)
            if not m:
                continue
            name = m.group(1)
            i, j = int(m.group(2)), int(m.group(3))
            if i == j:
                continue  # self-term, not a cross-term
            a, b = sorted([i, j])
            out[(name, a, b)] = float(m.group(4))
    return out


# ── Conflict resolution ──────────────────────────────────────────────────────

def _resolve_conflict(sources, name):
    """Reduce a list of disagreeing source values for one parameter to one.

    Args:
        sources: list of dicts like
            [{"file": "AlCuSiTiZn.meam", "Ec": 3.5058, "re": 2.53, "alpha": 4.65},
             {"file": "FeMn...meam",     "Ec": 3.5058, "re": 2.3984, "alpha": 5.0}]
        name: which parameter we're resolving — "Ec", "re", or "alpha".

    Returns:
        (resolved_value, resolution_method) where resolution_method is a short
        tag for the diagnostic log ("single", "agree", "mean", ...).

    Edit policy below — this is the place to change how disagreements collapse.
    """
    values = [s[name] for s in sources]
    if len(values) == 1:
        return values[0], "single"
    mean = sum(values) / len(values)
    spread = max(values) - min(values)
    if mean != 0 and spread / abs(mean) < 0.01:
        return values[0], "agree"
    return mean, "mean"


# ── Extraction ───────────────────────────────────────────────────────────────

def extract(previous_files=None, project_root_dir=None):
    """Collect literature-known pair values across all previous .meam files.

    Returns a dict shaped like:
        {
          "Al-Cu": {
              "Ec":    {"value": 3.5058, "method": "agree",  "sources": [...]},
              "re":    {"value": 2.4642, "method": "mean",   "sources": [...]},
              "alpha": {"value": 4.825,  "method": "mean",   "sources": [...]},
          },
          ...
        }

    Only pairs (sym_a, sym_b) where BOTH symbols are in ELEMS appear.
    Pairs with at least one of {Ec, re, alpha} present are returned.
    """
    if previous_files is None:
        previous_files = DEFAULT_PREVIOUS
    if project_root_dir is None:
        project_root_dir = project_root

    # raw_pairs[(sym_a, sym_b)] is a list of source dicts as built above.
    raw_pairs = defaultdict(list)
    for par_rel, lib_rel in previous_files:
        par_path = os.path.join(project_root_dir, par_rel)
        lib_path = os.path.join(project_root_dir, lib_rel)
        if not (os.path.exists(par_path) and os.path.exists(lib_path)):
            continue
        order = _parse_lib_order(lib_path)
        # group by (i,j) inside this file -> {Ec, re, alpha}
        per_pair = defaultdict(dict)
        for (name, i, j), val in _parse_pair_params(par_path).items():
            if i - 1 >= len(order) or j - 1 >= len(order):
                continue
            per_pair[(i, j)][name] = val
        for (i, j), names_dict in per_pair.items():
            sym_a, sym_b = sorted([order[i - 1], order[j - 1]])
            if sym_a not in ELEMS or sym_b not in ELEMS:
                continue
            raw_pairs[(sym_a, sym_b)].append({
                "file": os.path.basename(par_path),
                "Ec":    names_dict.get("Ec"),
                "re":    names_dict.get("re"),
                "alpha": names_dict.get("alpha"),
            })

    # Resolve and emit the final table.
    out = {}
    for pair, sources in raw_pairs.items():
        pair_key = f"{pair[0]}-{pair[1]}"
        entry = {}
        for name in ("Ec", "re", "alpha"):
            has_value = [s for s in sources if s[name] is not None]
            if not has_value:
                continue
            value, method = _resolve_conflict(has_value, name)
            entry[name] = {
                "value": round(value, 6),
                "method": method,
                "sources": [s["file"] for s in has_value],
            }
        if entry:
            out[pair_key] = entry
    return out


def known_pair_set(literature_table):
    """Return {(sym_a, sym_b)} of pairs that should NOT be optimized.

    Used by nn_optimizer.py to filter the default opt_spec. We freeze a pair
    only if all three of (Ec, re, alpha) are literature-known — otherwise the
    optimizer would be solving an under-constrained sub-problem for that pair.
    """
    frozen = set()
    for pair_key, entry in literature_table.items():
        if all(k in entry for k in ("Ec", "re", "alpha")):
            a, b = pair_key.split("-")
            frozen.add(tuple(sorted([a, b])))
    return frozen


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="Extract literature pair table from previous .meam files")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--print", action="store_true",
                   help="Also print a human-readable summary table")
    args = p.parse_args()

    table = extract()
    with open(args.output, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)

    n_full = sum(1 for v in table.values() if all(k in v for k in ("Ec", "re", "alpha")))
    n_partial = len(table) - n_full
    all_pairs = list(itertools.combinations(ELEMS, 2))
    n_total = len(all_pairs)
    print(f"Wrote {args.output}")
    print(f"  Pairs fully covered (Ec, re, alpha): {n_full} / {n_total}")
    print(f"  Pairs partially covered:             {n_partial} / {n_total}")
    print(f"  Pairs missing (DFT-seeded):          {n_total - len(table)} / {n_total}")

    if args.print:
        print("\n=== LITERATURE PAIR TABLE ===")
        for pair_key in sorted(table.keys()):
            entry = table[pair_key]
            parts = []
            for name in ("Ec", "re", "alpha"):
                if name in entry:
                    parts.append(f"{name}={entry[name]['value']:.4g}({entry[name]['method']})")
            srcs = sorted({s for v in entry.values() for s in v["sources"]})
            print(f"  {pair_key:8s}  {'  '.join(parts):60s}  <- {','.join(srcs)}")


if __name__ == "__main__":
    main()
