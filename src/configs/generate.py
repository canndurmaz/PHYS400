#!/usr/bin/env python3
"""Generate random alloy composition configs for MD simulations."""

import argparse
import json
import os
import random
import sys
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MD"))
from element import POTENTIALS

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_valid_subsets():
    valid = []
    for name, pot in POTENTIALS.items():
        symbols = sorted(e.symbol for e in pot["elements"])
        for r in range(2, min(len(symbols), 5) + 1):
            for combo in combinations(symbols, r):
                valid.append(combo)
    return list(set(valid))


def random_composition(elements):
    n = len(elements)
    weights = [random.random() for _ in range(n)]
    total = sum(weights)
    fracs = [w / total for w in weights]

    # If Al is present, force it to be dominant
    if "Al" in elements:
        dominant_idx = elements.index("Al")
    else:
        dominant_idx = fracs.index(max(fracs))

    if fracs[dominant_idx] < 0.5:
        dominant_frac = random.uniform(0.5, 0.95)
        remainder = 1.0 - dominant_frac
        other_weights = [random.random() for _ in range(n - 1)]
        other_total = sum(other_weights)
        other_fracs = [w / other_total * remainder for w in other_weights]
        fracs = []
        j = 0
        for i in range(n):
            if i == dominant_idx:
                fracs.append(dominant_frac)
            else:
                fracs.append(other_fracs[j])
                j += 1

    # Round to 4 decimal places; anything below 1e-4 becomes 0
    fracs = [round(f, 4) for f in fracs]
    fracs = [0.0 if f < 1e-4 else f for f in fracs]

    # Adjust dominant element so the sum is exactly 1.0
    diff = round(1.0 - sum(fracs), 4)
    fracs[dominant_idx] = round(fracs[dominant_idx] + diff, 4)

    return {sym: frac for sym, frac in zip(elements, fracs)}


def comp_to_filename(comp):
    parts = []
    for sym, frac in sorted(comp.items(), key=lambda x: -x[1]):
        frac_str = f"{frac:.4f}".lstrip("0").replace(".", "_")
        if frac >= 1.0:
            frac_str = "1_0000"
        if frac == 0.0:
            frac_str = "_0000"
        parts.append(f"{sym}{frac_str}")
    return "-".join(parts) + ".json"


def main():
    parser = argparse.ArgumentParser(description="Generate random alloy configs")
    parser.add_argument("--samples", type=int, default=1000, help="Number of configs to generate")
    args = parser.parse_args()

    random.seed(42)
    valid_subsets = get_valid_subsets()
    generated = set()
    written = 0

    while written < args.samples:
        subset = random.choice(valid_subsets)
        comp = random_composition(list(subset))
        fname = comp_to_filename(comp)

        if fname in generated:
            continue
        generated.add(fname)

        assert round(sum(comp.values()), 4) == 1.0, f"Sum != 1 for {fname}"

        config = {
            "composition": comp,
            "box_size_m": 4e-09,
            "temperature": 0.1,
            "total_steps": 1000,
            "thermo_interval": 10,
            "dump_interval": 50,
        }
        with open(os.path.join(OUTPUT_DIR, fname), "w") as f:
            json.dump(config, f, indent=4)
        written += 1

    print(f"Generated {written} config files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
