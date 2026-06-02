#!/usr/bin/env python3
"""Pick k representative alloys via k-means medoid + train/val split.

Reduces the full ~7800-entry results.json down to k representatives (default
k=100) selected as the real alloy closest to each k-means cluster centre in
element-fraction composition space. The representatives are then randomly
partitioned into a 70/30 train/validation split (configurable).

Output JSONs share the schema of the input results.json so they drop into
``nn_optimizer.optimize_nn`` via the ``--train`` and ``--val`` flags.

Why medoids and not centroids? K-means centres are synthetic compositions
with no DFT-computed targets. The medoid (the real alloy nearest to a
centre) carries the verified C11/C12/E/nu fields verbatim.
"""

import argparse
import json
import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Ground-state crystal structures for the 12 alloying elements supported by
# the pipeline. Used as a fallback when dft_results.json hasn't computed a
# reference for an element yet (e.g. Co/Ni absent from a partial DFT run);
# without this, the lattice filter would silently treat unknown-lattice
# elements as "compatible with anything" and let mixed-lattice alloys through.
_FALLBACK_LATTICE = {
    "Al": "fcc", "Co": "hcp", "Cr": "bcc", "Cu": "fcc",
    "Fe": "bcc", "Mg": "hcp", "Mn": "bcc", "Mo": "bcc",
    "Ni": "fcc", "Si": "diamond", "Ti": "hcp", "Zn": "hcp",
}


def _load_lattice_map():
    """Element -> natural lattice string. DFT-derived values take precedence
    over the literature fallback; the fallback covers elements that haven't
    been computed yet so the filter can still make a decision."""
    lattice = dict(_FALLBACK_LATTICE)
    dft_path = os.path.join(os.path.dirname(__file__), "dft_results.json")
    if os.path.isfile(dft_path):
        with open(dft_path) as f:
            dft = json.load(f)
        for el, data in dft.get("elements", {}).items():
            if "lattice" in data:
                lattice[el] = data["lattice"]
    return lattice


def _filter_lattice_coherent(results, min_frac=0.05, lattice_map=None):
    """Drop alloys whose elements with fraction ≥ min_frac span >1 lattice type.

    The LAMMPS evaluation in ``nn_optimizer._run_lammps_composition`` builds the
    simulation box from the *dominant* element's lattice only. When the minority
    elements have incompatible natural lattices (e.g. BCC Fe forced onto Al's FCC
    sites), the minimize either fails to converge or settles in a high-strain
    metastable state, producing E=0 / ν=0.5 garbage or C11>1000 GPa blow-ups.
    Filtering these alloys at training-set construction prevents that signal
    from poisoning the surrogate.
    """
    if lattice_map is None:
        lattice_map = _load_lattice_map()
    if not lattice_map:
        print("WARN lattice filter: dft_results.json unavailable, skipping",
              file=sys.stderr)
        return results
    kept = {}
    dropped = 0
    for name, entry in results.items():
        comp = entry.get("composition", {})
        majority = {e for e, f in comp.items() if f >= min_frac}
        lattices = {lattice_map.get(e) for e in majority}
        lattices.discard(None)
        if len(lattices) <= 1:
            kept[name] = entry
        else:
            dropped += 1
    if dropped:
        print(f"Lattice filter: dropped {dropped}/{len(results)} mixed-lattice "
              f"alloys ({len(kept)} retained)", file=sys.stderr)
    return kept


def _composition_matrix(results):
    """Return (names, elements, N x E fraction matrix)."""
    names = list(results.keys())
    elements = sorted({e for v in results.values() for e in v["composition"]})
    elem_idx = {e: i for i, e in enumerate(elements)}
    X = np.zeros((len(names), len(elements)))
    for row, name in enumerate(names):
        for e, frac in results[name]["composition"].items():
            X[row, elem_idx[e]] = frac
    return names, elements, X


def _kmeans(X, k, seed=0, max_iter=300, tol=1e-6):
    """Lloyd's algorithm with k-means++ initialisation, numpy-only.

    Returns (labels, centers). At our scale (~8k points, ~12 dims, k<=200)
    this runs in well under a second — no need for the sklearn dependency.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    for i in range(1, k):
        d2 = np.min(((X[:, None] - centers[:i]) ** 2).sum(axis=2), axis=1)
        total = d2.sum()
        centers[i] = X[rng.integers(n) if total == 0 else rng.choice(n, p=d2 / total)]

    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        d2 = ((X[:, None] - centers) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for c in range(k):
            mask = new_labels == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if np.array_equal(new_labels, labels) or shift < tol:
            labels = new_labels
            break
        labels = new_labels
    return labels, centers


def select_representatives(results, k, seed=0):
    """Return the list of result-keys that are k-means medoids."""
    names, _, X = _composition_matrix(results)
    if k >= len(names):
        return list(names)

    labels, centers = _kmeans(X, k, seed=seed)
    chosen = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if not len(members):
            continue
        d = np.linalg.norm(X[members] - centers[c], axis=1)
        chosen.append(names[members[np.argmin(d)]])
    return chosen


def split_train_val(names, val_frac=0.3, seed=0):
    """Random partition into (train_names, val_names)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_val = int(round(len(names) * val_frac))
    val_idx, train_idx = order[:n_val], order[n_val:]
    return [names[i] for i in train_idx], [names[i] for i in val_idx]


def build_subset(results, k=200, val_frac=0.3, seed=0):
    """End-to-end: lattice filter + k-means medoids + 70/30 split.

    Returns (train, val) dicts.
    """
    filtered = _filter_lattice_coherent(results)
    medoids = select_representatives(filtered, k, seed)
    train_names, val_names = split_train_val(medoids, val_frac, seed)
    train = {n: filtered[n] for n in train_names}
    val = {n: filtered[n] for n in val_names}
    return train, val


def main():
    default_input = os.path.join(project_root, "src", "ML", "results.json")
    default_train = os.path.join(project_root, "src", "ML", "results_train.json")
    default_val = os.path.join(project_root, "src", "ML", "results_val.json")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=default_input,
                    help=f"Full results JSON (default: {default_input})")
    ap.add_argument("--train-out", default=default_train,
                    help=f"Training subset output (default: {default_train})")
    ap.add_argument("--val-out", default=default_val,
                    help=f"Validation subset output (default: {default_val})")
    ap.add_argument("-k", type=int, default=200,
                    help="Total representatives selected before split (default 200)")
    ap.add_argument("--val-frac", type=float, default=0.3,
                    help="Fraction of representatives held out for validation (default 0.3)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for k-means + split (default 0)")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    train, val = build_subset(data, args.k, args.val_frac, args.seed)

    with open(args.train_out, "w") as f:
        json.dump(train, f, indent=2)
    with open(args.val_out, "w") as f:
        json.dump(val, f, indent=2)

    print(f"From {len(data)} entries -> k={args.k} medoids "
          f"-> {len(train)} train ({args.train_out}) "
          f"+ {len(val)} val ({args.val_out})")


if __name__ == "__main__":
    main()
