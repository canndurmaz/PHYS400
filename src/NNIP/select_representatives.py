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


def build_subset(results, k=100, val_frac=0.3, seed=0):
    """End-to-end: k-means medoids + 70/30 split. Returns (train, val) dicts."""
    medoids = select_representatives(results, k, seed)
    train_names, val_names = split_train_val(medoids, val_frac, seed)
    train = {n: results[n] for n in train_names}
    val = {n: results[n] for n in val_names}
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
    ap.add_argument("-k", type=int, default=100,
                    help="Total representatives selected before split (default 100)")
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
