"""Per-prediction uncertainty helpers: deep-ensemble support + KNN OOD flag.

Two distinct signals are exposed to the Flask UI; they capture different
failure modes and should be read together:

1.  **Ensemble σ** — when several ``alloy_model_*.keras`` checkpoints
    are present (deep ensemble; see :mod:`predict_from_model`), the
    spread of predictions across members estimates **epistemic +
    aleatoric** uncertainty for a given composition. With a single
    model this signal is exactly zero, so callers should gate any
    σ-based UI on ``ensemble_size > 1``.
2.  **KNN-distance OOD flag** — for any query composition, the mean
    L2 distance to its ``k`` nearest training compositions tells you
    whether the query is *inside*, on the *edge* of, or *outside* the
    convex hull of sampled space. Thresholds are derived from the
    training set's own self-distances, so they auto-adapt to whatever
    ``results.json`` happens to contain at request time.

Both signals are deliberately cheap (< 1 ms each, brute-force on a
~thousand-row training matrix) and free of TF / Keras state.
"""

from __future__ import annotations

import json
import os
from typing import Mapping

import numpy as np


# With ~hundreds of training points in a 12-dim composition space, k=5
# averages out single-point quirks while staying local enough to flag
# genuine extrapolation. Tune in callers if the training cloud grows.
DEFAULT_K = 5


def composition_to_vector(comp: Mapping[str, float],
                          all_elements: list[str]) -> np.ndarray:
    """Pack a {symbol: fraction} dict into the ordered element vector."""
    v = np.zeros(len(all_elements), dtype=np.float64)
    for el, frac in comp.items():
        if el in all_elements:
            v[all_elements.index(el)] = float(frac)
    return v


def load_training_X(results_path: str, all_elements: list[str]) -> np.ndarray:
    """Return the (N, n_elements) matrix of training compositions.

    No physical-validity filter is applied: every sampled composition
    counts as a neighbour for OOD purposes, even if its elastic constants
    later got dropped by the trainer's ν-filter.
    """
    if not os.path.exists(results_path):
        return np.empty((0, len(all_elements)))
    with open(results_path) as f:
        data = json.load(f)
    rows: list[list[float]] = []
    for rec in data.values():
        comp = rec.get("composition")
        if not isinstance(comp, dict):
            continue
        rows.append([float(comp.get(el, 0.0)) for el in all_elements])
    return (np.asarray(rows, dtype=np.float64)
            if rows else np.empty((0, len(all_elements))))


def knn_distance(query: np.ndarray, X_train: np.ndarray,
                 k: int = DEFAULT_K) -> float:
    """Mean L2 distance from ``query`` to its ``k`` nearest neighbours."""
    if X_train.shape[0] == 0:
        return float("inf")
    diffs = X_train - query[None, :]
    dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
    k_eff = min(k, dists.size)
    # ``partition`` is O(N), no full sort needed.
    nearest = np.partition(dists, k_eff - 1)[:k_eff]
    return float(nearest.mean())


def ood_thresholds(X_train: np.ndarray, k: int = DEFAULT_K) -> dict:
    """Self-distance percentiles used to bucket a query's KNN distance.

    For each training point we compute the mean distance to its k
    nearest *other* training points (excluding self). The 50th and
    95th percentiles of that distribution become the "in" / "edge" /
    "out" cutoffs. Returns zeros if the training set is too small.
    """
    n = X_train.shape[0]
    if n < k + 2:
        return {"p50": 0.0, "p95": 0.0, "n": n, "k": k}
    self_d = np.empty(n)
    for i in range(n):
        diffs = X_train - X_train[i][None, :]
        dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        # Drop the self-distance (which is 0 for i itself) then average
        # over the k smallest of the remainder.
        nearest = np.partition(dists, k)[:k + 1]
        nearest.sort()
        self_d[i] = nearest[1:k + 1].mean()
    return {
        "p50": float(np.percentile(self_d, 50)),
        "p95": float(np.percentile(self_d, 95)),
        "n":   int(n),
        "k":   int(k),
    }


def classify_ood(distance: float, thresholds: dict) -> str:
    """Bucket a KNN distance against training self-distance percentiles.

    Returns one of ``"in"`` (typical), ``"edge"`` (sparser region),
    ``"out"`` (extrapolation), or ``"unknown"`` (training set too small
    to calibrate).
    """
    p95 = thresholds.get("p95", 0.0)
    if not p95:
        return "unknown"
    if distance <= p95:
        return "in"
    if distance <= 2.0 * p95:
        return "edge"
    return "out"
