"""Active-learning utilities for the NN MEAM optimizer.

Active learning replaces blind uniform-random Phase-1 sampling with a
seed-and-iterate loop:

  1. Seed phase   — generate a Sobol/LHS-spread bootstrap of `seed_size`
                    parameter vectors and evaluate them with LAMMPS.
  2. Iteration    — until `budget` accepted samples are collected:
        a. Train a small ensemble of NN surrogates on the current samples.
        b. Score a large candidate pool by the ensemble's predictive
           variance (high variance = the ensemble disagrees = high
           information gain from running LAMMPS here).
        c. Pick a diverse top-B batch via greedy max-min on the
           high-variance subset, so parallel LAMMPS workers don't waste
           compute on near-duplicate queries.
        d. Evaluate the batch with the existing parallel worker pool;
           append accepted samples to the checkpoint.

The bottleneck of `nn_optimizer.optimize_nn` is LAMMPS evaluation (seconds
to minutes per parameter vector). The few extra seconds per iteration spent
training a 5-member ensemble and scoring a 500-vector pool are essentially
free compared to one extra LAMMPS call, so the budget shrinks ~3x for
equivalent surrogate quality.

This module exposes pure functions; control flow lives in
`nn_optimizer._phase1_sample_active`. The split keeps `nn_optimizer.py`
focused on orchestration and makes the active-learning pieces unit-testable
in isolation.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

logger = logging.getLogger("nn_optimizer")


# ── Candidate generation ─────────────────────────────────────────────────────

# Default per-dimension perturbation magnitude (matches the historical
# nn_optimizer setting of `pert = 1.0 + (rand - 0.5) * 0.2` → ±10 %).
PERT_RANGE = 0.2


def _scale_unit_box(unit_samples: np.ndarray, initial_vec: np.ndarray,
                    pert_range: float = PERT_RANGE) -> np.ndarray:
    """Map [-1, 1]^d candidates onto physical parameter vectors.

    Each unit-box sample ``u`` becomes ``initial_vec * (1 + (pert_range/2) * u)``,
    so a ``pert_range`` of 0.2 reproduces the ±10 % box used by the legacy
    random sampler.
    """
    return initial_vec[None, :] * (1.0 + 0.5 * pert_range * unit_samples)


def generate_candidates(initial_vec: np.ndarray, mode: str, n: int,
                        seed: int = 0,
                        pert_range: float = PERT_RANGE) -> np.ndarray:
    """Generate ``n`` candidate parameter vectors around ``initial_vec``.

    Args:
        initial_vec: baseline MEAM parameter vector (1-D).
        mode: one of ``"random"``, ``"lhs"``, ``"sobol"``.
        n: number of candidates to return.
        seed: RNG / sampler seed (passed straight through to numpy / scipy).
        pert_range: fractional half-width of the perturbation box
                    (default ``PERT_RANGE`` = 0.2, i.e. ±10 %).

    Returns:
        array of shape ``(n, d)``.
    """
    d = len(initial_vec)
    if mode == "random":
        rng = np.random.default_rng(seed)
        unit = rng.uniform(-1.0, 1.0, size=(n, d))
    elif mode == "lhs":
        from scipy.stats import qmc
        unit = qmc.LatinHypercube(d=d, seed=seed).random(n) * 2.0 - 1.0
    elif mode == "sobol":
        from scipy.stats import qmc
        unit = qmc.Sobol(d=d, scramble=True, seed=seed).random(n) * 2.0 - 1.0
    else:
        raise ValueError(f"Unknown sampling mode: {mode!r}")
    return _scale_unit_box(unit, initial_vec, pert_range)


# ── Ensemble training (acquisition surrogate, not the production NN) ─────────

def train_ensemble(X: np.ndarray, y: np.ndarray, n_models: int = 5,
                   hidden: Sequence[int] = (20, 20, 10),
                   epochs: int = 80, lr: float = 1e-3,
                   seed: int = 0) -> List["tf.keras.Model"]:
    """Train a small ensemble of MLPs on (X, y) for acquisition scoring.

    Each member uses the same architecture but a different random init.
    The ensemble is intentionally cheap — it exists only to produce a
    predictive-variance signal for batch selection, not to replace the
    production NN. Returns a list of compiled, fitted Keras models.

    Inputs are normalised internally; callers pass raw (X, y) and we
    handle the scaling so the per-iteration code stays simple.
    """
    import tensorflow as tf

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    n_in = X.shape[1]
    n_out = y.shape[1]

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    X_std[X_std == 0] = 1.0
    y_std[y_std == 0] = 1.0
    Xn = (X - X_mean) / X_std
    yn = (y - y_mean) / y_std
    batch_size = max(1, min(16, len(X)))

    models: List["tf.keras.Model"] = []
    for k in range(n_models):
        tf.keras.utils.set_random_seed(int(seed) + k)
        m = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_in,)),
            *[tf.keras.layers.Dense(h, activation="tanh") for h in hidden],
            tf.keras.layers.Dense(n_out),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
        m.fit(Xn, yn, epochs=epochs, batch_size=batch_size, verbose=0)
        # Stash normalisation so `ensemble_variance` can predict on raw X.
        m._nnal_norm = (X_mean, X_std, y_mean, y_std)
        models.append(m)
    return models


# ── Acquisition function ─────────────────────────────────────────────────────

def ensemble_variance(models: List["tf.keras.Model"],
                      X_pool: np.ndarray) -> np.ndarray:
    """Per-candidate average predictive variance across the ensemble.

    Returns an array of shape ``(n_pool,)``. Higher values mark candidates
    where the ensemble disagrees most — those are the LAMMPS calls most
    likely to teach the surrogate something it doesn't already know.
    """
    if not models:
        raise ValueError("ensemble is empty")
    X_pool = np.asarray(X_pool, dtype=np.float32)
    preds_norm = []
    for m in models:
        X_mean, X_std, y_mean, y_std = m._nnal_norm
        Xn = (X_pool - X_mean) / X_std
        yn = m(Xn, training=False).numpy()
        preds_norm.append(yn * y_std + y_mean)
    preds = np.stack(preds_norm, axis=0)  # (K, n_pool, n_out)
    return preds.var(axis=0).mean(axis=-1)


# ── Diversity-aware batch selection ──────────────────────────────────────────

def select_diverse_batch(X_pool: np.ndarray, scores: np.ndarray,
                          batch_size: int,
                          oversample: int = 10) -> List[int]:
    """Pick ``batch_size`` indices from the pool maximising score and spread.

    Strategy: restrict to the top ``batch_size * oversample`` candidates by
    acquisition score (these are the "interesting" ones), then greedily
    pick the one that maximises a normalised mix of (score) and
    (distance from already-picked) — a poor man's k-medoids on the
    high-variance subset. The diversity penalty keeps parallel LAMMPS
    workers from chasing duplicate queries.

    Returns a list of indices into ``X_pool`` of length ``min(batch_size,
    len(X_pool))``.
    """
    X_pool = np.asarray(X_pool, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pool, d = X_pool.shape
    batch_size = min(batch_size, n_pool)
    if batch_size <= 0:
        return []

    # Normalise dimensions so distance is dimensionless, since parameter
    # ranges vary across MEAM keys (Ec ~ eV, re ~ Å, alpha ~ dimensionless).
    ranges = X_pool.max(axis=0) - X_pool.min(axis=0)
    ranges[ranges < 1e-12] = 1.0
    X_norm = X_pool / ranges

    top_k = min(n_pool, max(batch_size, batch_size * oversample))
    top_idx = np.argpartition(scores, -top_k)[-top_k:]

    # Seed the picked set with the highest-score candidate.
    first = top_idx[np.argmax(scores[top_idx])]
    picked = [int(first)]
    remaining = set(int(i) for i in top_idx)
    remaining.discard(int(first))

    while len(picked) < batch_size and remaining:
        rem_idx = np.fromiter(remaining, dtype=np.int64)
        # Min distance from each remaining candidate to the picked set.
        diffs = X_norm[rem_idx, None, :] - X_norm[picked, :][None, :, :]
        dists = np.linalg.norm(diffs, axis=-1).min(axis=1)
        s_max = max(scores[rem_idx].max(), 1e-12)
        d_max = max(dists.max(), 1e-12)
        combined = scores[rem_idx] / s_max + dists / d_max
        choice = int(rem_idx[int(np.argmax(combined))])
        picked.append(choice)
        remaining.discard(choice)

    return picked


# ── End-to-end picker (one iteration of the active loop) ─────────────────────

def pick_next_batch(initial_vec: np.ndarray, X_samples: Sequence[np.ndarray],
                    y_samples: Sequence[np.ndarray],
                    batch_size: int, pool_size: int = 500,
                    ensemble_size: int = 5,
                    pool_mode: str = "sobol",
                    pert_range: float = PERT_RANGE,
                    seed: int = 0) -> np.ndarray:
    """Choose the next ``batch_size`` parameter vectors to evaluate.

    One iteration of the active loop:
      1. Generate ``pool_size`` candidates via ``pool_mode``.
      2. Train an ensemble on the current samples.
      3. Score the pool by predictive variance.
      4. Greedy diverse top-B selection.

    Returns an array of shape ``(batch_size, d)`` with the chosen
    parameter vectors. If the ensemble somehow produces all-zero variance
    (e.g. degenerate y_samples), falls back to the highest-spread Sobol
    candidates to keep making progress.
    """
    X_arr = np.asarray(X_samples)
    y_arr = np.asarray(y_samples)
    pool = generate_candidates(initial_vec, pool_mode, pool_size,
                               seed=seed, pert_range=pert_range)
    if len(X_arr) < 2:
        # Need at least 2 distinct training points before the ensemble's
        # variance is meaningful; fall back to diverse pool selection.
        scores = np.linalg.norm(pool - initial_vec[None, :], axis=1)
        picked = select_diverse_batch(pool, scores, batch_size)
        return pool[picked]

    models = train_ensemble(X_arr, y_arr, n_models=ensemble_size, seed=seed)
    scores = ensemble_variance(models, pool)
    if not np.isfinite(scores).any() or float(scores.max()) <= 0.0:
        logger.warning("Ensemble variance is degenerate; falling back to diverse-pool fill")
        scores = np.linalg.norm(pool - initial_vec[None, :], axis=1)
    picked = select_diverse_batch(pool, scores, batch_size)
    return pool[picked]
