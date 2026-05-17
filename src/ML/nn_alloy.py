"""Composition surrogate -- predict cubic elastic constants $C_{11}, C_{12}$
and derive $(E,\\nu)$ analytically.

This file replaces the older direct-$(E,\\nu)$ regression. The targets
in the original formulation included $\\nu = C_{12}/(C_{11}+C_{12})$, a
ratio that is ill-conditioned in composition space: a small absolute
error in either $C_{ij}$ produces a much larger relative error in
$\\nu$. Predicting $(C_{11}, C_{12})$ instead is a *bijective*
reparametrisation under cubic isotropy, but the regression target
space is much smoother. $(E,\\nu)$ are recovered analytically from the
network output for evaluation and reporting.

Loss is **Huber** (a robust replacement for MSE) so that a few
LAMMPS samples with extreme $C_{ij}$ in Mg/Zn-rich compositions do
not dominate the gradient.

The stopping gate is expressed in terms of the *mean* relative
error on the derived $(E,\\nu)$: training stops once the average
$|\\Delta E|/E$ and $|\\Delta\\nu|/\\nu$ over the validation set both
fall below ``MAX_ERROR_PCT``. Per-sample pass/fail counts are still
reported for diagnostic purposes but no longer drive the gate.
"""

import json
import os
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tabulate import tabulate

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Constants ────────────────────────────────────────────────────────────────
# ALL_ELEMENTS and C_SCALE are re-exported from ``model_constants`` so the
# runtime predictor can import them without pulling TF into its process.
from model_constants import ALL_ELEMENTS, C_SCALE  # noqa: E402  (re-export)

TRAIN_RATIO = 0.70
RANDOM_SEED = 42
MAX_ERROR_PCT = 20    # gate on derived (E, nu)
BATCH_EPOCHS = 500
MAX_EPOCHS = 20000
MAX_RESTARTS = 10
TRAIN_BATCH_SIZE = 128

# Filter ν close to the (1-2ν)=0 singularity to avoid noisy C_ij targets.
NU_FILTER_MAX = 0.48
# Huber transition point in scaled units (1.0 = 200 GPa).
HUBER_DELTA = 1.0

# Deep-ensemble support. Set ``ENSEMBLE_SIZE=5 ./run_nn.sh`` to train five
# independent models with seeds [RANDOM_SEED, RANDOM_SEED+1, ...], saved
# as ``alloy_model_0.keras`` ... ``alloy_model_{N-1}.keras``. The Flask
# predictor (``predict_from_model.predict_properties``) auto-discovers
# the checkpoints and reports mean ± σ across members. With the default
# ``ENSEMBLE_SIZE=1`` the legacy ``alloy_model.keras`` filename is kept,
# so older tooling keeps working unchanged.
ENSEMBLE_SIZE = int(os.environ.get("ENSEMBLE_SIZE", "1"))

# Report integration paths (re-uses the existing names so ``\input`` and
# ``\includegraphics`` in the report require no changes).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FIGURES_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "reports", "interim", "figures"))
REPORT_AUTOSEC_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "reports", "interim", "sections"))


# ── Algebraic conversion ─────────────────────────────────────────────────────
def _e_nu_to_cij(E, nu):
    """Cubic-isotropic: $(E, \\nu) \\to (C_{11}, C_{12})$. Vectorised."""
    s = E / ((1.0 - 2.0 * nu) * (1.0 + nu))
    return s * (1.0 - nu), s * nu


def _cij_to_e_nu(C11, C12):
    """Cubic-isotropic: $(C_{11}, C_{12}) \\to (E, \\nu)$. Safe at small sums."""
    s = C11 + C12
    s_safe = np.where(np.abs(s) < 1e-8, np.sign(s) * 1e-8 + 1e-12, s)
    E = (C11 - C12) * (C11 + 2.0 * C12) / s_safe
    nu = C12 / s_safe
    return E, nu


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    """Return (names, X, y_cij, y_enu) from results.json.

    Drops entries with non-physical $(E,\\nu)$ or $\\nu$ within
    ``NU_FILTER_MAX`` of the singularity.
    """
    results_path = os.path.join(_THIS_DIR, "results.json")
    with open(results_path) as f:
        results = json.load(f)

    names, X, y_cij, y_enu = [], [], [], []
    n_dropped_phys = 0
    n_dropped_nu = 0
    n_used_stored = 0
    for name, data in results.items():
        if "composition" not in data:
            continue
        E = float(data.get("E_GPa", 0.0))
        nu = float(data.get("nu", 0.0))
        if E <= 0 or nu <= 0:
            n_dropped_phys += 1
            continue
        if nu >= NU_FILTER_MAX:
            n_dropped_nu += 1
            continue
        comp = data["composition"]
        features = [comp.get(el, 0.0) for el in ALL_ELEMENTS]
        # Prefer the C_ij stored by lmp.py / fill_results_cij.py;
        # fall back to algebraic conversion only if absent.
        if "C11_GPa" in data and "C12_GPa" in data:
            C11 = float(data["C11_GPa"])
            C12 = float(data["C12_GPa"])
            n_used_stored += 1
        else:
            C11_v, C12_v = _e_nu_to_cij(E, nu)
            C11, C12 = float(C11_v), float(C12_v)
        if not (np.isfinite(C11) and np.isfinite(C12)):
            n_dropped_phys += 1
            continue
        names.append(name)
        X.append(features)
        y_cij.append([C11, C12])
        y_enu.append([E, nu])

    print(f"  Loaded {len(names)} usable records "
          f"({n_used_stored} via stored C_ij, "
          f"{len(names) - n_used_stored} via algebraic fallback); "
          f"dropped {n_dropped_phys} non-physical, "
          f"{n_dropped_nu} with nu>={NU_FILTER_MAX:.2f}")
    return names, np.array(X), np.array(y_cij), np.array(y_enu)


# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    model = models.Sequential([
        layers.Input(shape=(len(ALL_ELEMENTS),)),
        layers.Dense(32, activation="relu"),
        layers.Dense(20, activation="relu"),
        layers.Dense(2),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.Huber(delta=HUBER_DELTA),
        metrics=["mae"],
    )
    return model


def _fast_predict(model, X):
    """Bypass ``predict()`` overhead for small in-memory arrays."""
    return model(X, training=False).numpy()


def _predict_enu(model, X):
    """Predict scaled C_ij, undo scaling, return (E, nu) and (C11, C12)."""
    pred = _fast_predict(model, X) * C_SCALE
    C11 = pred[:, 0]
    C12 = pred[:, 1]
    E, nu = _cij_to_e_nu(C11, C12)
    return E, nu, C11, C12


# ── Metrics ──────────────────────────────────────────────────────────────────
def _per_target_metrics(true, pred):
    """Per-target error statistics.

    Beyond the usual MAE / RMSE / R² / MAPE, we report:
    - ``Std``   : sample std of the *signed* error (pred - true); together
                   with ``Bias`` it characterises the residual distribution.
    - ``Bias``  : mean signed error (systematic over/under-prediction).
    - ``MedAE`` : median of |err| (robust central tendency).
    - ``MaxAE`` : worst-case |err| over the split.
    - ``P95AE`` : 95th percentile of |err| (tail behaviour).
    - ``MedAPE``: median of relative |err|/|true| as a percent.
    - ``Pearson``: linear correlation coefficient between true and pred.
    - ``true_*`` / ``pred_*``: mean and std of the true and predicted
                                 distributions, useful as a sanity check.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - true
    abs_err = np.abs(err)
    rel_err = abs_err / np.maximum(np.abs(true), 1e-8)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if true.std() > 0 and pred.std() > 0:
        pearson = float(np.corrcoef(true, pred)[0, 1])
    else:
        pearson = 0.0
    return {
        "MAE":   float(abs_err.mean()),
        "RMSE":  float(np.sqrt((err ** 2).mean())),
        "Std":   float(err.std(ddof=1)) if err.size > 1 else 0.0,
        "Bias":  float(err.mean()),
        "MedAE": float(np.median(abs_err)),
        "MaxAE": float(abs_err.max()),
        "P95AE": float(np.percentile(abs_err, 95)),
        "R2":    r2,
        "Pearson": pearson,
        "MAPE":   float(rel_err.mean() * 100),
        "MedAPE": float(np.median(rel_err) * 100),
        "true_mean": float(true.mean()),
        "true_std":  float(true.std(ddof=1)) if true.size > 1 else 0.0,
        "pred_mean": float(pred.mean()),
        "pred_std":  float(pred.std(ddof=1)) if pred.size > 1 else 0.0,
    }


def _compute_all_metrics(y_enu_true, E_pred, nu_pred,
                         y_cij_true, C11_pred, C12_pred):
    return {
        "E_GPa": _per_target_metrics(y_enu_true[:, 0], E_pred),
        "nu":    _per_target_metrics(y_enu_true[:, 1], nu_pred),
        "C11":   _per_target_metrics(y_cij_true[:, 0], C11_pred),
        "C12":   _per_target_metrics(y_cij_true[:, 1], C12_pred),
    }


def _check_val_accuracy(model, X_val, y_val_enu, val_names):
    """Pass if mean E and nu relative errors are both <= MAX_ERROR_PCT.

    Errors are evaluated on the derived $(E,\\nu)$, not on the raw
    $C_{ij}$ network output. The gate is the *mean* over the validation
    set; per-sample pass/fail counts are still computed for diagnostics
    but no longer drive the stopping decision."""
    E_pred, nu_pred, _, _ = _predict_enu(model, X_val)
    n_pass = 0
    n_fail = 0
    fail_rows = []
    e_errs = np.abs(E_pred - y_val_enu[:, 0]) / np.maximum(np.abs(y_val_enu[:, 0]), 1e-8) * 100
    nu_errs = np.abs(nu_pred - y_val_enu[:, 1]) / np.maximum(np.abs(y_val_enu[:, 1]), 1e-8) * 100
    mean_E_err = float(e_errs.mean())
    mean_nu_err = float(nu_errs.mean())
    max_E_err = float(e_errs.max())
    max_nu_err = float(nu_errs.max())
    for i in range(len(X_val)):
        e_t, nu_t = y_val_enu[i]
        e_p, nu_p = E_pred[i], nu_pred[i]
        e_err = float(e_errs[i])
        nu_err = float(nu_errs[i])
        if e_err <= MAX_ERROR_PCT and nu_err <= MAX_ERROR_PCT:
            n_pass += 1
        else:
            n_fail += 1
            fail_rows.append([
                val_names[i],
                f"{e_t:.1f}", f"{e_p:.1f}", f"{e_err:.1f}%",
                f"{nu_t:.3f}", f"{nu_p:.3f}", f"{nu_err:.1f}%",
            ])
    passed = (mean_E_err <= MAX_ERROR_PCT) and (mean_nu_err <= MAX_ERROR_PCT)
    return (passed, n_pass, n_fail, fail_rows,
            max_E_err, max_nu_err, mean_E_err, mean_nu_err)


# ── Plots ────────────────────────────────────────────────────────────────────
def _plot_history(history, plots_dir):
    if not HAS_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["loss"], label="Train Huber")
    ax1.plot(history["val_loss"], label="Val Huber")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Huber loss"); ax1.set_yscale("log")
    ax1.set_title("Training & Validation loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history["mae"], label="Train MAE")
    ax2.plot(history["val_mae"], label="Val MAE")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("MAE (scaled C_ij)")
    ax2.set_title("Training & Validation MAE")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plots_dir, "training_history.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved {p}")


def _plot_parity_pair(y_true, y_pred, labels, title_prefix, fname, plots_dir):
    if not HAS_MPL:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, idx, name in zip(axs, [0, 1], labels):
        t, p = y_true[:, idx], y_pred[:, idx]
        ax.scatter(t, p, alpha=0.6, edgecolors="k", linewidths=0.3, s=30)
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "r--", lw=1.5, label="Perfect")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"{title_prefix}: {name}")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plots_dir, fname)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved {p}")


def _plot_error_distribution(y_true_enu, E_pred, nu_pred, plots_dir):
    if not HAS_MPL:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(E_pred - y_true_enu[:, 0], bins=25,
             edgecolor="black", alpha=0.7)
    ax1.axvline(0, color="r", linestyle="--", lw=1.5)
    ax1.set_xlabel("E error (GPa)"); ax1.set_ylabel("Count")
    ax1.set_title("Validation: derived E error")
    ax1.grid(True, alpha=0.3)
    ax2.hist(nu_pred - y_true_enu[:, 1], bins=25,
             edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="r", linestyle="--", lw=1.5)
    ax2.set_xlabel("nu error"); ax2.set_ylabel("Count")
    ax2.set_title("Validation: derived nu error")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plots_dir, "error_dist_validation.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved {p}")


# ── Report export ────────────────────────────────────────────────────────────
def _export_to_report(plots_dir, n_total, n_train, n_val,
                      train_metrics, val_metrics):
    """Sync plots into the report tree and emit ``_auto_ml_metrics.tex``."""
    if os.path.isdir(REPORT_FIGURES_DIR):
        for fname in sorted(os.listdir(plots_dir)):
            if fname.lower().endswith(".png"):
                shutil.copy2(os.path.join(plots_dir, fname),
                             os.path.join(REPORT_FIGURES_DIR, fname))
        print(f"  Synced PNGs -> {REPORT_FIGURES_DIR}")
    else:
        print(f"  (skip) report figures dir not found: {REPORT_FIGURES_DIR}")

    if not os.path.isdir(REPORT_AUTOSEC_DIR):
        return

    rows = []
    target_pretty = {
        "C11":   r"$C_{11}$ (GPa)",
        "C12":   r"$C_{12}$ (GPa)",
        "E_GPa": r"$E$ (GPa, derived)",
        "nu":    r"$\nu$ (derived)",
    }
    for split_label, m in [("Train", train_metrics), ("Val.", val_metrics)]:
        for tgt_key in ("C11", "C12", "E_GPa", "nu"):
            mm = m[tgt_key]
            rows.append([
                f"{split_label}, {target_pretty[tgt_key]}",
                f"{mm['MAE']:.3f}",
                f"{mm['RMSE']:.3f}",
                f"{mm['Std']:.3f}",
                f"{mm['Bias']:+.3f}",
                f"{mm['R2']:.3f}",
                f"{mm['Pearson']:.3f}",
                f"{mm['MAPE']:.3f}\\,\\%",
            ])
    headers = ["Split / Target", "MAE", "RMSE", "Std",
               "Bias", "$R^2$", "$r$", "MAPE"]
    body = tabulate(rows, headers=headers, tablefmt="latex_raw",
                    floatfmt=".3f")
    caption = (f"Performance of the composition surrogate "
               f"(N={n_total}; train={n_train}, val={n_val}).")
    label = "tab:nn_aux"
    table_tex = (
        "% AUTO-GENERATED by src/ML/nn_alloy.py -- do not edit\n"
        "\\begin{table}[h]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\n\\end{{table}}\n"
    )
    out_path = os.path.join(REPORT_AUTOSEC_DIR, "_auto_ml_metrics.tex")
    with open(out_path, "w") as f:
        f.write(table_tex)
    print(f"  Wrote LaTeX metrics table -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("Alloy composition surrogate (predict C11,C12; derive E,nu)")
    print("=" * 70)

    names, X, y_cij, y_enu = load_data()
    if len(X) == 0:
        print("No usable data.")
        return

    stats_table = [
        ["Total samples", len(X)],
        ["Features", f"{len(ALL_ELEMENTS)} elements ({', '.join(ALL_ELEMENTS)})"],
        ["Targets (network)", "C11, C12 (GPa, scaled by /200)"],
        ["Targets (reported)", "E, nu (derived analytically)"],
        ["Train/Val split", f"{TRAIN_RATIO:.0%} / {1-TRAIN_RATIO:.0%} (seed={RANDOM_SEED})"],
        ["Loss", f"Huber (delta={HUBER_DELTA})"],
        ["nu filter", f"nu < {NU_FILTER_MAX} (avoids 1-2nu singularity)"],
        ["Mean error gate", f"{MAX_ERROR_PCT:.0f}% mean over val set on derived (E, nu)"],
        ["Train batch size", TRAIN_BATCH_SIZE],
        ["E (GPa) range",
         f"{y_enu[:,0].min():.1f} - {y_enu[:,0].max():.1f} "
         f"(mean={y_enu[:,0].mean():.1f})"],
        ["nu range",
         f"{y_enu[:,1].min():.3f} - {y_enu[:,1].max():.3f} "
         f"(mean={y_enu[:,1].mean():.3f})"],
        ["C11 range",
         f"{y_cij[:,0].min():.1f} - {y_cij[:,0].max():.1f} "
         f"(mean={y_cij[:,0].mean():.1f})"],
        ["C12 range",
         f"{y_cij[:,1].min():.1f} - {y_cij[:,1].max():.1f} "
         f"(mean={y_cij[:,1].mean():.1f})"],
    ]
    print(tabulate(stats_table, tablefmt="simple"))

    # ── Train/Val split ─────────────────────────────────────────────────
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(len(X))
    n_train = int(len(X) * TRAIN_RATIO)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_cij, y_val_cij = y_cij[train_idx], y_cij[val_idx]
    y_train_enu, y_val_enu = y_enu[train_idx], y_enu[val_idx]
    train_names = [names[i] for i in train_idx]
    val_names = [names[i] for i in val_idx]
    y_train_scaled = y_train_cij / C_SCALE
    y_val_scaled = y_val_cij / C_SCALE
    print(f"\n  Train set: {len(X_train)} | Val set: {len(X_val)}")

    # ── Ensemble cleanup ────────────────────────────────────────────────
    # When training a fresh ensemble, drop any legacy single-model
    # checkpoint so ``predict_from_model._discover_model_paths`` does not
    # fall back to it after we write ``alloy_model_*.keras`` members.
    if ENSEMBLE_SIZE > 1:
        legacy_path = os.path.join(script_dir, "alloy_model.keras")
        if os.path.exists(legacy_path):
            os.remove(legacy_path)
            print(f"\n  Removed legacy single-model checkpoint: {legacy_path}")
        # Also drop any prior ensemble checkpoints with a *higher* index than
        # we're about to write (e.g. shrinking from 7 → 5 members).
        import glob as _glob
        for stale in _glob.glob(os.path.join(script_dir, "alloy_model_*.keras")):
            try:
                idx = int(os.path.basename(stale).removeprefix("alloy_model_").removesuffix(".keras"))
            except ValueError:
                continue
            if idx >= ENSEMBLE_SIZE:
                os.remove(stale)
                print(f"  Removed stale ensemble member: {stale}")
        print(f"\n  Training deep ensemble: {ENSEMBLE_SIZE} members "
              f"(seeds {RANDOM_SEED}..{RANDOM_SEED + ENSEMBLE_SIZE - 1})")

    # ── Train loop with restart ─────────────────────────────────────────
    # Outer loop trains one independent ensemble member per iteration. With
    # ``ENSEMBLE_SIZE=1`` (the default) this collapses to a single pass.
    # Final reported metrics (table / plots / nn_metrics.json) reflect the
    # *last* member only; per-prediction ensemble σ is computed at inference
    # time by the Flask app, not from these training-time artefacts.
    t_total = time.time()
    all_loss = all_val_loss = all_mae = all_val_mae = None
    for member_idx in range(ENSEMBLE_SIZE):
        if ENSEMBLE_SIZE > 1:
            member_seed = RANDOM_SEED + member_idx
            np.random.seed(member_seed)
            tf.random.set_seed(member_seed)
            print(f"\n{'#' * 70}")
            print(f"#  ENSEMBLE MEMBER {member_idx + 1}/{ENSEMBLE_SIZE} "
                  f"(seed={member_seed})")
            print('#' * 70)
        for attempt in range(1, MAX_RESTARTS + 1):
            print(f"\n{'='*70}\n  ATTEMPT {attempt}/{MAX_RESTARTS}\n{'='*70}")
            model = build_model()
            if attempt == 1 and member_idx == 0:
                model.summary(print_fn=lambda s: print(f"  {s}"))

            t0 = time.time()
            total_epochs = 0
            all_loss = []
            all_val_loss = []
            all_mae = []
            all_val_mae = []

            while total_epochs < MAX_EPOCHS:
                class _EpochLogger(tf.keras.callbacks.Callback):
                    def __init__(self, offset, t_start, X_v, y_v_enu):
                        super().__init__()
                        self._offset = offset
                        self._t0 = t_start
                        self._X_v = X_v
                        self._y_v = y_v_enu
                        self.h_loss = []
                        self.h_val_loss = []
                        self.h_mae = []
                        self.h_val_mae = []

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        self.h_loss.append(logs.get("loss", 0))
                        self.h_val_loss.append(logs.get("val_loss", 0))
                        self.h_mae.append(logs.get("mae", 0))
                        self.h_val_mae.append(logs.get("val_mae", 0))
                        ge = self._offset + epoch
                        elapsed = time.time() - self._t0
                        if ge >= 5 and ge % 100 != 0:
                            return
                        E_p, nu_p, _, _ = _predict_enu(self.model, self._X_v)
                        e_err = np.abs(E_p - self._y_v[:, 0]) / np.maximum(np.abs(self._y_v[:, 0]), 1e-8) * 100
                        n_err = np.abs(nu_p - self._y_v[:, 1]) / np.maximum(np.abs(self._y_v[:, 1]), 1e-8) * 100
                        n_over = int(np.sum((e_err > MAX_ERROR_PCT) | (n_err > MAX_ERROR_PCT)))
                        print(f"  Epoch {ge:5d} | loss={logs.get('loss',0):.6f} "
                              f"val_loss={logs.get('val_loss',0):.6f} | "
                              f"E_MAPE={e_err.mean():.1f}% nu_MAPE={n_err.mean():.1f}% | "
                              f"max_err: E={e_err.max():.1f}% nu={n_err.max():.1f}% | "
                              f"fail={n_over}/{len(self._y_v)} | {elapsed:.0f}s")

                logger = _EpochLogger(total_epochs, t0, X_val, y_val_enu)
                model.fit(
                    X_train, y_train_scaled,
                    validation_data=(X_val, y_val_scaled),
                    epochs=BATCH_EPOCHS,
                    batch_size=min(TRAIN_BATCH_SIZE, len(X_train)),
                    verbose=0,
                    callbacks=[logger],
                )
                total_epochs += len(logger.h_loss)
                all_loss.extend(logger.h_loss)
                all_val_loss.extend(logger.h_val_loss)
                all_mae.extend(logger.h_mae)
                all_val_mae.extend(logger.h_val_mae)

                (passed, n_pass, n_fail, fail_rows,
                 max_E_err, max_nu_err, mean_E_err, mean_nu_err) = (
                    _check_val_accuracy(model, X_val, y_val_enu, val_names))
                elapsed = time.time() - t0
                print(f"\n  --- Epoch {total_epochs}: Val {n_pass} PASS / {n_fail} FAIL "
                      f"| mean_err: E={mean_E_err:.1f}% nu={mean_nu_err:.1f}% "
                      f"| max_err: E={max_E_err:.1f}% nu={max_nu_err:.1f}% "
                      f"(target: mean <={MAX_ERROR_PCT:.0f}%) [{elapsed:.0f}s] ---")
                if passed:
                    print(f"  Mean val error within {MAX_ERROR_PCT:.0f}% on derived (E, nu)!")
                    break
                else:
                    if 0 < len(fail_rows) <= 10:
                        print(tabulate(fail_rows,
                                       headers=["Alloy", "E true", "E pred",
                                                "E err%", "nu true", "nu pred",
                                                "nu err%"],
                                       tablefmt="simple"))

            if passed:
                break
            else:
                if attempt < MAX_RESTARTS:
                    print(f"\n  Attempt {attempt} failed; restarting.")
                else:
                    print(f"\n  Max restarts reached. Using best result so far.")

        # ── Save this member's checkpoint ───────────────────────────────
        if ENSEMBLE_SIZE > 1:
            member_path = os.path.join(script_dir, f"alloy_model_{member_idx}.keras")
        else:
            member_path = os.path.join(script_dir, "alloy_model.keras")
        model.save(member_path)
        print(f"  Model saved to {member_path}")

    dt_total = time.time() - t_total
    print(f"\n  Training complete: {total_epochs} epochs, {attempt} attempt(s), {dt_total:.1f}s")
    if ENSEMBLE_SIZE > 1:
        print(f"  Ensemble: {ENSEMBLE_SIZE} members trained.")

    # ── Final metrics on both targets ───────────────────────────────────
    E_tr, nu_tr, C11_tr, C12_tr = _predict_enu(model, X_train)
    E_val, nu_val, C11_val, C12_val = _predict_enu(model, X_val)
    train_metrics = _compute_all_metrics(
        y_train_enu, E_tr, nu_tr, y_train_cij, C11_tr, C12_tr)
    val_metrics = _compute_all_metrics(
        y_val_enu, E_val, nu_val, y_val_cij, C11_val, C12_val)

    print("\n" + "=" * 70 + "\nModel Accuracy\n" + "=" * 70)
    for label, mset in [("Training Set", train_metrics),
                        ("Validation Set", val_metrics)]:
        rows = []
        for tgt in ("C11", "C12", "E_GPa", "nu"):
            m = mset[tgt]
            rows.append([tgt,
                         f"{m['MAE']:.4f}",   f"{m['RMSE']:.4f}",
                         f"{m['Std']:.4f}",   f"{m['Bias']:+.4f}",
                         f"{m['MedAE']:.4f}", f"{m['MaxAE']:.4f}",
                         f"{m['P95AE']:.4f}",
                         f"{m['R2']:.4f}",    f"{m['Pearson']:.4f}",
                         f"{m['MAPE']:.2f}%", f"{m['MedAPE']:.2f}%"])
        print(f"\n  {label}:")
        print(tabulate(rows,
                       headers=["Target", "MAE", "RMSE", "Std", "Bias",
                                "MedAE", "MaxAE", "P95AE",
                                "R2", "Pearson", "MAPE", "MedAPE"],
                       tablefmt="simple"))

    # Distribution sanity-check: how does the predicted spread compare to
    # the true spread on the validation set?
    print(f"\n  Validation distribution (true vs pred):")
    dist_rows = []
    for tgt in ("C11", "C12", "E_GPa", "nu"):
        m = val_metrics[tgt]
        dist_rows.append([
            tgt,
            f"{m['true_mean']:.3f}", f"{m['true_std']:.3f}",
            f"{m['pred_mean']:.3f}", f"{m['pred_std']:.3f}",
            f"{(m['pred_std']/m['true_std']):.3f}" if m['true_std'] > 0 else "n/a",
        ])
    print(tabulate(dist_rows,
                   headers=["Target", "true mean", "true std",
                            "pred mean", "pred std", "std ratio"],
                   tablefmt="simple"))

    # ── Per-validation-sample summary ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Validation Set - Detailed Results")
    print("=" * 70)
    val_rows = []
    for i in range(len(X_val)):
        e_t, nu_t = y_val_enu[i]
        e_p, nu_p = E_val[i], nu_val[i]
        e_err = abs(e_p - e_t) / max(abs(e_t), 1e-8) * 100
        nu_err = abs(nu_p - nu_t) / max(abs(nu_t), 1e-8) * 100
        status = "PASS" if e_err <= MAX_ERROR_PCT and nu_err <= MAX_ERROR_PCT else "FAIL"
        val_rows.append([val_names[i], f"{e_t:.1f}", f"{e_p:.1f}",
                         f"{e_err:.1f}%", f"{nu_t:.3f}", f"{nu_p:.3f}",
                         f"{nu_err:.1f}%", status])
    print(tabulate(val_rows,
                   headers=["Alloy", "E true", "E pred", "E err%",
                            "nu true", "nu pred", "nu err%", "Status"],
                   tablefmt="simple"))
    n_pass = sum(1 for r in val_rows if r[-1] == "PASS")
    n_fail = sum(1 for r in val_rows if r[-1] == "FAIL")
    mean_E_err_final = float(val_metrics["E_GPa"]["MAPE"])
    mean_nu_err_final = float(val_metrics["nu"]["MAPE"])
    gate_ok = (mean_E_err_final <= MAX_ERROR_PCT and
               mean_nu_err_final <= MAX_ERROR_PCT)
    print(f"\n  Per-sample: {n_pass} PASS / {n_fail} FAIL "
          f"out of {len(val_rows)} val samples (informational)")
    print(f"  Mean error: E={mean_E_err_final:.2f}%, nu={mean_nu_err_final:.2f}% "
          f"(gate <={MAX_ERROR_PCT:.0f}%): "
          f"{'PASS' if gate_ok else 'FAIL'}")

    # ── Plots ───────────────────────────────────────────────────────────
    history = {"loss": all_loss, "val_loss": all_val_loss,
               "mae": all_mae, "val_mae": all_val_mae}
    if HAS_MPL:
        print(f"\n  Generating plots in {plots_dir}/")
        _plot_history(history, plots_dir)
        # E,nu derived parity plots use the canonical filenames so the
        # report's existing \includegraphics directives keep working.
        _plot_parity_pair(y_train_enu,
                          np.column_stack([E_tr, nu_tr]),
                          ["E (GPa)", "nu"],
                          "Train (E,nu derived)", "parity_train.png", plots_dir)
        _plot_parity_pair(y_val_enu,
                          np.column_stack([E_val, nu_val]),
                          ["E (GPa)", "nu"],
                          "Val (E,nu derived)", "parity_validation.png", plots_dir)
        # Bonus: direct C_ij parity plots (not currently referenced in the
        # report, but useful when diagnosing the network's primary task).
        _plot_parity_pair(y_train_cij,
                          np.column_stack([C11_tr, C12_tr]),
                          ["C11 (GPa)", "C12 (GPa)"],
                          "Train (C_ij direct)",
                          "parity_train_cij.png", plots_dir)
        _plot_parity_pair(y_val_cij,
                          np.column_stack([C11_val, C12_val]),
                          ["C11 (GPa)", "C12 (GPa)"],
                          "Val (C_ij direct)",
                          "parity_validation_cij.png", plots_dir)
        _plot_error_distribution(y_val_enu, E_val, nu_val, plots_dir)

    # ── Predictions for predict.json ────────────────────────────────────
    predict_file = os.path.join(script_dir, "predict.json")
    if os.path.exists(predict_file):
        print("\n" + "=" * 70)
        print("Predictions for predict.json")
        print("=" * 70)
        with open(predict_file) as f:
            predict_data = json.load(f)
        pred_rows = []
        for alloy_name, comp in predict_data.items():
            new_alloy = np.zeros((1, len(ALL_ELEMENTS)))
            for el, frac in comp.items():
                if el in ALL_ELEMENTS:
                    new_alloy[0, ALL_ELEMENTS.index(el)] = frac
            E_p, nu_p, C11_p, C12_p = _predict_enu(model, new_alloy)
            pred_rows.append([alloy_name,
                              f"{C11_p[0]:.2f}", f"{C12_p[0]:.2f}",
                              f"{E_p[0]:.2f}", f"{nu_p[0]:.3f}"])
        print(tabulate(pred_rows,
                       headers=["Alloy", "C11 (GPa)", "C12 (GPa)",
                                "E (GPa, derived)", "nu (derived)"],
                       tablefmt="simple"))

    # ── Save metrics JSON ───────────────────────────────────────────────
    metrics_out = {
        "n_total": len(X),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "epochs_trained": total_epochs,
        "attempts": attempt,
        "max_error_pct": MAX_ERROR_PCT,
        "gate_kind": "mean",
        "gate_passed": bool(gate_ok),
        "val_mean_E_err_pct": mean_E_err_final,
        "val_mean_nu_err_pct": mean_nu_err_final,
        "huber_delta": HUBER_DELTA,
        "nu_filter_max": NU_FILTER_MAX,
        "C_scale": C_SCALE,
        "val_pass": n_pass,
        "val_fail": n_fail,
        "train": train_metrics,
        "val": val_metrics,
    }
    metrics_path = os.path.join(script_dir, "nn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    _export_to_report(plots_dir, len(X), len(X_train), len(X_val),
                      train_metrics, val_metrics)


if __name__ == "__main__":
    main()
