"""Inverse alloy-design surrogate -- map a target $(E,\\nu)$ back to a
composition.

This is the *mirror* of ``src/ML/nn_alloy.py``. Where the forward
surrogate learns ``composition -> (C11, C12)``, this network learns the
inverse ``(C11, C12) -> composition``. As in the forward model we work
in $C_{ij}$ space rather than $(E,\\nu)$ directly: the user types
$(E,\\nu)$ but we convert to $(C_{11}, C_{12})$ first, because
$\\nu = C_{12}/(C_{11}+C_{12})$ is ill-conditioned and the $C_{ij}$
target space is far smoother.

The inverse map is fundamentally **one-to-many**: many compositions can
realise the same $(E,\\nu)$, and some targets are not achievable at all.
A plain ``target -> composition`` regression would average over all
valid compositions and produce a blurry, often unphysical answer. We
avoid that with a **forward-consistency (cycle) loss**: the network's
proposed composition is pushed back through the *frozen* forward model
(``src/ML/alloy_model_*.keras``) and the loss is the Huber error between
the reconstructed $(C_{11}, C_{12})$ and the target. Any composition
that reconstructs the target is rewarded -- exactly the right objective
for a one-to-many map.

Two auxiliary terms keep the solutions grounded:

* a small **composition-reconstruction** term (MSE against the real
  alloy that produced this target) keeps proposals near realistic,
  few-element compositions rather than dense softmax smears;
* an optional **entropy penalty** (``INV_ENTROPY_WEIGHT``) encourages
  sparse (few-element) compositions when enabled.

The 12-d composition output goes through a **softmax**, so proposals are
automatically non-negative and sum to 1 -- the simplex constraint, for
free.

A deep ensemble (``ENSEMBLE_SIZE``, default 5) is trained with distinct
seeds. At inference the independently-trained members give *diverse*
candidate compositions, which is what feeds the app's top-K list.
"""

from __future__ import annotations

import glob
import json
import os
import sys
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

# ── Cross-module reuse ───────────────────────────────────────────────────────
# The element basis, output scale and the cubic-isotropic algebra are the
# single source of truth in ``src/ML/model_constants`` -- import them rather
# than re-declaring (drift between forward and inverse would be a silent bug).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "ML"))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from model_constants import ALL_ELEMENTS, C_SCALE, cij_to_e_nu  # noqa: E402

# ── Hyper-parameters (mirroring nn_alloy.py where it makes sense) ─────────────
TRAIN_RATIO = 0.70
RANDOM_SEED = 42
NU_FILTER_MAX = 0.48          # avoid the (1-2nu)=0 singularity in C_ij targets
HUBER_DELTA = 1.0             # Huber transition in scaled units (1.0 = 200 GPa)
MAX_EPOCHS = int(os.environ.get("INV_MAX_EPOCHS", "4000"))
PATIENCE = int(os.environ.get("INV_PATIENCE", "200"))   # early-stop patience on val loss
TRAIN_BATCH_SIZE = 128

# Loss weights. The forward-consistency (recon) term is primary; the
# composition term grounds proposals near real alloys. Tunable via env.
RECON_WEIGHT = float(os.environ.get("INV_RECON_WEIGHT", "1.0"))
COMP_WEIGHT = float(os.environ.get("INV_COMP_WEIGHT", "0.5"))
ENTROPY_WEIGHT = float(os.environ.get("INV_ENTROPY_WEIGHT", "0.0"))

ENSEMBLE_SIZE = int(os.environ.get("ENSEMBLE_SIZE", "1"))


# ── Algebraic conversion ─────────────────────────────────────────────────────
def _e_nu_to_cij(E, nu):
    """Cubic-isotropic: $(E, \\nu) \\to (C_{11}, C_{12})$. Vectorised."""
    E = np.asarray(E, dtype=float)
    nu = np.asarray(nu, dtype=float)
    s = E / ((1.0 - 2.0 * nu) * (1.0 + nu))
    return s * (1.0 - nu), s * nu


# ── Frozen forward model ─────────────────────────────────────────────────────
def _discover_forward_keras() -> str:
    """Return the path to a forward checkpoint to use for the cycle loss.

    Prefers the first ensemble member ``alloy_model_0.keras``; falls back
    to the legacy single ``alloy_model.keras``. Uses *one* member (not the
    full ensemble) so the training graph stays small and differentiable;
    the full ensemble is only used at inference for the reported check.
    """
    ensemble = sorted(glob.glob(os.path.join(_ML_DIR, "alloy_model_*.keras")))
    if ensemble:
        return ensemble[0]
    legacy = os.path.join(_ML_DIR, "alloy_model.keras")
    if os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(
        f"No forward checkpoint (.keras) found in {_ML_DIR}. "
        "Train the forward surrogate first: 'src/ML/run_nn.sh'."
    )


def _load_frozen_forward() -> tf.keras.Model:
    path = _discover_forward_keras()
    fwd = tf.keras.models.load_model(path)
    fwd.trainable = False
    for layer in fwd.layers:
        layer.trainable = False
    print(f"  Forward (frozen) loaded from {os.path.relpath(path, _THIS_DIR)}")
    return fwd


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    """Return (names, X_cij_scaled, y_comp, y_enu) from ML's results.json.

    Same physicality filter as the forward surrogate (E>0, nu>0,
    nu<NU_FILTER_MAX) plus the mechanical-stability guard C11>=C12, so the
    inverse net never learns to target an unstable point.
    """
    results_path = os.path.join(_ML_DIR, "results.json")
    with open(results_path) as f:
        results = json.load(f)

    names, X_cij, y_comp, y_enu = [], [], [], []
    n_drop_phys = n_drop_nu = n_drop_cauchy = 0
    for name, data in results.items():
        if "composition" not in data:
            continue
        E = float(data.get("E_GPa", 0.0))
        nu = float(data.get("nu", 0.0))
        if E <= 0 or nu <= 0:
            n_drop_phys += 1
            continue
        if nu >= NU_FILTER_MAX:
            n_drop_nu += 1
            continue
        if "C11_GPa" in data and "C12_GPa" in data:
            C11 = float(data["C11_GPa"])
            C12 = float(data["C12_GPa"])
        else:
            C11_v, C12_v = _e_nu_to_cij(E, nu)
            C11, C12 = float(C11_v), float(C12_v)
        if not (np.isfinite(C11) and np.isfinite(C12)):
            n_drop_phys += 1
            continue
        if C11 < C12:
            n_drop_cauchy += 1
            continue
        comp = data["composition"]
        features = [comp.get(el, 0.0) for el in ALL_ELEMENTS]
        # Renormalise the stored composition defensively so the softmax
        # target is a proper simplex point.
        tot = sum(features)
        if tot <= 0:
            n_drop_phys += 1
            continue
        features = [f / tot for f in features]
        names.append(name)
        X_cij.append([C11, C12])
        y_comp.append(features)
        y_enu.append([E, nu])

    print(f"  Loaded {len(names)} usable records; dropped "
          f"{n_drop_phys} non-physical, {n_drop_nu} with nu>={NU_FILTER_MAX:.2f}, "
          f"{n_drop_cauchy} with C11<C12")
    X_cij = np.asarray(X_cij, dtype=np.float64)
    return names, X_cij, np.asarray(y_comp, dtype=np.float64), np.asarray(y_enu, dtype=np.float64)


# ── Model ────────────────────────────────────────────────────────────────────
def build_training_model(forward: tf.keras.Model):
    """Inverse net + frozen forward, wired for the cycle loss.

    Inputs : scaled target (C11, C12)            shape (2,)
    Outputs: [recon_cij, comp]
        - ``recon_cij`` = forward(comp), scaled (C11, C12)  -> Huber vs target
        - ``comp``      = softmax composition (12-d)        -> MSE  vs real alloy
    """
    target = layers.Input(shape=(2,), name="target_cij")
    h = layers.Dense(32, activation="relu")(target)
    h = layers.Dense(20, activation="relu")(h)
    comp = layers.Dense(len(ALL_ELEMENTS), activation="softmax", name="comp")(h)
    recon = forward(comp)                       # frozen; scaled (C11, C12)
    recon = layers.Activation("linear", name="recon")(recon)

    model = models.Model(inputs=target, outputs=[recon, comp])

    if ENTROPY_WEIGHT > 0:
        # Sum_i p_i log p_i is negative; adding +H*entropy (entropy = -sum p log p)
        # to the loss *penalises* high entropy -> pushes toward sparse comps.
        entropy = -tf.reduce_mean(
            tf.reduce_sum(comp * tf.math.log(comp + 1e-9), axis=-1))
        model.add_loss(ENTROPY_WEIGHT * entropy)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss={"recon": tf.keras.losses.Huber(delta=HUBER_DELTA),
              "comp": tf.keras.losses.MeanSquaredError()},
        loss_weights={"recon": RECON_WEIGHT, "comp": COMP_WEIGHT},
    )
    return model


def _inverse_only(training_model: tf.keras.Model) -> tf.keras.Model:
    """Extract the target->composition sub-model for saving / inference."""
    return models.Model(inputs=training_model.input,
                        outputs=training_model.get_layer("comp").output)


# ── Round-trip metrics ───────────────────────────────────────────────────────
def _roundtrip_enu(inv_model, forward, X_cij_scaled):
    """target -> comp -> forward -> achieved (E, nu, C11, C12)."""
    comp = inv_model(X_cij_scaled, training=False).numpy()
    recon = forward(comp, training=False).numpy() * C_SCALE
    C11, C12 = recon[:, 0], recon[:, 1]
    E, nu = cij_to_e_nu(C11, C12)
    return E, nu, C11, C12, comp


def _metrics(true, pred):
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    err = pred - true
    abs_err = np.abs(err)
    rel = abs_err / np.maximum(np.abs(true), 1e-8)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE": float(rel.mean() * 100),
        "MedAPE": float(np.median(rel) * 100),
        "P95AE": float(np.percentile(abs_err, 95)),
        "R2": r2,
    }


# ── Plots ────────────────────────────────────────────────────────────────────
def _plot_parity(target, achieved, label, fname, plots_dir):
    if not HAS_MPL:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, idx, name in zip(axs, [0, 1], ["E (GPa)", "nu"]):
        t, p = target[:, idx], achieved[:, idx]
        ax.scatter(t, p, alpha=0.5, edgecolors="k", linewidths=0.3, s=25)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1.5,
                label="Perfect round-trip")
        ax.set_xlabel(f"Target {name}")
        ax.set_ylabel(f"Achieved {name} (target→comp→forward)")
        ax.set_title(f"{label}: {name}")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plots_dir, fname)
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  Saved {p}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    plots_dir = os.path.join(_THIS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("Inverse alloy-design surrogate ((C11,C12) -> composition)")
    print("=" * 70)

    forward = _load_frozen_forward()
    names, X_cij, y_comp, y_enu = load_data()
    if len(X_cij) == 0:
        print("No usable data.")
        return

    X_in = X_cij / C_SCALE        # scaled target = forward output space

    print(tabulate([
        ["Total samples", len(X_in)],
        ["Input (network)", "target C11, C12 (GPa, scaled by /200)"],
        ["Output (network)", f"{len(ALL_ELEMENTS)}-d composition (softmax simplex)"],
        ["Train/Val split", f"{TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%} (seed={RANDOM_SEED})"],
        ["Loss", f"Huber recon (w={RECON_WEIGHT}) + MSE comp (w={COMP_WEIGHT})"
                 + (f" + entropy (w={ENTROPY_WEIGHT})" if ENTROPY_WEIGHT > 0 else "")],
        ["E (GPa) range", f"{y_enu[:,0].min():.1f}-{y_enu[:,0].max():.1f}"],
        ["nu range", f"{y_enu[:,1].min():.3f}-{y_enu[:,1].max():.3f}"],
    ], tablefmt="simple"))

    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.permutation(len(X_in))
    n_train = int(len(X_in) * TRAIN_RATIO)
    tr, va = idx[:n_train], idx[n_train:]
    X_tr, X_va = X_in[tr], X_in[va]
    yc_tr, yc_va = y_comp[tr], y_comp[va]
    enu_va = y_enu[va]
    print(f"\n  Train: {len(X_tr)} | Val: {len(X_va)}")

    # Drop stale ensemble checkpoints with index >= ENSEMBLE_SIZE.
    if ENSEMBLE_SIZE > 1:
        legacy = os.path.join(_THIS_DIR, "inv_model.keras")
        if os.path.exists(legacy):
            os.remove(legacy)
        for stale in glob.glob(os.path.join(_THIS_DIR, "inv_model_*.keras")):
            try:
                i = int(os.path.basename(stale)[len("inv_model_"):-len(".keras")])
            except ValueError:
                continue
            if i >= ENSEMBLE_SIZE:
                os.remove(stale)

    t0 = time.time()
    last_inv = None
    history = None
    for member in range(ENSEMBLE_SIZE):
        seed = RANDOM_SEED + member
        np.random.seed(seed)
        tf.random.set_seed(seed)
        if ENSEMBLE_SIZE > 1:
            print(f"\n{'#'*70}\n#  ENSEMBLE MEMBER {member+1}/{ENSEMBLE_SIZE} (seed={seed})\n{'#'*70}")

        model = build_training_model(forward)
        if member == 0:
            model.summary(print_fn=lambda s: print(f"  {s}"))

        hist = model.fit(
            X_tr, {"recon": X_tr, "comp": yc_tr},
            validation_data=(X_va, {"recon": X_va, "comp": yc_va}),
            epochs=MAX_EPOCHS,
            batch_size=min(TRAIN_BATCH_SIZE, len(X_tr)),
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True)],
        )
        history = hist.history
        n_ep = len(history["loss"])
        print(f"  trained {n_ep} epochs | "
              f"val_loss={history['val_loss'][-1]:.5f} "
              f"recon={history['val_recon_loss'][-1]:.5f} "
              f"comp={history['val_comp_loss'][-1]:.5f}")

        inv = _inverse_only(model)
        path = (os.path.join(_THIS_DIR, f"inv_model_{member}.keras")
                if ENSEMBLE_SIZE > 1 else os.path.join(_THIS_DIR, "inv_model.keras"))
        inv.save(path)
        print(f"  Saved {os.path.basename(path)}")
        last_inv = inv

    print(f"\n  Training complete in {time.time()-t0:.1f}s")

    # ── Round-trip metrics (uses the last member + frozen forward) ──────
    E_tr, nu_tr, _, _, _ = _roundtrip_enu(last_inv, forward, X_tr)
    E_va, nu_va, _, _, comp_va = _roundtrip_enu(last_inv, forward, X_va)
    y_tr_enu = y_enu[tr]

    train_m = {"E_GPa": _metrics(y_tr_enu[:, 0], E_tr), "nu": _metrics(y_tr_enu[:, 1], nu_tr)}
    val_m = {"E_GPa": _metrics(enu_va[:, 0], E_va), "nu": _metrics(enu_va[:, 1], nu_va)}

    print("\n" + "=" * 70 + "\nRound-trip accuracy  (target -> comp -> forward -> achieved)\n" + "=" * 70)
    for lbl, m in [("Train", train_m), ("Val", val_m)]:
        rows = [[t, f"{m[t]['MAE']:.4f}", f"{m[t]['RMSE']:.4f}",
                 f"{m[t]['R2']:.4f}", f"{m[t]['MAPE']:.2f}%", f"{m[t]['MedAPE']:.2f}%"]
                for t in ("E_GPa", "nu")]
        print(f"\n  {lbl} set:")
        print(tabulate(rows, headers=["Target", "MAE", "RMSE", "R2", "MAPE", "MedAPE"],
                       tablefmt="simple"))

    # Composition reconstruction (secondary): mean L1 over val.
    comp_l1 = float(np.abs(comp_va - yc_va).sum(axis=1).mean())
    print(f"\n  Composition recon (val): mean L1 = {comp_l1:.4f}")

    if HAS_MPL:
        print(f"\n  Generating plots in {plots_dir}/")
        _plot_parity(y_tr_enu, np.column_stack([E_tr, nu_tr]),
                     "Train (round-trip)", "inv_parity_train.png", plots_dir)
        _plot_parity(enu_va, np.column_stack([E_va, nu_va]),
                     "Val (round-trip)", "inv_parity_validation.png", plots_dir)
        if history is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(history["loss"], label="train")
            ax.plot(history["val_loss"], label="val")
            ax.set_xlabel("Epoch"); ax.set_ylabel("total loss"); ax.set_yscale("log")
            ax.set_title("Inverse training loss (last member)")
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "inv_training_history.png"), dpi=150)
            plt.close(fig)

    metrics_out = {
        "n_total": len(X_in),
        "n_train": len(X_tr),
        "n_val": len(X_va),
        "ensemble_size": ENSEMBLE_SIZE,
        "recon_weight": RECON_WEIGHT,
        "comp_weight": COMP_WEIGHT,
        "entropy_weight": ENTROPY_WEIGHT,
        "huber_delta": HUBER_DELTA,
        "nu_filter_max": NU_FILTER_MAX,
        "C_scale": C_SCALE,
        "comp_recon_l1_val": comp_l1,
        "train": train_m,
        "val": val_m,
    }
    with open(os.path.join(_THIS_DIR, "inv_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n  Metrics saved to inv_metrics.json")


if __name__ == "__main__":
    main()
