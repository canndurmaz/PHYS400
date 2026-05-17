"""Flask UI for the alloy elastic-property surrogate.

Wraps :func:`predict_from_model.predict_properties` behind a small JSON
endpoint and serves a single-page form for entering compositions. The model
is loaded lazily on the first prediction request, so ``flask run`` boots
quickly.

Run:
    python app.py            # listens on http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from model_constants import ALL_ELEMENTS  # TF-free constants
from predict_from_model import predict_properties
from uncertainty import (
    classify_ood,
    composition_to_vector,
    knn_distance,
    load_training_X,
    ood_thresholds,
)

# ── Resource root ──────────────────────────────────────────────────────────
# In a normal Python run, data files (templates/static/models/json) live
# next to this file. In a PyInstaller-frozen bundle they're extracted to a
# temp dir whose path is exposed as ``sys._MEIPASS``. Detect that mode and
# resolve everything against the right root.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _RESOURCE_ROOT = sys._MEIPASS  # type: ignore[attr-defined]
else:
    _RESOURCE_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_RESOURCE_ROOT, "templates"),
    static_folder=os.path.join(_RESOURCE_ROOT, "static"),
)

# ── Calibration: load nn_metrics.json so the UI can display the model's
# validation-set MAPE alongside each prediction. The calibration panel and
# the (E, ν) ground-truth cloud are re-read whenever the underlying JSON
# files change on disk (mtime-keyed cache), so re-running ``nn_alloy.py``
# (or ``run_nn.sh``) while the Flask server is up refreshes the panel on
# the next request without a restart. Unchanged files are served from
# cache for free (one stat() syscall, no JSON re-parse).
_THIS_DIR = _RESOURCE_ROOT
_METRICS_PATH = os.path.join(_THIS_DIR, "nn_metrics.json")
# Model path is only used for the "trained on" date stamp in the calibration
# panel. Prefer an ONNX checkpoint (runtime format); fall back to the legacy
# .keras name so the panel still populates in a dev environment that hasn't
# converted yet.
_MODEL_PATH   = next(
    (os.path.join(_THIS_DIR, p) for p in
     ("alloy_model_0.onnx", "alloy_model.onnx",
      "alloy_model_0.keras", "alloy_model.keras")
     if os.path.exists(os.path.join(_THIS_DIR, p))),
    os.path.join(_THIS_DIR, "alloy_model.keras"),
)
_RESULTS_PATH = os.path.join(_THIS_DIR, "results.json")
_CLOUD_MAX    = 800        # rendered SVG dots; ~800 reads dense without lagging


def _load_cloud(max_n: int = _CLOUD_MAX) -> list[dict]:
    """Load (E, ν) ground-truth points for the Ashby-style backdrop.

    Decimates with a fixed seed so the cloud is stable across reloads.
    """
    if not os.path.exists(_RESULTS_PATH):
        return []
    with open(_RESULTS_PATH) as f:
        data = json.load(f)
    pts: list[dict] = []
    for _name, rec in data.items():
        E = rec.get("E_GPa")
        nu = rec.get("nu")
        if not (isinstance(E, (int, float)) and isinstance(nu, (int, float))):
            continue
        if E <= 0 or not (0 < nu < 0.5):
            continue
        pts.append({"E": round(float(E), 3), "nu": round(float(nu), 4)})
    if len(pts) > max_n:
        rng = random.Random(42)        # deterministic so the visual context
        pts = rng.sample(pts, max_n)   # doesn't shuffle on every refresh
    return pts


def _load_calibration() -> dict:
    if not os.path.exists(_METRICS_PATH):
        return {}
    with open(_METRICS_PATH) as f:
        m = json.load(f)
    val   = m.get("val", {})
    train = m.get("train", {})

    def _safe(d: dict, k: str):
        v = d.get(k)
        return v if isinstance(v, (int, float)) else None

    def _stats(d: dict) -> dict:
        return {k: _safe(d, k) for k in (
            "MAE", "RMSE", "MAPE", "R2", "Pearson", "Bias",
            "P95AE", "MedAE", "Std", "true_mean", "true_std",
        )}

    trained_iso = None
    if os.path.exists(_MODEL_PATH):
        ts = os.path.getmtime(_MODEL_PATH)
        trained_iso = datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")

    val_pass = m.get("val_pass")
    val_fail = m.get("val_fail")
    val_pass_pct = (
        100.0 * val_pass / (val_pass + val_fail)
        if isinstance(val_pass, int) and isinstance(val_fail, int) and (val_pass + val_fail) > 0
        else None
    )

    return {
        # corpus
        "n_total": m.get("n_total"),
        "n_train": m.get("n_train"),
        "n_val":   m.get("n_val"),
        "trained": trained_iso,
        # training meta
        "epochs":        m.get("epochs_trained"),
        "attempts":      m.get("attempts"),
        "gate_passed":   m.get("gate_passed"),
        "max_error_pct": m.get("max_error_pct"),
        "huber_delta":   m.get("huber_delta"),
        "nu_filter_max": m.get("nu_filter_max"),
        "c_scale":       m.get("C_scale"),
        "val_pass":      val_pass,
        "val_fail":      val_fail,
        "val_pass_pct":  val_pass_pct,
        # per-target stats (held-out validation set)
        "val": {
            "E":   _stats(val.get("E_GPa", {})),
            "nu":  _stats(val.get("nu",    {})),
            "C11": _stats(val.get("C11",   {})),
            "C12": _stats(val.get("C12",   {})),
        },
        "train": {
            "E":   _stats(train.get("E_GPa", {})),
            "nu":  _stats(train.get("nu",    {})),
            "C11": _stats(train.get("C11",   {})),
            "C12": _stats(train.get("C12",   {})),
        },
        # legacy convenience: short summaries used in the panel hint + footer
        "mape": {
            "E":   _safe(val.get("E_GPa", {}), "MAPE"),
            "nu":  _safe(val.get("nu",    {}), "MAPE"),
            "C11": _safe(val.get("C11",   {}), "MAPE"),
            "C12": _safe(val.get("C12",   {}), "MAPE"),
        },
        "r2": {
            "E":   _safe(val.get("E_GPa", {}), "R2"),
            "nu":  _safe(val.get("nu",    {}), "R2"),
            "C11": _safe(val.get("C11",   {}), "R2"),
            "C12": _safe(val.get("C12",   {}), "R2"),
        },
    }


# mtime-keyed caches: re-parse only when the underlying file is rewritten
# by a fresh training run. Initial values force a load on the first call.
_CAL_CACHE: dict = {"mtime": None, "data": {}}
_CLOUD_CACHE: dict = {"mtime": None, "data": []}
# OOD cache holds the training composition matrix *and* the precomputed
# self-distance percentiles used to classify a query. Both are derived
# from ``results.json``, so a single mtime key suffices.
_OOD_CACHE: dict = {"mtime": None, "X": None, "thresholds": {}}


def _file_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _calibration() -> dict:
    """Return calibration, re-reading ``nn_metrics.json`` on mtime change."""
    mtime = _file_mtime(_METRICS_PATH)
    if mtime != _CAL_CACHE["mtime"]:
        _CAL_CACHE["mtime"] = mtime
        _CAL_CACHE["data"] = _load_calibration()
    return _CAL_CACHE["data"]


def _cloud() -> list[dict]:
    """Return the (E, ν) cloud, re-reading ``results.json`` on mtime change."""
    mtime = _file_mtime(_RESULTS_PATH)
    if mtime != _CLOUD_CACHE["mtime"]:
        _CLOUD_CACHE["mtime"] = mtime
        _CLOUD_CACHE["data"] = _load_cloud()
    return _CLOUD_CACHE["data"]


def _ood() -> dict:
    """Return cached training matrix + OOD thresholds.

    Rebuilds on ``results.json`` mtime change so a fresh active-learning
    round (which appends new compositions to results.json) auto-updates
    the OOD baseline without a server restart. The threshold computation
    is O(N²) in the training set size; for N≲1k it runs in well under
    a second and is amortised across many predictions.
    """
    mtime = _file_mtime(_RESULTS_PATH)
    if mtime != _OOD_CACHE["mtime"]:
        X = load_training_X(_RESULTS_PATH, ALL_ELEMENTS)
        _OOD_CACHE["mtime"] = mtime
        _OOD_CACHE["X"] = X
        _OOD_CACHE["thresholds"] = ood_thresholds(X)
    return _OOD_CACHE


def _uncertainty_for_query(comp_norm: dict[str, float]) -> dict:
    """KNN-distance + OOD bucket for a normalised composition.

    Pure-NumPy and ~sub-millisecond on a few-thousand-row training set,
    so it's safe to call inline from ``/api/predict``. The ensemble σ
    is provided separately by ``predict_properties`` itself; the two
    signals are intentionally independent (data-density vs model-spread).
    """
    cache = _ood()
    X = cache["X"]
    if X is None or X.shape[0] == 0:
        return {"knn_distance": None, "ood_class": "unknown",
                "ood_p50": None, "ood_p95": None, "n_train_ref": 0}
    query = composition_to_vector(comp_norm, ALL_ELEMENTS)
    d = knn_distance(query, X)
    thr = cache["thresholds"]
    return {
        "knn_distance": d,
        "ood_class":    classify_ood(d, thr),
        "ood_p50":      thr.get("p50"),
        "ood_p95":      thr.get("p95"),
        "n_train_ref":  thr.get("n", X.shape[0]),
    }


# Tolerance on the composition sum. MD/DFT inputs always normalise to 1.0,
# but the form lets users type free fractions and we'd rather not reject
# input that's off by floating-point noise. We re-normalise on the server
# side and report the original sum so the UI can warn when it's far off.
_SUM_TOLERANCE = 1e-3


@app.route("/")
def index():
    return render_template(
        "index.html",
        elements=ALL_ELEMENTS,
        cal=_calibration(),
        cloud=_cloud(),
    )


def _sensitivity(comp_norm: dict[str, float]) -> list[dict]:
    """Per-element ±10 % perturbation sensitivity of (E, ν).

    For each element with a positive fraction, scale it by (1 ± 0.10) and
    rebalance the *other* elements proportionally so the composition still
    sums to 1.0. Run inference for both signs and report the deltas relative
    to the unperturbed prediction. Two extra forward passes per element ─
    cheap (~5 ms each post-warmup).
    """
    items = [(el, f) for el, f in comp_norm.items() if f > 0]
    if len(items) < 2:
        return []     # single-element compositions have no meaningful tornado

    base = predict_properties(comp_norm)
    base_E, base_nu = base["E_GPa"], base["nu"]
    delta = 0.10
    rows: list[dict] = []
    for el, x in items:
        sum_others = 1.0 - x
        if sum_others <= 1e-9:
            continue        # this element is essentially the whole alloy
        deltas = {}
        for sign, key in ((+1, "plus"), (-1, "minus")):
            new_x = x * (1.0 + sign * delta)
            new_x = max(1e-6, min(0.999999, new_x))
            scale = (1.0 - new_x) / sum_others
            new_comp = {e: f * scale for e, f in comp_norm.items() if e != el}
            new_comp[el] = new_x
            out = predict_properties(new_comp)
            deltas[f"dE_{key}"]  = out["E_GPa"] - base_E
            deltas[f"dnu_{key}"] = out["nu"]    - base_nu
        rows.append({"element": el, "frac": x, **deltas})

    # Sort by max absolute ΔE descending — the engineer's "what matters most".
    rows.sort(key=lambda r: max(abs(r["dE_plus"]), abs(r["dE_minus"])), reverse=True)
    return rows


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    raw_comp = payload.get("composition", {})
    if not isinstance(raw_comp, dict) or not raw_comp:
        return jsonify(error="Provide a non-empty 'composition' object."), 400

    # Coerce to float, skip blanks, reject negatives. Unknown symbols are
    # left in the dict and surfaced back through 'unknown_elements'.
    comp: dict[str, float] = {}
    for el, val in raw_comp.items():
        if val in (None, "", "null"):
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            return jsonify(error=f"Fraction for {el} is not a number: {val!r}"), 400
        if f < 0:
            return jsonify(error=f"Fraction for {el} is negative: {f}"), 400
        if f > 0:
            comp[el] = f

    if not comp:
        return jsonify(error="At least one element must have a positive fraction."), 400

    total = sum(comp.values())
    if total <= 0:
        return jsonify(error="Composition sums to zero."), 400

    # Re-normalise so the model sees a proper mole-fraction vector even if
    # the user typed weights that don't sum to 1.
    normalised = {el: f / total for el, f in comp.items()}

    try:
        out = predict_properties(normalised)
        out["sensitivity"] = _sensitivity(normalised)
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    # Attach the two composition-dependent uncertainty signals:
    #   - ``ensemble_size`` + ``*_std`` (from predict_properties): how much
    #     do independently trained ensemble members disagree at this point?
    #   - ``uncertainty`` block (below): how far is this composition from
    #     anything we have training data for?
    # The two are complementary — an ensemble can be confidently wrong in
    # a sparsely sampled region, and a known region can still have
    # genuinely noisy ground truth.
    out["uncertainty"] = _uncertainty_for_query(normalised)
    out["input_sum"] = total
    out["normalised_composition"] = normalised
    return jsonify(out)


if __name__ == "__main__":
    # ``debug=True`` enables auto-reload on file save. Bind to localhost only;
    # this app exposes raw model inference and isn't hardened for the open net.
    app.run(host="127.0.0.1", port=5000, debug=True)
