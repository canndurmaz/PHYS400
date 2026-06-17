"""Flask UI for the inverse alloy-design suggester.

The mirror of ``src/ML/app.py``: where that app takes a composition and
predicts $(E, \\nu)$, this one takes a target $(E, \\nu)$ (plus optional
element constraints) and suggests compositions that achieve it. The
heavy lifting lives in :func:`suggest.suggest`; this module is just the
JSON endpoint + the single-page form.

Run:
    python app.py            # listens on http://127.0.0.1:5002
"""

from __future__ import annotations

import json
import os
import random
import sys

from flask import Flask, jsonify, render_template, request

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "ML"))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from model_constants import ALL_ELEMENTS                # noqa: E402
from suggest import suggest                             # noqa: E402

app = Flask(__name__,
            template_folder=os.path.join(_THIS_DIR, "templates"),
            static_folder=os.path.join(_THIS_DIR, "static"))

_RESULTS_PATH = os.path.join(_ML_DIR, "results.json")
_INV_METRICS_PATH = os.path.join(_THIS_DIR, "inv_metrics.json")
_CLOUD_MAX = 800


# ── (E, nu) ground-truth cloud (shared backdrop with the forward app) ────────
def _load_cloud(max_n: int = _CLOUD_MAX) -> list[dict]:
    if not os.path.exists(_RESULTS_PATH):
        return []
    with open(_RESULTS_PATH) as f:
        data = json.load(f)
    pts = []
    for rec in data.values():
        E, nu = rec.get("E_GPa"), rec.get("nu")
        if not (isinstance(E, (int, float)) and isinstance(nu, (int, float))):
            continue
        if E <= 0 or not (0 < nu < 0.5):
            continue
        pts.append({"E": round(float(E), 3), "nu": round(float(nu), 4)})
    if len(pts) > max_n:
        pts = random.Random(42).sample(pts, max_n)
    return pts


def _load_metrics() -> dict:
    if not os.path.exists(_INV_METRICS_PATH):
        return {}
    with open(_INV_METRICS_PATH) as f:
        return json.load(f)


# mtime-keyed caches (re-read when a fresh training run rewrites the files).
_CLOUD_CACHE = {"mtime": None, "data": []}
_METRICS_CACHE = {"mtime": None, "data": {}}


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _cloud():
    m = _mtime(_RESULTS_PATH)
    if m != _CLOUD_CACHE["mtime"]:
        _CLOUD_CACHE.update(mtime=m, data=_load_cloud())
    return _CLOUD_CACHE["data"]


def _metrics():
    m = _mtime(_INV_METRICS_PATH)
    if m != _METRICS_CACHE["mtime"]:
        _METRICS_CACHE.update(mtime=m, data=_load_metrics())
    return _METRICS_CACHE["data"]


@app.route("/")
def index():
    return render_template("index.html",
                           elements=ALL_ELEMENTS,
                           cloud=_cloud(),
                           metrics=_metrics())


@app.post("/api/suggest")
def api_suggest():
    payload = request.get_json(silent=True) or {}

    def _num(key):
        v = payload.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    target_E = _num("E_GPa")
    target_nu = _num("nu")
    if target_E is None or target_nu is None:
        return jsonify(error="Provide numeric 'E_GPa' and 'nu'."), 400
    if target_E <= 0:
        return jsonify(error="E must be positive."), 400
    if not (0.0 < target_nu < 0.5):
        return jsonify(error="nu must be in (0, 0.5)."), 400

    forbid = [e for e in payload.get("forbid", []) if e in ALL_ELEMENTS]
    require = [e for e in payload.get("require", []) if e in ALL_ELEMENTS]
    overlap = set(forbid) & set(require)
    if overlap:
        return jsonify(error=f"Element(s) both required and forbidden: "
                             f"{', '.join(sorted(overlap))}."), 400

    # Shape constraints -------------------------------------------------------
    max_elements = payload.get("max_elements")
    if max_elements in (None, "", 0, "0"):
        max_elements = None
    else:
        try:
            max_elements = int(max_elements)
        except (TypeError, ValueError):
            return jsonify(error="max_elements must be an integer."), 400
        if max_elements < 1:
            return jsonify(error="max_elements must be >= 1."), 400

    al_dominant = bool(payload.get("al_dominant", False))
    try:
        al_min = float(payload.get("al_min", 0.5))
    except (TypeError, ValueError):
        al_min = 0.5
    if not (0.0 < al_min < 1.0):
        al_min = 0.5

    if al_dominant and "Al" in forbid:
        return jsonify(error="Al-dominant conflicts with forbidding Al."), 400
    min_needed = len(set(require) | ({"Al"} if al_dominant else set()))
    if max_elements is not None and max_elements < min_needed:
        return jsonify(error=f"max_elements={max_elements} is too small for "
                             f"{min_needed} pinned element(s)."), 400

    try:
        k = int(payload.get("k", 5))
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(k, 10))
    refine = bool(payload.get("refine", True))

    try:
        result = suggest(target_E, target_nu, forbid=forbid, require=require,
                         k=k, refine_iters=40 if refine else 0,
                         max_elements=max_elements, al_dominant=al_dominant,
                         al_min=al_min)
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    return jsonify(result)


if __name__ == "__main__":
    # Localhost only; raw model inference, not hardened for the open net.
    # Port 5002 keeps clear of the forward app (5000) and MEAM app (5001).
    app.run(host="127.0.0.1", port=5002, debug=True)
