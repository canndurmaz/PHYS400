"""Standalone predictor for the alloy elastic-property surrogate.

Loads ensemble checkpoints (one or many) and returns per-target mean ± σ
for $(E, \\nu, C_{11}, C_{12})$ over a single composition. The network
outputs *scaled* $(C_{11}, C_{12})$; the analytical conversion to
$(E, \\nu)$ uses :func:`model_constants.cij_to_e_nu`.

Backend selection (transparent to callers):

* ``alloy_model*.onnx`` is preferred when present. This is the
  TF-free runtime used by the packaged desktop app — only
  ``onnxruntime`` (~30 MB) is required, not TensorFlow (~500 MB).
* ``alloy_model*.keras`` is loaded via TensorFlow as a development
  fallback when no ``.onnx`` files exist (e.g. fresh training, before
  ``convert_to_onnx.py`` has been run).

The runtime checks for the ``.onnx`` family first to avoid importing TF
on the happy path. With both formats present, ONNX wins.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Mapping

import numpy as np

from model_constants import ALL_ELEMENTS, C_SCALE, cij_to_e_nu

# When frozen by PyInstaller, data files live under ``sys._MEIPASS`` rather
# than next to this source file. Resolve the bundle root the same way
# ``app.py`` does so both find the same checkpoints.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _THIS_DIR = sys._MEIPASS  # type: ignore[attr-defined]
else:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy-loaded list of callables: each takes a (1, n_elements) float32 array
# and returns a (1, 2) float32 array of scaled (C11, C12). One callable per
# ensemble member, in filename-sorted order.
_PREDICTORS: list = []
_BACKEND: str | None = None  # "onnx" or "keras"


def _composition_to_features(comp: Mapping[str, float]) -> tuple[np.ndarray, list[str]]:
    """Pack a {symbol: fraction} dict into the model's 10-d input vector.

    Returns the feature row and a list of any unsupported symbols (so callers
    can surface a warning instead of silently zeroing them).
    """
    features = np.zeros((1, len(ALL_ELEMENTS)), dtype=np.float32)
    unknown: list[str] = []
    for el, frac in comp.items():
        if el in ALL_ELEMENTS:
            features[0, ALL_ELEMENTS.index(el)] = float(frac)
        else:
            unknown.append(el)
    return features, unknown


def _discover_paths(suffix: str) -> list[str]:
    """Return ensemble checkpoint paths for one extension, deterministically.

    Prefers the numbered ``alloy_model_*<suffix>`` pattern; falls back to
    the legacy single ``alloy_model<suffix>``. Returns ``[]`` if neither
    is on disk.
    """
    ensemble = sorted(glob.glob(os.path.join(_THIS_DIR, f"alloy_model_*{suffix}")))
    if ensemble:
        return ensemble
    legacy = os.path.join(_THIS_DIR, f"alloy_model{suffix}")
    return [legacy] if os.path.exists(legacy) else []


def _load_onnx_predictors(paths: list[str]) -> list:
    """Wrap each .onnx checkpoint in a closure that runs inference."""
    import onnxruntime as ort
    # ``CPUExecutionProvider`` is the only one we need (no GPU in the
    # packaged app); naming it explicitly silences ORT's provider warning.
    sessions = [
        ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in paths
    ]

    def _make_caller(sess):
        input_name = sess.get_inputs()[0].name
        # The ONNX graph emits exactly one output (scaled C_ij); using
        # ``run(None, ...)`` returns the list of all outputs, then index 0.
        def _call(x: np.ndarray) -> np.ndarray:
            return sess.run(None, {input_name: x})[0]
        return _call

    return [_make_caller(s) for s in sessions]


def _load_keras_predictors(paths: list[str]) -> list:
    """Dev fallback: wrap each .keras checkpoint via TF, same call shape."""
    import tensorflow as tf
    models = [tf.keras.models.load_model(p) for p in paths]

    def _make_caller(m):
        def _call(x: np.ndarray) -> np.ndarray:
            return m(x, training=False).numpy()
        return _call

    return [_make_caller(m) for m in models]


def _get_predictors() -> tuple[list, str]:
    """Lazy-load all ensemble members; return (callables, backend tag)."""
    global _PREDICTORS, _BACKEND
    if _PREDICTORS:
        return _PREDICTORS, _BACKEND  # type: ignore[return-value]

    onnx_paths = _discover_paths(".onnx")
    if onnx_paths:
        _PREDICTORS = _load_onnx_predictors(onnx_paths)
        _BACKEND = "onnx"
        return _PREDICTORS, _BACKEND

    keras_paths = _discover_paths(".keras")
    if keras_paths:
        _PREDICTORS = _load_keras_predictors(keras_paths)
        _BACKEND = "keras"
        return _PREDICTORS, _BACKEND

    raise FileNotFoundError(
        f"No model checkpoints found in {_THIS_DIR}. "
        "Run 'src/ML/run_nn.sh' first to train, then "
        "'python convert_to_onnx.py' to produce the runtime ONNX bundle."
    )


def predict_properties(composition: Mapping[str, float]) -> dict:
    """Predict $(E, \\nu, C_{11}, C_{12})$ with per-target ensemble σ.

    Parameters
    ----------
    composition : mapping of element symbol -> mole fraction.

    Returns
    -------
    dict with the per-target mean (``E_GPa``, ``nu``, ``C11_GPa``,
    ``C12_GPa``) and the sample std across ensemble members
    (``*_std``). ``ensemble_size`` is the number of checkpoints found;
    when 1, every std is exactly 0. ``backend`` indicates which runtime
    served the prediction (``"onnx"`` or ``"keras"``).
    """
    predictors, backend = _get_predictors()
    features, unknown = _composition_to_features(composition)

    # Propagate each member's (C11, C12) through the nonlinear (E, ν)
    # conversion *independently* before aggregating. Averaging in C-space
    # first would bias σ_E and σ_ν near the (1-2ν)=0 singularity.
    C11_samples = np.empty(len(predictors))
    C12_samples = np.empty(len(predictors))
    for i, call in enumerate(predictors):
        scaled = call(features)
        C11_samples[i] = scaled[0, 0] * C_SCALE
        C12_samples[i] = scaled[0, 1] * C_SCALE
    E_samples, nu_samples = cij_to_e_nu(C11_samples, C12_samples)

    def _ms(arr: np.ndarray) -> tuple[float, float]:
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return mean, std

    E_mean,   E_std   = _ms(E_samples)
    nu_mean,  nu_std  = _ms(nu_samples)
    C11_mean, C11_std = _ms(C11_samples)
    C12_mean, C12_std = _ms(C12_samples)

    return {
        "E_GPa":   E_mean,   "E_GPa_std":   E_std,
        "nu":      nu_mean,  "nu_std":      nu_std,
        "C11_GPa": C11_mean, "C11_GPa_std": C11_std,
        "C12_GPa": C12_mean, "C12_GPa_std": C12_std,
        "ensemble_size":    len(predictors),
        "backend":          backend,
        "unknown_elements": unknown,
    }


def predict_from_json(config_path: str) -> None:
    """CLI entry point: read a JSON file and print predictions to stdout.

    Accepts either ``{"composition": {...}}`` or a bare ``{...}`` dict, and
    additionally supports the ``predict.json`` batch format
    ``{"name1": {...}, "name2": {...}}``.
    """
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    # Detect batch format: top-level values are themselves dicts of fractions.
    is_batch = (
        isinstance(cfg, dict)
        and cfg
        and all(isinstance(v, dict) for v in cfg.values())
        and "composition" not in cfg
        and not any(k in ALL_ELEMENTS for k in cfg)
    )

    if is_batch:
        for name, comp in cfg.items():
            _print_block(name, comp)
    else:
        comp = cfg.get("composition", cfg) if isinstance(cfg, dict) else cfg
        _print_block(os.path.basename(config_path), comp)


def _print_block(label: str, comp: Mapping[str, float]) -> None:
    out = predict_properties(comp)
    for el in out["unknown_elements"]:
        print(f"Warning: Element {el} not supported by the model.")
    n = out["ensemble_size"]
    print("\n" + "=" * 50)
    print(f"{'Prediction for ' + label:^50}")
    if n > 1:
        print(f"{'(ensemble of ' + str(n) + ' members, mean ± σ)':^50}")
    print(f"{'[backend: ' + out['backend'] + ']':^50}")
    print("=" * 50)
    if n > 1:
        print(f"  Young's Modulus (E):  {out['E_GPa']:8.2f} ± {out['E_GPa_std']:.2f} GPa")
        print(f"  Poisson's Ratio (nu): {out['nu']:8.3f} ± {out['nu_std']:.3f}")
        print(f"  C11:                  {out['C11_GPa']:8.2f} ± {out['C11_GPa_std']:.2f} GPa")
        print(f"  C12:                  {out['C12_GPa']:8.2f} ± {out['C12_GPa_std']:.2f} GPa")
    else:
        print(f"  Young's Modulus (E):  {out['E_GPa']:8.2f} GPa")
        print(f"  Poisson's Ratio (nu): {out['nu']:8.3f}")
        print(f"  C11:                  {out['C11_GPa']:8.2f} GPa")
        print(f"  C12:                  {out['C12_GPa']:8.2f} GPa")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_from_model.py <path_to_composition.json>")
    else:
        predict_from_json(sys.argv[1])
