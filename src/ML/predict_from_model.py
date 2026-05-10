"""Standalone predictor for the alloy elastic-property surrogate.

The trained model in ``alloy_model.keras`` outputs *scaled* cubic elastic
constants $(C_{11}, C_{12})$ -- not $(E, \\nu)$ directly. We undo the training
scale and recover $(E, \\nu)$ analytically under cubic isotropy.

Reuses the constants and conversion from ``nn_alloy.py`` so this file stays in
sync with whatever the training script saves.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Mapping

import numpy as np
import tensorflow as tf

from nn_alloy import ALL_ELEMENTS, C_SCALE, _cij_to_e_nu

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_THIS_DIR, "alloy_model.keras")
_MODEL = None  # lazy-loaded singleton


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


def _get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {_MODEL_PATH}. "
                "Run 'src/ML/run_nn.sh' first to train and save the model."
            )
        _MODEL = tf.keras.models.load_model(_MODEL_PATH)
    return _MODEL


def predict_properties(composition: Mapping[str, float]) -> dict:
    """Predict $(E, \\nu, C_{11}, C_{12})$ for a single composition.

    Parameters
    ----------
    composition : mapping of element symbol -> mole fraction.

    Returns
    -------
    dict with keys ``E_GPa``, ``nu``, ``C11_GPa``, ``C12_GPa``, plus
    ``unknown_elements`` listing any symbols outside the trained 10-element
    basis (those are silently treated as 0 fraction).
    """
    model = _get_model()
    features, unknown = _composition_to_features(composition)
    # ``model(X, training=False)`` is ~20x faster than ``model.predict`` for
    # single-row inputs (skips the dataset/batching pipeline).
    scaled = model(features, training=False).numpy()
    C11, C12 = (scaled[0] * C_SCALE).tolist()
    E, nu = _cij_to_e_nu(np.array([C11]), np.array([C12]))
    return {
        "E_GPa": float(E[0]),
        "nu": float(nu[0]),
        "C11_GPa": float(C11),
        "C12_GPa": float(C12),
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
    print("\n" + "=" * 50)
    print(f"{'Prediction for ' + label:^50}")
    print("=" * 50)
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
