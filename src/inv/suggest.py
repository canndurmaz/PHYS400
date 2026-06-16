"""Runtime inverse-design suggester.

Given a target $(E, \\nu)$ (and optional element constraints), produce a
ranked list of candidate compositions. Mirrors
``src/ML/predict_from_model.py`` in spirit: load the trained ensemble,
run it, and return structured results -- but here the network is the
*inverse* net and every candidate is validated against the *forward*
model before it is reported.

Pipeline per query
------------------
1. Convert the target $(E,\\nu)$ to scaled $(C_{11}, C_{12})$.
2. Run every inverse-ensemble member -> N diverse candidate compositions.
3. Add **retrieval** candidates: the nearest real training alloys in
   $(E,\\nu)$ space (guaranteed physical; a safety net when the net
   extrapolates).
4. Apply element constraints (forbid -> 0, require -> floored), renormalise.
5. Optionally **refine** each net candidate with a few gradient-free
   hill-climbing steps that reduce the forward $(E,\\nu)$ error.
6. **Forward-check** every candidate through the ML ensemble
   (``predict_properties``): achieved $(E,\\nu,C_{11},C_{12})$ + ensemble σ.
7. De-duplicate near-identical compositions and rank by $(E,\\nu)$ error.

The forward model is ML's trained ensemble -- a single source of truth
for the composition->property map (no duplicate forward training here).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Mapping, Sequence

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "ML"))
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from model_constants import ALL_ELEMENTS, C_SCALE, cij_to_e_nu  # noqa: E402
from predict_from_model import predict_properties               # noqa: E402

_RESULTS_PATH = os.path.join(_ML_DIR, "results.json")

# Drop composition components below this fraction before reporting -- the
# softmax never emits exact zeros, and a 0.3% trace element is noise, not a
# design choice.
_TRACE_FLOOR = 0.005
# Floor assigned to a *required* element the net left empty, before renorm.
_REQUIRE_FLOOR = 0.02
# Two candidates whose composition vectors are within this L1 distance are
# considered the same alloy (keep the more accurate one).
_DEDUP_L1 = 0.04


# ── Inverse-ensemble loading ─────────────────────────────────────────────────
_INV_MODELS: list = []


def _discover_inv_paths() -> list[str]:
    ens = sorted(glob.glob(os.path.join(_THIS_DIR, "inv_model_*.keras")))
    if ens:
        return ens
    legacy = os.path.join(_THIS_DIR, "inv_model.keras")
    return [legacy] if os.path.exists(legacy) else []


def _get_inv_models() -> list:
    """Lazy-load the inverse ensemble (Keras; no ONNX export for this tool)."""
    global _INV_MODELS
    if _INV_MODELS:
        return _INV_MODELS
    paths = _discover_inv_paths()
    if not paths:
        raise FileNotFoundError(
            f"No inverse checkpoints found in {_THIS_DIR}. "
            "Train first: 'src/inv/run_inv.sh'."
        )
    import tensorflow as tf
    _INV_MODELS = [tf.keras.models.load_model(p) for p in paths]
    return _INV_MODELS


# ── Composition helpers ──────────────────────────────────────────────────────
def _e_nu_to_cij_scaled(E: float, nu: float) -> np.ndarray:
    s = E / ((1.0 - 2.0 * nu) * (1.0 + nu))
    C11, C12 = s * (1.0 - nu), s * nu
    return np.array([[C11 / C_SCALE, C12 / C_SCALE]], dtype=np.float32)


def _vec_to_comp(vec: np.ndarray) -> dict[str, float]:
    """Sparse {symbol: fraction} dict from a 12-d vector (drop traces, renorm)."""
    keep = {ALL_ELEMENTS[i]: float(v) for i, v in enumerate(vec) if v >= _TRACE_FLOOR}
    tot = sum(keep.values())
    if tot <= 0:                      # degenerate -- fall back to the argmax
        i = int(np.argmax(vec))
        return {ALL_ELEMENTS[i]: 1.0}
    return {el: round(f / tot, 4) for el, f in keep.items()}


def _apply_constraints(comp: Mapping[str, float],
                       forbid: Sequence[str], require: Sequence[str]) -> dict[str, float]:
    out = {el: float(f) for el, f in comp.items() if el not in forbid}
    for el in require:
        if out.get(el, 0.0) < _REQUIRE_FLOOR:
            out[el] = _REQUIRE_FLOOR
    tot = sum(out.values())
    if tot <= 0:
        return {require[0]: 1.0} if require else {}
    return {el: f / tot for el, f in out.items()}


def _comp_key(comp: Mapping[str, float]) -> tuple:
    """A coarse hashable signature for de-duplication."""
    return tuple(sorted((el, round(f, 2)) for el, f in comp.items() if f >= _TRACE_FLOOR))


def _comp_vec(comp: Mapping[str, float]) -> np.ndarray:
    v = np.zeros(len(ALL_ELEMENTS))
    for el, f in comp.items():
        if el in ALL_ELEMENTS:
            v[ALL_ELEMENTS.index(el)] = f
    return v


# ── Training (E, nu) corpus for retrieval + achievability ────────────────────
_CORPUS: dict = {"mtime": None, "E": None, "nu": None, "comp": None}


def _load_corpus() -> dict:
    try:
        mtime = os.path.getmtime(_RESULTS_PATH)
    except OSError:
        return {"E": np.empty(0), "nu": np.empty(0), "comp": []}
    if mtime == _CORPUS["mtime"]:
        return _CORPUS
    with open(_RESULTS_PATH) as f:
        data = json.load(f)
    E, nu, comp = [], [], []
    for rec in data.values():
        e, n = rec.get("E_GPa"), rec.get("nu")
        c = rec.get("composition")
        if not (isinstance(e, (int, float)) and isinstance(n, (int, float)) and isinstance(c, dict)):
            continue
        if e <= 0 or not (0 < n < 0.5):
            continue
        E.append(float(e)); nu.append(float(n)); comp.append(c)
    _CORPUS.update(mtime=mtime, E=np.asarray(E), nu=np.asarray(nu), comp=comp)
    return _CORPUS


def _retrieval_candidates(target_E: float, target_nu: float,
                          forbid: Sequence[str], require: Sequence[str],
                          n: int) -> list[dict[str, float]]:
    """Nearest real training alloys in normalised (E, nu) space."""
    corpus = _load_corpus()
    if corpus["E"].size == 0:
        return []
    # Normalise the two axes by their spread so neither dominates the metric.
    e_sd = max(corpus["E"].std(), 1e-6)
    nu_sd = max(corpus["nu"].std(), 1e-6)
    d = np.sqrt(((corpus["E"] - target_E) / e_sd) ** 2
                + ((corpus["nu"] - target_nu) / nu_sd) ** 2)
    out: list[dict[str, float]] = []
    for i in np.argsort(d):
        c = corpus["comp"][i]
        if any(c.get(el, 0.0) > 0 for el in forbid):
            continue
        if not all(c.get(el, 0.0) > 0 for el in require):
            continue
        tot = sum(c.values())
        if tot <= 0:
            continue
        out.append({el: v / tot for el, v in c.items() if v > 0})
        if len(out) >= n:
            break
    return out


def _achievability(target_E: float, target_nu: float) -> dict:
    """How close the target sits to the training (E, nu) cloud."""
    corpus = _load_corpus()
    if corpus["E"].size == 0:
        return {"class": "unknown", "nearest_dist": None}
    e_sd = max(corpus["E"].std(), 1e-6)
    nu_sd = max(corpus["nu"].std(), 1e-6)
    d = np.sqrt(((corpus["E"] - target_E) / e_sd) ** 2
                + ((corpus["nu"] - target_nu) / nu_sd) ** 2)
    nearest = float(d.min())
    # Thresholds in normalised-σ units: within ~0.5σ is well-sampled.
    cls = "in" if nearest <= 0.5 else ("edge" if nearest <= 1.5 else "out")
    in_range = (corpus["E"].min() <= target_E <= corpus["E"].max()
                and corpus["nu"].min() <= target_nu <= corpus["nu"].max())
    return {"class": cls, "nearest_dist": round(nearest, 3), "in_bbox": bool(in_range)}


# ── Forward check + scoring ──────────────────────────────────────────────────
def _score(comp: Mapping[str, float], target_E: float, target_nu: float) -> dict:
    out = predict_properties(comp)
    e_err = abs(out["E_GPa"] - target_E) / max(abs(target_E), 1e-8) * 100
    nu_err = abs(out["nu"] - target_nu) / max(abs(target_nu), 1e-8) * 100
    return {
        "composition": {el: round(float(f), 4) for el, f in comp.items()},
        "E_GPa": out["E_GPa"], "nu": out["nu"],
        "C11_GPa": out["C11_GPa"], "C12_GPa": out["C12_GPa"],
        "E_GPa_std": out["E_GPa_std"], "nu_std": out["nu_std"],
        "e_err_pct": round(e_err, 2), "nu_err_pct": round(nu_err, 2),
        "score": round(e_err + nu_err, 3),
        "ensemble_size": out["ensemble_size"],
    }


def _refine(comp: Mapping[str, float], target_E: float, target_nu: float,
            forbid: Sequence[str], require: Sequence[str],
            iters: int) -> dict:
    """Gradient-free hill-climb: shift mass between two elements, keep if the
    forward (E, nu) error drops. Cheap polish on a net proposal."""
    best = _score(comp, target_E, target_nu)
    if iters <= 0:
        return best
    els = [e for e in ALL_ELEMENTS if e not in forbid]
    rng = np.random.RandomState(abs(hash(_comp_key(comp))) % (2 ** 31))
    cur = dict(best["composition"])
    cur_score = best["score"]
    for _ in range(iters):
        present = [e for e in cur if cur[e] > _TRACE_FLOOR]
        if not present or not els:
            break
        donor = present[rng.randint(len(present))]
        recv = els[rng.randint(len(els))]
        if donor == recv or donor in require and cur[donor] <= _REQUIRE_FLOOR + 1e-6:
            continue
        step = cur[donor] * rng.uniform(0.1, 0.5)
        trial = dict(cur)
        trial[donor] = cur[donor] - step
        trial[recv] = cur.get(recv, 0.0) + step
        trial = {e: f for e, f in trial.items() if f > 1e-6}
        trial = _apply_constraints(trial, forbid, require)
        cand = _score(trial, target_E, target_nu)
        if cand["score"] < cur_score:
            cur, cur_score, best = trial, cand["score"], cand
    return best


# ── Public API ───────────────────────────────────────────────────────────────
def suggest(target_E: float, target_nu: float,
            forbid: Sequence[str] | None = None,
            require: Sequence[str] | None = None,
            k: int = 5, refine_iters: int = 40) -> dict:
    """Return ranked candidate compositions for a target (E, nu)."""
    forbid = [e for e in (forbid or []) if e in ALL_ELEMENTS]
    require = [e for e in (require or []) if e in ALL_ELEMENTS and e not in forbid]

    target = _e_nu_to_cij_scaled(target_E, target_nu)

    # 1) inverse-net proposals (one per ensemble member -> diverse)
    raw: list[dict[str, float]] = []
    for m in _get_inv_models():
        vec = m(target, training=False).numpy()[0]
        comp = _apply_constraints(_vec_to_comp(vec), forbid, require)
        if comp:
            raw.append(comp)

    # 2) retrieval proposals from the real training corpus
    retrieved = _retrieval_candidates(target_E, target_nu, forbid, require, n=k)

    # 3) score everything (refine only the net proposals -- retrieved alloys
    #    are already real and shouldn't be perturbed off the data manifold)
    scored: list[dict] = []
    for comp in raw:
        r = _refine(comp, target_E, target_nu, forbid, require, refine_iters)
        r["source"] = "inverse-net"
        scored.append(r)
    for comp in retrieved:
        r = _score(comp, target_E, target_nu)
        r["source"] = "retrieval"
        scored.append(r)

    # 4) de-dup (keep the lowest score per coarse signature) then rank
    by_key: dict[tuple, dict] = {}
    for r in sorted(scored, key=lambda x: x["score"]):
        key = _comp_key(r["composition"])
        if key in by_key:
            continue
        # also drop if L1-close to an already-kept candidate
        vec = _comp_vec(r["composition"])
        if any(np.abs(vec - _comp_vec(kept["composition"])).sum() < _DEDUP_L1
               for kept in by_key.values()):
            continue
        by_key[key] = r
    ranked = sorted(by_key.values(), key=lambda x: x["score"])[:k]

    return {
        "target": {"E_GPa": target_E, "nu": target_nu},
        "constraints": {"forbid": forbid, "require": require},
        "achievability": _achievability(target_E, target_nu),
        "candidates": ranked,
        "n_evaluated": len(scored),
    }


def _cli():
    if len(sys.argv) < 3:
        print("Usage: python suggest.py <E_GPa> <nu> [forbid=El,..] [require=El,..]")
        return
    E, nu = float(sys.argv[1]), float(sys.argv[2])
    forbid, require = [], []
    for tok in sys.argv[3:]:
        if tok.startswith("forbid="):
            forbid = tok.split("=", 1)[1].split(",")
        elif tok.startswith("require="):
            require = tok.split("=", 1)[1].split(",")
    res = suggest(E, nu, forbid=forbid, require=require)
    a = res["achievability"]
    print("\n" + "=" * 60)
    print(f"  Target: E = {E:.1f} GPa, nu = {nu:.3f}")
    print(f"  Achievability: {a['class']} (nearest training dist={a['nearest_dist']})")
    print("=" * 60)
    for i, c in enumerate(res["candidates"], 1):
        formula = " ".join(f"{el}{f:.3f}" for el, f in
                           sorted(c["composition"].items(), key=lambda x: -x[1]))
        print(f"\n  #{i} [{c['source']}]  score={c['score']}")
        print(f"     {formula}")
        print(f"     -> E={c['E_GPa']:.1f} GPa (err {c['e_err_pct']}%), "
              f"nu={c['nu']:.3f} (err {c['nu_err_pct']}%)")
    print()


if __name__ == "__main__":
    _cli()
