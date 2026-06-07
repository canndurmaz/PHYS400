"""Compare ML-surrogate and optimized-MEAM predictions to ASM/MatWeb data.

For each seed material this:
  1. scrapes the literature elastic data + composition (``matweb_scraper``),
  2. runs the ML surrogate (``src/ML/predict_from_model.predict_properties`` —
     instant ONNX/Keras forward pass),
  3. runs a real optimized-MEAM LAMMPS strain sweep (``src/MEAM/runner.py`` as a
     subprocess, exactly as the MEAM web app does), unless ``--no-meam``,
  4. computes absolute + percent error vs the literature E and ν,
and writes everything to ``literature_comparison.json`` with an aggregate
MAE/MAPE summary for each model.

Run:
    ./phys/bin/python3 src/literature/compare.py            # ML + MEAM
    ./phys/bin/python3 src/literature/compare.py --no-meam  # fast, ML only
    ./phys/bin/python3 src/literature/compare.py --refresh  # re-pull pages
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional

# ── Repo-relative imports ───────────────────────────────────────────────────
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)                       # materials, matweb_scraper
sys.path.insert(0, os.path.join(_REPO, "src", "ML"))  # predict_from_model

import materials                                  # noqa: E402
import matweb_scraper as scraper                  # noqa: E402

_OUT_PATH = os.path.join(_THIS, "literature_comparison.json")
_MEAM_RUNNER = os.path.join(_REPO, "src", "MEAM", "runner.py")
_PY = os.path.join(_REPO, "phys", "bin", "python3")

# MEAM knobs. A 2.5 nm cubic box is a real supercell (a few thousand atoms)
# that keeps the per-material LAMMPS minimise to ~10–30 s; bump --box for a
# larger, slower, less noisy cell. Mirrors src/MD/config.py's knob schema.
_DEFAULT_KNOBS = {
    "box_size_m": 2.5e-9,
    "temperature": 300.0,
    "total_steps": 1000,
    "thermo_interval": 10,
    "dump_interval": 50,
}


def _pct_err(pred: Optional[float], ref: Optional[float]) -> Optional[float]:
    if pred is None or ref is None or ref == 0:
        return None
    return 100.0 * (pred - ref) / ref


def _errors(pred: dict, lit: dict) -> dict:
    """Abs + pct error of a prediction block against the literature block."""
    out: dict = {}
    for key in ("E_GPa", "nu"):
        p, r = pred.get(key), lit.get(key)
        out[key] = {
            "pred": p,
            "lit": r,
            "abs_err": (p - r) if (p is not None and r is not None) else None,
            "pct_err": _pct_err(p, r),
        }
    return out


def run_ml(composition: dict[str, float]) -> dict:
    """ML surrogate prediction (lazy import so --no-meam users still need it)."""
    from predict_from_model import predict_properties
    out = predict_properties(composition)
    return {
        "E_GPa": out["E_GPa"], "nu": out["nu"],
        "C11_GPa": out["C11_GPa"], "C12_GPa": out["C12_GPa"],
        "E_GPa_std": out.get("E_GPa_std"), "nu_std": out.get("nu_std"),
        "ensemble_size": out.get("ensemble_size"),
        "backend": out.get("backend"),
    }


def run_meam(composition: dict[str, float], knobs: dict) -> dict:
    """Run one optimized-MEAM LAMMPS sweep via the MEAM app's runner subprocess.

    Speaks the same JSON-line stdin/stdout protocol as src/MEAM/app.py: we
    write the spec to stdin and read newline-delimited progress objects,
    keeping only the terminal ``result`` (or ``error``).
    """
    spec = {"composition": composition, "knobs": knobs,
            "do_viz": False, "render_output_path": None}
    env = {**os.environ,
           "LD_LIBRARY_PATH": os.path.expanduser("~/.local/lib")
                              + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
    proc = subprocess.run(
        [_PY, _MEAM_RUNNER], input=json.dumps(spec),
        capture_output=True, text=True, env=env,
    )
    result: Optional[dict] = None
    error: Optional[dict] = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result":
            result = obj
        elif obj.get("type") == "error":
            error = obj
    if result is not None:
        return {k: result.get(k) for k in
                ("E_GPa", "nu", "C11_GPa", "C12_GPa",
                 "physical", "physical_reason")}
    msg = (error or {}).get("message", f"runner exited {proc.returncode}")
    return {"error": msg}


def _meam_is_valid(rec: dict) -> bool:
    """True if a record's MEAM prediction is present and physically sound.

    Unphysical sweeps (negative E/ν, C11<C12 → ``physical: False``) and errored
    runs are excluded from MEAM aggregates: a -99 GPa "modulus" would otherwise
    poison the MAPE. They remain in the per-material records, flagged, so the
    failure is still visible.
    """
    meam = rec.get("predictions", {}).get("MEAM", {})
    return "error" not in meam and meam.get("physical", True)


def _aggregate(records: list[dict], model: str) -> dict:
    """MAE / MAPE / signed-bias over materials that have both pred & lit.

    For MEAM, only physically valid runs contribute (see ``_meam_is_valid``).
    """
    if model == "MEAM":
        records = [r for r in records if _meam_is_valid(r)]
    stats: dict = {}
    for key in ("E_GPa", "nu"):
        abs_errs, pct_errs = [], []
        for rec in records:
            blk = rec.get("comparison", {}).get(model, {}).get(key)
            if not blk:
                continue
            if blk["abs_err"] is not None:
                abs_errs.append(blk["abs_err"])
            if blk["pct_err"] is not None:
                pct_errs.append(blk["pct_err"])
        n = len(pct_errs)
        stats[key] = {
            "n": n,
            "MAE": (sum(abs(x) for x in abs_errs) / len(abs_errs)) if abs_errs else None,
            "MAPE": (sum(abs(x) for x in pct_errs) / n) if n else None,
            "bias_pct": (sum(pct_errs) / n) if n else None,
        }
    return stats


def _aggregate_block(records: list[dict], no_meam: bool) -> dict:
    """Overall + per-family aggregates for every active model."""
    models = ["ML"] if no_meam else ["ML", "MEAM"]
    families = sorted({r["family"] for r in records})
    block: dict = {}
    for model in models:
        block[model] = {
            "overall": _aggregate(records, model),
            "by_family": {
                fam: _aggregate([r for r in records if r["family"] == fam], model)
                for fam in families
            },
        }
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-meam", action="store_true",
                    help="skip the LAMMPS sweeps (ML only; fast)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull datasheets instead of using the HTML cache")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N materials")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="skip materials whose basis coverage is below this "
                         "(0–1); out-of-basis mass is dropped before modelling")
    ap.add_argument("--box", type=float, default=_DEFAULT_KNOBS["box_size_m"],
                    help="MEAM cubic box edge in metres (default 2.5e-9)")
    args = ap.parse_args()

    knobs = {**_DEFAULT_KNOBS, "box_size_m": args.box}
    ids = materials.all_ids()
    if args.limit:
        ids = ids[: args.limit]

    records: list[dict] = []
    skipped: list[dict] = []
    print(f"Materials to process: {len(ids)} "
          f"(MEAM: {'off' if args.no_meam else 'on'})\n")

    for i, bid in enumerate(ids, 1):
        try:
            mat = scraper.scrape(bid, refresh=args.refresh)
        except Exception as exc:                      # network / parse failure
            skipped.append({"bassnum": bid, "reason": f"scrape error: {exc}"})
            print(f"[{i:>2}/{len(ids)}] {bid:<11} SKIP (scrape error: {exc})")
            continue
        if mat is None:
            skipped.append({"bassnum": bid, "reason": "no elastic/composition data"})
            print(f"[{i:>2}/{len(ids)}] {bid:<11} SKIP (no usable data)")
            continue
        if mat["basis_coverage"] < args.min_coverage:
            skipped.append({"bassnum": bid, "reason":
                            f"coverage {mat['basis_coverage']:.2f} < "
                            f"{args.min_coverage:.2f}"})
            print(f"[{i:>2}/{len(ids)}] {bid:<11} SKIP (low coverage "
                  f"{mat['basis_coverage']:.2f})")
            continue

        comp, lit = mat["composition"], mat["lit"]
        ml = run_ml(comp)
        comparison = {"ML": _errors(ml, lit)}
        preds = {"ML": ml}

        meam_note = ""
        if not args.no_meam:
            meam = run_meam(comp, knobs)
            preds["MEAM"] = meam
            if "error" not in meam:
                comparison["MEAM"] = _errors(meam, lit)
                meam_note = (f" | MEAM E={meam['E_GPa']:.0f}"
                             if meam.get("E_GPa") is not None else "")
            else:
                meam_note = f" | MEAM ERR: {meam['error'][:40]}"

        records.append({**mat, "family": materials.family_of(bid),
                        "predictions": preds, "comparison": comparison})
        e_ml = comparison["ML"]["E_GPa"]["pct_err"]
        print(f"[{i:>2}/{len(ids)}] {bid:<11} {mat['name'][:30]:<30} "
              f"litE={lit['E_GPa']:.0f} mlE={ml['E_GPa']:.0f} "
              f"(ML ΔE={e_ml:+.0f}%){meam_note}")

    summary = _aggregate_block(records, args.no_meam)
    meam_unphysical = (
        0 if args.no_meam
        else sum(1 for r in records if not _meam_is_valid(r))
    )
    payload = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "ASM Aerospace Specification Metals (asm.matweb.com)",
            "basis": list(materials.BASIS),
            "n_materials": len(records),
            "n_skipped": len(skipped),
            "meam_enabled": not args.no_meam,
            "meam_unphysical": meam_unphysical,
            "meam_knobs": None if args.no_meam else knobs,
            "notes": (
                "Literature E/ν are polycrystalline datasheet values; model "
                "C11/C12 are single-crystal cubic. Out-of-basis elements "
                "(C, V, …) are dropped before modelling — see basis_coverage."
            ),
        },
        "summary": summary,
        "materials": records,
        "skipped": skipped,
    }
    with open(_OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote {_OUT_PATH}  ({len(records)} materials, "
          f"{len(skipped)} skipped)")
    _print_summary(summary, args.no_meam)
    return 0


def _fmt_row(label: str, key: str, s: dict) -> str:
    mae = f"{s['MAE']:.3f}" if s.get("MAE") is not None else "  --"
    mape = f"{s['MAPE']:.1f}" if s.get("MAPE") is not None else "  --"
    bias = f"{s['bias_pct']:+.1f}" if s.get("bias_pct") is not None else "  --"
    return f"{label:<14} {key:<6} {s.get('n', 0):>3} {mae:>8} {mape:>8} {bias:>8}"


def _print_summary(summary: dict, no_meam: bool) -> None:
    models = ["ML"] if no_meam else ["ML", "MEAM"]
    header = f"{'model':<14} {'target':<6} {'n':>3} {'MAE':>8} {'MAPE %':>8} {'bias %':>8}"

    print("\n=== Overall error vs literature ===")
    print(header); print("-" * len(header))
    for model in models:
        st = summary[model]["overall"]
        for key in ("E_GPa", "nu"):
            print(_fmt_row(model, key, st.get(key, {})))

    print("\n=== Young's modulus (E) error by material family ===")
    print(header); print("-" * len(header))
    for model in models:
        for fam, st in summary[model]["by_family"].items():
            print(_fmt_row(f"{model}/{fam}", "E_GPa", st.get("E_GPa", {})))


if __name__ == "__main__":
    sys.exit(main())
