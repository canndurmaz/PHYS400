#!/usr/bin/env python3
"""DFT-driven N-element MEAM potential pipeline.

Orchestrates the full workflow:
  1. Element discovery (auto from EAM library files, or --elements CLI override)
  2. DFT reference calculations
  3. MEAM initialization from DFT
  4. NN surrogate optimization against results.json targets
  5. Verification of optimized potential

Usage:
    python pipeline.py                          # Auto-discover elements from EAM/
    python pipeline.py --elements Al Cu Zn Mg   # Specify elements explicitly
    python pipeline.py --skip-dft --elements Al Cu Zn Mg  # Skip DFT stage
    python pipeline.py --samples 50             # More NN samples
"""

import argparse
import json
import os
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress tkinter dialogs before any config imports
os.environ.setdefault("TK_SILENT", "1")

from src.NNIP.logging_config import setup_logger

logger = setup_logger("pipeline")


def run_pipeline(elements=None, skip_dft=False, n_samples=50,
                 skip_optimize=False, skip_verify=False, n_parallel=4,
                 no_plots=False):
    """Run the full MEAM potential pipeline.

    Args:
        elements: list of element symbols (auto-discovered from EAM if None)
        skip_dft: skip DFT reference calculations (use existing dft_results.json)
        n_samples: number of NN parameter samples
        skip_optimize: skip NN optimization stage
        skip_verify: skip verification stage
        n_parallel: max parallel DFT workers (default: 4)
        no_plots: skip visualization generation
    """
    eam_dir = os.path.join(project_root, "EAM")
    meam_opt_dir = os.path.dirname(__file__)
    stage_timings = {}
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("DFT-Driven N-Element MEAM Potential Pipeline")
    logger.info("=" * 60)

    # ── Stage 0: Element Discovery ──────────────────────────────────────────
    t0 = time.time()
    if elements is None:
        logger.info("\n[Stage 0] Dynamic Element Discovery from EAM library files")
        sys.path.insert(0, os.path.join(project_root, "src", "MD"))
        from element import scan_eam_dir
        potentials = scan_eam_dir(eam_dir)
        all_symbols = set()
        for pot in potentials.values():
            for elem in pot["elements"]:
                all_symbols.add(elem.symbol)
        elements = sorted(all_symbols)
        logger.info(f"  Discovered {len(elements)} elements from {len(potentials)} potentials: {elements}")
        if not elements:
            logger.info("No elements found in EAM library files. Exiting.")
            return
    stage_timings["element_discovery"] = round(time.time() - t0, 2)

    logger.info(f"\nSelected elements: {elements}")

    # ── Stage 1: DFT Reference ─────────────────────────────────────────────
    t0 = time.time()
    dft_results_path = os.path.join(meam_opt_dir, "dft_results.json")
    logger.info(f"\n[Stage 1] DFT Reference Calculations")
    logger.info(f"  dft_results.json path: {dft_results_path}")

    if not skip_dft:
        logger.info(f"  Mode: RUNNING DFT (skip_dft=False)")
        logger.info("-" * 40)

        logger.info("  Ensuring pseudopotentials are available...")
        from src.NNIP.download_pseudopotentials import ensure_pseudopotentials
        ensure_pseudopotentials(elements)

        from src.NNIP.dft_reference import generate_dft_reference

        dft_work_dir = os.path.join(meam_opt_dir, "dft_scratch")
        logger.info(f"  Scratch directory: {dft_work_dir}")
        logger.info(f"  Calling generate_dft_reference({elements}, workers={n_parallel})...")
        dft_results = generate_dft_reference(
            elements,
            output_path=dft_results_path,
            work_dir=dft_work_dir,
            n_workers=n_parallel,
        )
        logger.info(f"  DFT complete. Results saved to {dft_results_path}")
        logger.info(f"  Elements computed: {list(dft_results.get('elements', {}).keys())}")
        logger.info(f"  Binary pairs computed: {list(dft_results.get('binary_pairs', {}).keys())}")
    else:
        logger.info(f"  Mode: SKIPPING DFT (using existing file)")
        if os.path.exists(dft_results_path):
            with open(dft_results_path) as f:
                dft_results = json.load(f)
            logger.info(f"  Loaded existing dft_results.json")
            logger.info(f"  Elements in file: {list(dft_results.get('elements', {}).keys())}")
            logger.info(f"  Pairs in file: {list(dft_results.get('binary_pairs', {}).keys())}")
        else:
            logger.info(f"  WARNING: {dft_results_path} not found. Stage 2 will use base values only.")
            dft_results = {"elements": {}, "binary_pairs": {}}
    stage_timings["dft_reference"] = round(time.time() - t0, 2)

    # ── Stage 2: DFT-to-MEAM Initialization ────────────────────────────────
    t0 = time.time()
    logger.info(f"\n[Stage 2] MEAM Initialization from DFT")
    logger.info("-" * 40)
    logger.info(f"  DFT results to apply:")
    for sym, data in dft_results.get("elements", {}).items():
        logger.info(f"    {sym}: a_lat={data.get('a_lat','?')}, E_coh={data.get('E_coh','?')}, B={data.get('B_GPa','?')} GPa")
    for pair, data in dft_results.get("binary_pairs", {}).items():
        logger.info(f"    {pair}: E_form={data.get('E_form','?')} eV/atom")

    from src.NNIP.dft_to_meam import initialize_meam_from_dft

    init_dir = os.path.join(eam_dir, "dft_initialized")

    # Auto-discover merge config if available, otherwise proceed without
    merge_config = None
    configs_dir = os.path.join(project_root, "src", "configs")
    if os.path.isdir(configs_dir):
        for f in os.listdir(configs_dir):
            if f.startswith("meam_merge") and f.endswith(".json"):
                merge_config = os.path.join(configs_dir, f)
                break

    logger.info(f"  Output directory: {init_dir}")
    logger.info(f"  EAM base directory: {eam_dir}")
    logger.info(f"  Merge config: {merge_config or '(auto-discover base files)'}")
    try:
        lib_init, par_init = initialize_meam_from_dft(
            dft_results, elements, eam_dir,
            merge_config_path=merge_config, output_dir=init_dir,
        )
        logger.info(f"  DFT-initialized MEAM files created successfully.")
    except FileNotFoundError as exc:
        logger.error(f"  FATAL: {exc}")
        logger.info(f"  Ensure library_*.meam files in {eam_dir} cover all selected elements.")
        return

    logger.info(f"  Library: {lib_init}  (exists: {os.path.exists(lib_init)})")
    logger.info(f"  Params:  {par_init}  (exists: {os.path.exists(par_init)})")
    stage_timings["meam_init"] = round(time.time() - t0, 2)

    # ── Stage 3: NN Optimization ───────────────────────────────────────────
    t0 = time.time()
    if not skip_optimize:
        logger.info("\n[Stage 3] NN Surrogate Optimization")
        logger.info("-" * 40)
        from src.NNIP.nn_optimizer import optimize_nn

        lib_opt, par_opt = optimize_nn(
            lib_init, par_init, n_samples=n_samples,
            n_parallel=n_parallel,
        )
    else:
        logger.info("\n[Stage 3] Skipping NN optimization")
        lib_opt, par_opt = lib_init, par_init
    stage_timings["nn_optimization"] = round(time.time() - t0, 2)

    # ── Stage 4: Verification ──────────────────────────────────────────────
    t0 = time.time()
    verification_results = None
    if not skip_verify:
        logger.info("\n[Stage 4] Verification")
        logger.info("-" * 40)
        verification_results = _verify_against_targets(lib_opt, par_opt)
    stage_timings["verification"] = round(time.time() - t0, 2)

    # ── Stage 5: Visualization ─────────────────────────────────────────────
    if not no_plots:
        t0 = time.time()
        from src.NNIP.visualize import plot_all
        plot_all(verification_results=verification_results)
        stage_timings["visualization"] = round(time.time() - t0, 2)

    stage_timings["total"] = round(time.time() - pipeline_start, 2)

    # ── Write pipeline summary ─────────────────────────────────────────────
    summary = {
        "elements": elements,
        "n_samples": n_samples,
        "skip_dft": skip_dft,
        "skip_optimize": skip_optimize,
        "skip_verify": skip_verify,
        "stage_timings_sec": stage_timings,
    }
    if verification_results:
        summary["verification"] = verification_results
    summary_path = os.path.join(meam_opt_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nPipeline summary saved to {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Optimized library: {lib_opt}")
    logger.info(f"  Optimized params:  {par_opt}")
    for stage, dt in stage_timings.items():
        logger.info(f"  {stage}: {dt}s")
    logger.info("=" * 60)


def _verify_against_targets(lib_path, params_path):
    """Verify optimized potential against results.json targets.

    Returns:
        dict: {name: {E_opt, E_target, E_err_pct, nu_opt, nu_target, nu_err_pct}}
    """
    from src.NNIP.nn_optimizer import (
        load_training_targets, _run_lammps_composition,
    )

    entries = load_training_targets()
    total_E_err = 0
    total_nu_err = 0
    n = 0
    results = {}

    for name, data in entries.items():
        E, nu = _run_lammps_composition(lib_path, params_path, data["composition"])
        E_target = data["E_GPa"]
        nu_target = data["nu"]

        E_err = abs(E - E_target) / max(abs(E_target), 1e-6) * 100
        nu_err = abs(nu - nu_target) / max(abs(nu_target), 1e-6) * 100

        logger.info(f"  {name}:")
        logger.info(f"    E:  {E:.2f} vs {E_target:.2f} GPa ({E_err:.1f}% error)")
        logger.info(f"    nu: {nu:.3f} vs {nu_target:.3f} ({nu_err:.1f}% error)")

        results[name] = {
            "E_opt": E, "E_target": E_target, "E_err_pct": round(E_err, 2),
            "nu_opt": nu, "nu_target": nu_target, "nu_err_pct": round(nu_err, 2),
        }

        if E != 0:
            total_E_err += E_err
            total_nu_err += nu_err
            n += 1

    if n > 0:
        logger.info(f"\n  Average errors: E={total_E_err/n:.1f}%, nu={total_nu_err/n:.1f}%")
        if total_E_err / n < 10 and total_nu_err / n < 10:
            logger.info("  PASS: Within 10% target")
        else:
            logger.info("  WARNING: Some targets exceed 10% error")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="DFT-Driven N-Element MEAM Potential Pipeline"
    )
    parser.add_argument(
        "--elements", nargs="+", default=None,
        help="Element symbols (auto-discovered from EAM/ if omitted). e.g. --elements Al Cu Fe"
    )
    parser.add_argument(
        "--skip-dft", action="store_true",
        help="Skip DFT stage, use existing dft_results.json"
    )
    parser.add_argument(
        "--skip-optimize", action="store_true",
        help="Skip NN optimization stage"
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip verification stage"
    )
    parser.add_argument(
        "--samples", type=int, default=50,
        help="Number of NN parameter samples (default: 50)"
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="Max parallel DFT workers (default: 4)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip visualization/plot generation"
    )
    args = parser.parse_args()

    os.environ["TK_SILENT"] = "1"

    run_pipeline(
        elements=args.elements,
        skip_dft=args.skip_dft,
        n_samples=args.samples,
        skip_optimize=args.skip_optimize,
        skip_verify=args.skip_verify,
        n_parallel=args.parallel,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
