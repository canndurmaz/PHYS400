#!/usr/bin/env python3
"""DFT-driven N-element MEAM potential pipeline.

Orchestrates the full workflow:
  1. GUI element selection (or --elements CLI override)
  2. DFT reference calculations
  3. MEAM initialization from DFT
  4. NN surrogate optimization against results.json targets
  5. Verification of optimized potential

Usage:
    python pipeline.py                          # Full pipeline with GUI
    python pipeline.py --elements Al Cu Zn Mg   # Skip GUI
    python pipeline.py --skip-dft --elements Al Cu Zn Mg  # Skip DFT stage
    python pipeline.py --samples 50             # More NN samples
"""

import argparse
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress tkinter dialogs before any config imports
os.environ.setdefault("TK_SILENT", "1")


def run_pipeline(elements=None, skip_dft=False, skip_gui=False, n_samples=30,
                 skip_optimize=False, skip_verify=False):
    """Run the full MEAM potential pipeline.

    Args:
        elements: list of element symbols (bypasses GUI if provided)
        skip_dft: skip DFT reference calculations (use existing dft_results.json)
        skip_gui: skip GUI even if elements not provided
        n_samples: number of NN parameter samples
        skip_optimize: skip NN optimization stage
        skip_verify: skip verification stage
    """
    eam_dir = os.path.join(project_root, "EAM")
    meam_opt_dir = os.path.dirname(__file__)

    print("=" * 60)
    print("DFT-Driven N-Element MEAM Potential Pipeline")
    print("=" * 60)

    # ── Stage 0: Element Selection ──────────────────────────────────────────
    if elements is None and not skip_gui:
        print("\n[Stage 0] Element Selection via GUI", flush=True)
        print("  Importing element_selector...", flush=True)
        from src.NNIP.element_selector import select_elements
        print("  Launching GUI...", flush=True)
        elements = select_elements()
        print(f"  GUI returned: {elements}", flush=True)
        if not elements:
            print("No elements selected. Exiting.")
            return
    elif elements is None:
        print("ERROR: No elements specified. Use --elements or remove --skip-gui")
        return

    print(f"\nSelected elements: {elements}", flush=True)

    # ── Stage 1: DFT Reference ─────────────────────────────────────────────
    dft_results_path = os.path.join(meam_opt_dir, "dft_results.json")
    print(f"\n[Stage 1] DFT Reference Calculations", flush=True)
    print(f"  dft_results.json path: {dft_results_path}", flush=True)

    if not skip_dft:
        print(f"  Mode: RUNNING DFT (skip_dft=False)", flush=True)
        print("-" * 40, flush=True)
        from src.NNIP.dft_reference import generate_dft_reference

        dft_work_dir = os.path.join(meam_opt_dir, "dft_scratch")
        print(f"  Scratch directory: {dft_work_dir}", flush=True)
        print(f"  Calling generate_dft_reference({elements})...", flush=True)
        dft_results = generate_dft_reference(
            elements,
            output_path=dft_results_path,
            work_dir=dft_work_dir,
        )
        print(f"  DFT complete. Results saved to {dft_results_path}", flush=True)
        print(f"  Elements computed: {list(dft_results.get('elements', {}).keys())}", flush=True)
        print(f"  Binary pairs computed: {list(dft_results.get('binary_pairs', {}).keys())}", flush=True)
    else:
        print(f"  Mode: SKIPPING DFT (using existing file)", flush=True)
        if os.path.exists(dft_results_path):
            with open(dft_results_path) as f:
                dft_results = json.load(f)
            print(f"  Loaded existing dft_results.json", flush=True)
            print(f"  Elements in file: {list(dft_results.get('elements', {}).keys())}", flush=True)
            print(f"  Pairs in file: {list(dft_results.get('binary_pairs', {}).keys())}", flush=True)
        else:
            print(f"  WARNING: {dft_results_path} not found. Stage 2 will use base values only.", flush=True)
            dft_results = {"elements": {}, "binary_pairs": {}}

    # ── Stage 2: DFT-to-MEAM Initialization ────────────────────────────────
    print(f"\n[Stage 2] MEAM Initialization from DFT", flush=True)
    print("-" * 40, flush=True)
    print(f"  DFT results to apply:", flush=True)
    for sym, data in dft_results.get("elements", {}).items():
        print(f"    {sym}: a_lat={data.get('a_lat','?')}, E_coh={data.get('E_coh','?')}, B={data.get('B_GPa','?')} GPa", flush=True)
    for pair, data in dft_results.get("binary_pairs", {}).items():
        print(f"    {pair}: E_form={data.get('E_form','?')} eV/atom", flush=True)

    from src.NNIP.dft_to_meam import initialize_meam_from_dft

    init_dir = os.path.join(eam_dir, "dft_initialized")
    print(f"  Output directory: {init_dir}", flush=True)
    print(f"  EAM base directory: {eam_dir}", flush=True)
    try:
        lib_init, par_init = initialize_meam_from_dft(
            dft_results, elements, eam_dir, output_dir=init_dir,
        )
        print(f"  DFT-initialized MEAM files created successfully.", flush=True)
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}", flush=True)
        print(f"  Falling back to existing merged files...", flush=True)
        prefix = "".join(elements)
        lib_init = os.path.join(eam_dir, f"library_{prefix}.meam")
        par_init = os.path.join(eam_dir, f"{prefix}.meam")
        print(f"  Trying: {lib_init}", flush=True)
        if not os.path.exists(lib_init):
            lib_init = os.path.join(eam_dir, "library_AlZnMgCuCrFeMnSiTi.meam")
            par_init = os.path.join(eam_dir, "AlZnMgCuCrFeMnSiTi.meam")
            print(f"  Trying 9-element fallback: {lib_init}", flush=True)
        if not os.path.exists(lib_init):
            print("  FATAL: No suitable MEAM files found. Run merge_potentials.py first.", flush=True)
            return

    print(f"  Library: {lib_init}  (exists: {os.path.exists(lib_init)})", flush=True)
    print(f"  Params:  {par_init}  (exists: {os.path.exists(par_init)})", flush=True)

    # ── Stage 3: NN Optimization ───────────────────────────────────────────
    if not skip_optimize:
        print("\n[Stage 3] NN Surrogate Optimization")
        print("-" * 40)
        from src.NNIP.nn_optimizer import optimize_nn

        lib_opt, par_opt = optimize_nn(
            lib_init, par_init, n_samples=n_samples,
        )
    else:
        print("\n[Stage 3] Skipping NN optimization")
        lib_opt, par_opt = lib_init, par_init

    # ── Stage 4: Verification ──────────────────────────────────────────────
    if not skip_verify:
        print("\n[Stage 4] Verification")
        print("-" * 40)
        _verify_against_targets(lib_opt, par_opt)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Optimized library: {lib_opt}")
    print(f"  Optimized params:  {par_opt}")
    print("=" * 60)


def _verify_against_targets(lib_path, params_path):
    """Verify optimized potential against results.json targets."""
    from src.NNIP.nn_optimizer import (
        load_training_targets, _run_lammps_composition,
    )

    entries = load_training_targets()
    total_E_err = 0
    total_nu_err = 0
    n = 0

    for name, data in entries.items():
        E, nu = _run_lammps_composition(lib_path, params_path, data["composition"])
        E_target = data["E_GPa"]
        nu_target = data["nu"]

        E_err = abs(E - E_target) / max(abs(E_target), 1e-6) * 100
        nu_err = abs(nu - nu_target) / max(abs(nu_target), 1e-6) * 100

        print(f"  {name}:")
        print(f"    E:  {E:.2f} vs {E_target:.2f} GPa ({E_err:.1f}% error)")
        print(f"    nu: {nu:.3f} vs {nu_target:.3f} ({nu_err:.1f}% error)")

        if E != 0:
            total_E_err += E_err
            total_nu_err += nu_err
            n += 1

    if n > 0:
        print(f"\n  Average errors: E={total_E_err/n:.1f}%, nu={total_nu_err/n:.1f}%")
        if total_E_err / n < 10 and total_nu_err / n < 10:
            print("  PASS: Within 10% target")
        else:
            print("  WARNING: Some targets exceed 10% error")


def main():
    parser = argparse.ArgumentParser(
        description="DFT-Driven N-Element MEAM Potential Pipeline"
    )
    parser.add_argument(
        "--elements", nargs="+", default=None,
        help="Element symbols (bypasses GUI). e.g. --elements Al Cu Fe"
    )
    parser.add_argument(
        "--skip-dft", action="store_true",
        help="Skip DFT stage, use existing dft_results.json"
    )
    parser.add_argument(
        "--skip-gui", action="store_true",
        help="Skip GUI (requires --elements)"
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
        "--samples", type=int, default=30,
        help="Number of NN parameter samples (default: 30)"
    )
    args = parser.parse_args()

    os.environ["TK_SILENT"] = "1"

    run_pipeline(
        elements=args.elements,
        skip_dft=args.skip_dft,
        skip_gui=args.skip_gui or (args.elements is not None),
        n_samples=args.samples,
        skip_optimize=args.skip_optimize,
        skip_verify=args.skip_verify,
    )


if __name__ == "__main__":
    main()
