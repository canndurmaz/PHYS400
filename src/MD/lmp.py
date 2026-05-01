import os
import json
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from lammps import lammps
from config import load_configuration, get_derived_quantities, select_multiple_configs

# Downstream surrogates (src/ML/nn_alloy.py and src/NNIP/nn_optimizer.py)
# drop any sample with ν close to the (1-2ν)=0 singularity, since the
# (E,ν) -> (C11,C12) algebra is ill-conditioned there. We apply the same
# filter at write time so results.json only ever contains entries the
# surrogates can actually use.
NU_FILTER_MAX = 0.48

# ── Elastic Properties ────────────────────────────────────────
def get_elastic_moduli(L, delta=1e-3):
    """Estimate Young's Modulus and Poisson's Ratio via axial strain.

    Strains in x, y, and z independently with central differences,
    re-minimizing atoms after each deformation for accurate stress.
    Averaging over 3 directions reduces noise from local disorder
    in random alloy supercells.
    """
    print(f"Estimating elastic properties (delta={delta})...")

    # Baseline stress after atom-only re-minimize
    L.command("minimize 1.0e-6 1.0e-8 100 1000")
    L.command("run 0")
    p0 = {
        "pxx": L.get_thermo("pxx"),
        "pyy": L.get_thermo("pyy"),
        "pzz": L.get_thermo("pzz"),
    }
    print(f"  Baseline (bar): pxx={p0['pxx']:.2f}, pyy={p0['pyy']:.2f}, pzz={p0['pzz']:.2f}")

    directions = ["x", "y", "z"]
    axial_map = {"x": "pxx", "y": "pyy", "z": "pzz"}
    transverse_map = {
        "x": ["pyy", "pzz"],
        "y": ["pxx", "pzz"],
        "z": ["pxx", "pyy"],
    }

    c11_samples = []
    c12_samples = []

    for d in directions:
        ax = axial_map[d]
        tr = transverse_map[d]

        # +delta strain
        L.command(f"change_box all {d} scale {1.0 + delta} remap units box")
        L.command("minimize 1.0e-6 1.0e-8 100 1000")
        L.command("run 0")
        p_plus = {k: L.get_thermo(k) for k in ["pxx", "pyy", "pzz"]}

        # Revert to original
        L.command(f"change_box all {d} scale {1.0 / (1.0 + delta)} remap units box")

        # -delta strain
        L.command(f"change_box all {d} scale {1.0 - delta} remap units box")
        L.command("minimize 1.0e-6 1.0e-8 100 1000")
        L.command("run 0")
        p_minus = {k: L.get_thermo(k) for k in ["pxx", "pyy", "pzz"]}

        # Revert to original
        L.command(f"change_box all {d} scale {1.0 / (1.0 - delta)} remap units box")

        # Central difference: C = -(dp/de), bar -> GPa
        c11_d = -(p_plus[ax] - p_minus[ax]) / (2 * delta) * 1e-4
        c11_samples.append(c11_d)

        for t in tr:
            c12_d = -(p_plus[t] - p_minus[t]) / (2 * delta) * 1e-4
            c12_samples.append(c12_d)

        print(f"  strain {d}: C11={c11_d:.2f} GPa, "
              f"C12={sum(-(p_plus[t] - p_minus[t]) / (2*delta) * 1e-4 for t in tr) / len(tr):.2f} GPa")

    c11 = sum(c11_samples) / len(c11_samples)
    c12 = sum(c12_samples) / len(c12_samples)
    print(f"  Averaged: C11={c11:.2f} GPa, C12={c12:.2f} GPa")

    # Formula for cubic/isotropic materials
    E = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)
    nu = c12 / (c11 + c12)
    return E, nu, c11, c12

# ── Save Results to ML ────────────────────────────────────────
def save_to_ml_results(E_val, nu_val, C11_val, C12_val, config_name, composition):
    # Path to ML results relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.abspath(os.path.join(script_dir, "..", "ML", "results.json"))

    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            pass

    # Update entry. C11_GPa/C12_GPa are persisted alongside E_GPa/nu so
    # that downstream consumers (src/ML/nn_alloy.py and
    # src/NNIP/nn_optimizer.py) can read the elastic constants directly
    # instead of reconstructing them algebraically.
    results[config_name] = {
        "composition": composition,
        "E_GPa": round(E_val, 2),
        "nu": round(nu_val, 3),
        "C11_GPa": round(C11_val, 2),
        "C12_GPa": round(C12_val, 2),
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results updated in {results_path} for: {config_name}")

# ── Simulation Function ───────────────────────────────────────
def is_already_done(config_path):
    """Check if this config's results already exist in results.json."""
    if not config_path:
        return False
    config_name = os.path.basename(config_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.abspath(os.path.join(script_dir, "..", "ML", "results.json"))
    if not os.path.exists(results_path):
        return False
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
        return config_name in results
    except (json.JSONDecodeError, OSError):
        return False

def run_simulation(config_path, viz=False, sim_md=False, force=False):
    if not force and is_already_done(config_path):
        print(f"  SKIP (already in results.json): {os.path.basename(config_path)}")
        return

    print(f"\n" + "="*60)
    print(f"Processing: {config_path if config_path else 'Default'}")
    print("="*60)

    config, config_name = load_configuration(config_path)
    comp, sel, a_m, n_rep, pot, dominant = get_derived_quantities(config)

    # pair_coeff strings — use meam_label (the label in the library file)
    library_elements = " ".join(e.meam_label for e in pot["elements"])
    active_elements = " ".join(e.meam_label for e in sel)

    L = lammps(cmdargs=["-log", "none", "-screen", "none"])
    L.command("units metal")
    L.command("atom_style atomic")
    L.command("boundary p p p")

    L.command(f"lattice {dominant.lattice_type} {a_m:.4f}")
    L.command(f"region box block 0 {n_rep} 0 {n_rep} 0 {n_rep}")
    L.command(f"create_box {len(sel)} box")
    L.command("create_atoms 1 box")

    remaining = 1.0
    for i, elem in enumerate(sel[1:], start=2):
        frac = comp[elem.symbol] / remaining
        L.command(f"set type 1 type/fraction {i} {frac:.6f} 12345")
        remaining -= comp[elem.symbol]

    L.command("pair_style meam")
    L.command(
        f"pair_coeff * * {pot['library']} {library_elements} "
        f"{pot['params']} {active_elements}"
    )

    for i, elem in enumerate(sel, start=1):
        L.command(f"mass {i} {elem.mass}")

    print("Relaxing ground state (box + atoms)...")
    L.command("fix boxrelax all box/relax aniso 0.0 vmax 0.001")
    L.command("minimize 1.0e-6 1.0e-8 1000 10000")
    L.command("unfix boxrelax")
    # Atom-only re-minimize at fixed box to clean up residual forces
    L.command("minimize 1.0e-6 1.0e-8 200 2000")

    E, nu, C11, C12 = get_elastic_moduli(L)
    print(f"\nProperties: E = {E:.2f} GPa, nu = {nu:.3f}, "
          f"C11 = {C11:.2f} GPa, C12 = {C12:.2f} GPa\n")

    reason = None
    if E < 0:
        reason = f"negative Young's modulus ({E:.2f} GPa)"
    elif nu < 0:
        reason = f"negative Poisson's ratio ({nu:.3f})"
    elif nu >= NU_FILTER_MAX:
        reason = f"Poisson's ratio nu={nu:.3f} >= {NU_FILTER_MAX:.2f} (singularity)"
    elif C11 < C12:
        reason = f"mechanical instability (C11={C11:.2f} < C12={C12:.2f})"
    if reason is not None:
        print(f"  DISCARDED: {reason}")
        L.close()
        if config_path and os.path.isfile(config_path):
            os.remove(config_path)
            print(f"  Deleted config: {config_path}")
        return

    save_to_ml_results(E, nu, C11, C12, config_name, comp)

    # ── MD Run ────────────────────────────────────────────────────
    traj_file = f"traj_{config_name}.lammpstrj"

    if sim_md:
        temp = config.get("temperature", 300.0)
        total_steps = config.get("total_steps", 1000)
        thermo_int = config.get("thermo_interval", 10)
        dump_int = config.get("dump_interval", 50)

        L.command(f"velocity all create {temp} 54321 dist gaussian")
        L.command(f"fix nvt all nvt temp {temp} {temp} 0.1")
        L.command(f"thermo {thermo_int}")
        L.command(f"dump traj all atom {dump_int} {traj_file}")
        print(f"Running MD: {total_steps} steps at {temp} K...")
        L.command(f"run {total_steps}")
        L.command("undump traj")
        L.command("unfix nvt")

    L.close()

    # ── Visualization ─────────────────────────────────────────────
    if viz:
        if not os.path.exists(traj_file):
            print(f"Warning: {traj_file} not found. Run with --simMD to generate trajectory.")
        else:
            from viz import render
            render(comp, sel, traj_file=traj_file)

def clear_old_videos():
    """Remove all existing mp4 files from the visualization directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vis_dir = os.path.join(script_dir, "visualization")
    if os.path.exists(vis_dir):
        import glob
        videos = glob.glob(os.path.join(vis_dir, "*.mp4"))
        for v in videos:
            try:
                os.remove(v)
            except:
                pass
        if videos:
            print(f"Cleared {len(videos)} existing video(s) from {vis_dir}")


# ── Parallel elastic-only worker ──────────────────────────────────────
def _eval_elastic_only(args):
    """Process-safe worker: build a fresh LAMMPS, return computed (E,nu).

    Returns a result dict with one of:
      {"status":"skip",    "path":...}
      {"status":"discard", "path":..., "name":..., "nu":...}
      {"status":"ok",      "path":..., "name":..., "composition":..., "E":..., "nu":...}
      {"status":"error",   "path":..., "error":"..."}

    All ``results.json`` I/O and config-file deletions are performed by the
    main process to keep the worker free of shared-state side effects.
    """
    config_path, force = args
    if not force and is_already_done(config_path):
        return {"status": "skip", "path": config_path}
    try:
        config, config_name = load_configuration(config_path)
        comp, sel, a_m, n_rep, pot, dominant = get_derived_quantities(config)
        library_elements = " ".join(e.meam_label for e in pot["elements"])
        active_elements = " ".join(e.meam_label for e in sel)

        L = lammps(cmdargs=["-log", "none", "-screen", "none"])
        try:
            L.command("units metal")
            L.command("atom_style atomic")
            L.command("boundary p p p")
            L.command(f"lattice {dominant.lattice_type} {a_m:.4f}")
            L.command(f"region box block 0 {n_rep} 0 {n_rep} 0 {n_rep}")
            L.command(f"create_box {len(sel)} box")
            L.command("create_atoms 1 box")

            remaining = 1.0
            for i, elem in enumerate(sel[1:], start=2):
                frac = comp[elem.symbol] / remaining
                L.command(f"set type 1 type/fraction {i} {frac:.6f} 12345")
                remaining -= comp[elem.symbol]

            L.command("pair_style meam")
            L.command(
                f"pair_coeff * * {pot['library']} {library_elements} "
                f"{pot['params']} {active_elements}"
            )
            for i, elem in enumerate(sel, start=1):
                L.command(f"mass {i} {elem.mass}")

            L.command("fix boxrelax all box/relax aniso 0.0 vmax 0.001")
            L.command("minimize 1.0e-6 1.0e-8 1000 10000")
            L.command("unfix boxrelax")
            L.command("minimize 1.0e-6 1.0e-8 200 2000")

            E, nu, C11, C12 = get_elastic_moduli(L)
        finally:
            L.close()

        reason = None
        if E < 0:
            reason = f"E={E:.2f} GPa < 0"
        elif nu < 0:
            reason = f"nu={nu:.3f} < 0"
        elif nu >= NU_FILTER_MAX:
            reason = f"nu={nu:.3f} >= {NU_FILTER_MAX:.2f} (singularity)"
        elif C11 < C12:
            reason = f"C11={C11:.2f} < C12={C12:.2f} (instability)"
        if reason is not None:
            return {"status": "discard", "path": config_path,
                    "name": config_name, "E": E, "nu": nu,
                    "C11": C11, "C12": C12, "reason": reason}
        return {"status": "ok", "path": config_path, "name": config_name,
                "composition": comp, "E": E, "nu": nu,
                "C11": C11, "C12": C12}
    except Exception as ex:
        return {"status": "error", "path": config_path, "error": str(ex)}


def _run_parallel(paths, force, n_jobs):
    """Dispatch elastic-only evaluation across ``n_jobs`` worker processes."""
    pending = [p for p in paths if force or not is_already_done(p)]
    n_skip = len(paths) - len(pending)
    if n_skip:
        print(f"Skipping {n_skip} configs already in results.json")
    if not pending:
        print("Nothing to do.")
        return

    # Pin each LAMMPS worker to a single OMP thread so n_jobs workers
    # saturate physical cores cleanly (no oversubscription).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print(f"Parallel mode: {n_jobs} worker(s), "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}, "
          f"{len(pending)} config(s) pending")

    t_start = time.time()
    completed = 0
    n_ok = n_disc = n_err = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(_eval_elastic_only, (p, force)): p for p in pending}
        for fut in as_completed(futures):
            completed += 1
            try:
                r = fut.result()
            except Exception as ex:
                p = futures[fut]
                r = {"status": "error", "path": p, "error": str(ex)}

            wall = time.time() - t_start
            avg = wall / completed
            eta = avg * (len(pending) - completed)
            tag = os.path.basename(r["path"])
            wm, ws = divmod(int(wall), 60)
            em, es = divmod(int(eta), 60)
            print(f"[{completed}/{len(pending)}] {tag}: {r['status']} | "
                  f"elapsed {wm}m {ws:02d}s | ETA {em}m {es:02d}s")

            if r["status"] == "ok":
                save_to_ml_results(r["E"], r["nu"], r["C11"], r["C12"],
                                   r["name"], r["composition"])
                n_ok += 1
            elif r["status"] == "discard":
                print(f"  DISCARDED ({r.get('reason','?')}): "
                      f"E={r['E']:.2f} GPa, nu={r['nu']:.3f}, "
                      f"C11={r.get('C11', float('nan')):.2f}, "
                      f"C12={r.get('C12', float('nan')):.2f}")
                if os.path.isfile(r["path"]):
                    try:
                        os.remove(r["path"])
                        print(f"  Deleted config: {r['path']}")
                    except OSError as ex:
                        print(f"  Could not delete {r['path']}: {ex}")
                n_disc += 1
            else:
                print(f"  ERROR: {r.get('error','?')}")
                n_err += 1

    total = time.time() - t_start
    tm, ts = divmod(int(total), 60)
    th, tm = divmod(tm, 60)
    print(f"\nParallel done: {n_ok} ok / {n_disc} discarded / {n_err} errored "
          f"in {th}h {tm}m {ts}s")

if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(description="MEAM MD Simulation")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--simMD", action="store_true", help="Run NVT molecular dynamics")
    parser.add_argument("--all", metavar="DIR", help="Run all .json configs in DIR")
    parser.add_argument("--force", action="store_true", help="Re-run even if already in results.json")
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help="Number of parallel workers for the elastic-only "
                             "path (default: cpu_count // 2). Set to 1 to "
                             "force the sequential code path.")
    args = parser.parse_args()

    if args.viz:
        clear_old_videos()

    if args.all:
        paths = sorted(glob.glob(os.path.join(args.all, "*.json")))
        if not paths:
            print(f"No .json files found in {args.all}")
            sys.exit(1)
        print(f"Running {len(paths)} configs from {args.all}")
    else:
        paths = select_multiple_configs()

    if not paths:
        print("No files selected or operation cancelled.")
        sys.exit(0)

    # Decide whether to run in parallel. The parallel path covers only the
    # elastic-only workflow (the dominant case for batch processing); --viz
    # and --simMD keep the original sequential behaviour because MD
    # trajectories and OVITO rendering are inherently per-process.
    n_jobs = args.jobs if args.jobs is not None else max(1, mp.cpu_count() // 2)
    use_parallel = (
        not args.viz and not args.simMD
        and n_jobs > 1 and len(paths) > 1
    )

    if use_parallel:
        _run_parallel(paths, force=args.force, n_jobs=n_jobs)
    else:
        total = len(paths)
        elapsed_times = []
        t_start = time.time()

        for i, p in enumerate(paths, 1):
            t_config = time.time()
            try:
                eta_str = ""
                if elapsed_times:
                    avg = sum(elapsed_times) / len(elapsed_times)
                    remaining = (total - i + 1) * avg
                    rm, rs = divmod(int(remaining), 60)
                    rh, rm = divmod(rm, 60)
                    if rh > 0:
                        eta_str = f" | ETA: {rh}h {rm}m {rs}s"
                    else:
                        eta_str = f" | ETA: {rm}m {rs}s"

                wall = time.time() - t_start
                wm, ws = divmod(int(wall), 60)
                wh, wm = divmod(wm, 60)
                wall_str = f"{wh}h {wm}m {ws}s" if wh > 0 else f"{wm}m {ws}s"

                print(f"\n[{i}/{total}] {os.path.basename(p)} | Elapsed: {wall_str}{eta_str}")
                run_simulation(p, viz=args.viz, sim_md=args.simMD, force=args.force)
            except Exception as ex:
                print(f"Error processing {p}: {ex}")

            dt = time.time() - t_config
            elapsed_times.append(dt)

        wall_total = time.time() - t_start
        tm, ts = divmod(int(wall_total), 60)
        th, tm = divmod(tm, 60)
        avg_str = f"{sum(elapsed_times)/len(elapsed_times):.1f}s" if elapsed_times else "N/A"
        print(f"\nDone: {total} configs in {th}h {tm}m {ts}s (avg {avg_str}/config)")
