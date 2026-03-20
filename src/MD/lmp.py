import os
import json
import sys
from lammps import lammps
from config import load_configuration, get_derived_quantities, select_multiple_configs

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
    return E, nu

# ── Save Results to ML ────────────────────────────────────────
def save_to_ml_results(E_val, nu_val, config_name, composition):
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
            
    # Update entry
    results[config_name] = {
        "composition": composition,
        "E_GPa": round(E_val, 2),
        "nu": round(nu_val, 3)
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

    E, nu = get_elastic_moduli(L)
    print(f"\nProperties: Young's Modulus (E) = {E:.2f} GPa, Poisson's Ratio (nu) = {nu:.3f}\n")

    if nu < 0:
        print(f"  DISCARDED: negative Poisson's ratio ({nu:.3f})")
        L.close()
        if config_path and os.path.isfile(config_path):
            os.remove(config_path)
            print(f"  Deleted config: {config_path}")
        return

    save_to_ml_results(E, nu, config_name, comp)

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

if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(description="MEAM MD Simulation")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    parser.add_argument("--simMD", action="store_true", help="Run NVT molecular dynamics")
    parser.add_argument("--all", metavar="DIR", help="Run all .json configs in DIR")
    parser.add_argument("--force", action="store_true", help="Re-run even if already in results.json")
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
    else:
        for i, p in enumerate(paths, 1):
            try:
                print(f"\n[{i}/{len(paths)}] {os.path.basename(p)}")
                run_simulation(p, viz=args.viz, sim_md=args.simMD, force=args.force)
            except Exception as ex:
                print(f"Error processing {p}: {ex}")
