import os
import json
import sys
from lammps import lammps
from config import load_configuration, get_derived_quantities, select_multiple_configs

# ── Elastic Properties ────────────────────────────────────────
def get_elastic_moduli(L, delta=1e-3):
    """Estimate Young's Modulus and Poisson's Ratio via axial strain."""
    print(f"Estimating elastic properties (delta={delta})...")
    
    # Baseline stress
    L.command("run 0")
    s0_xx = -L.get_thermo("pxx") * 1e-4
    s0_yy = -L.get_thermo("pyy") * 1e-4
    
    # Apply +delta strain in x
    L.command(f"change_box all x scale {1.0 + delta} remap units box")
    L.command("minimize 1e-10 1e-10 1000 10000")
    s_plus_xx = -L.get_thermo("pxx") * 1e-4
    s_plus_yy = -L.get_thermo("pyy") * 1e-4
    
    # C11, C12 estimate from axial perturbation
    c11 = (s_plus_xx - s0_xx) / delta
    c12 = (s_plus_yy - s0_yy) / delta
    
    # Revert strain for production run
    L.command(f"change_box all x scale {1.0 / (1.0 + delta)} remap units box")
    L.command("minimize 1e-10 1e-10 1000 10000")
    
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
def run_simulation(config_path):
    print(f"\n" + "="*60)
    print(f"Processing: {config_path if config_path else 'Default'}")
    print("="*60)
    
    config, config_name = load_configuration(config_path)
    comp, sel, a_m, n_rep, pot = get_derived_quantities(config)

    # pair_coeff strings
    library_elements = " ".join(e.symbol for e in pot["elements"])
    active_elements = " ".join(e.symbol for e in sel)

    L = lammps(cmdargs=["-log", "none", "-screen", "none"])
    L.command("units metal")
    L.command("atom_style atomic")
    L.command("boundary p p p")

    L.command(f"lattice {sel[0].lattice_type} {a_m:.4f}")
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

    print("Relaxing ground state...")
    L.command("minimize 1.0e-4 1.0e-6 100 1000")

    E, nu = get_elastic_moduli(L)
    print(f"\nProperties: Young's Modulus (E) = {E:.2f} GPa, Poisson's Ratio (nu) = {nu:.3f}\n")

    save_to_ml_results(E, nu, config_name, comp)

    # ── Visualization ─────────────────────────────────────────────
    from viz import render
    render(comp, sel)
    
    L.close()

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
    clear_old_videos()
    # Select multiple files via GUI
    paths = select_multiple_configs()
    
    if not paths:
        print("No files selected or operation cancelled.")
        # Optional: run_simulation(None) to run default if nothing selected?
        # Let's just exit to respect user cancellation
    else:
        for p in paths:
            try:
                run_simulation(p)
            except Exception as ex:
                print(f"Error processing {p}: {ex}")
