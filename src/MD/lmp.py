import os
import json
import sys
from lammps import lammps
from config import CONFIG, composition, selected, a_mean, n_repeats, potential

# pair_coeff strings
library_elements = " ".join(e.symbol for e in potential["elements"])
active_elements = " ".join(e.symbol for e in selected)

# ── LAMMPS simulation ─────────────────────────────────────────
L = lammps()

# 1. Initialization
L.command("units metal")
L.command("atom_style atomic")
L.command("boundary p p p")

# 2. Geometry — use first element's lattice type, mean lattice constant
L.command(f"lattice {selected[0].lattice_type} {a_mean:.4f}")
L.command(f"region box block 0 {n_repeats} 0 {n_repeats} 0 {n_repeats}")
L.command(f"create_box {len(selected)} box")
L.command("create_atoms 1 box")

# Assign compositions via set type/fraction (skip first element — it's already type 1)
# Each iteration converts a fraction of remaining type-1 atoms to the next type.
remaining = 1.0
for i, elem in enumerate(selected[1:], start=2):
    frac = composition[elem.symbol] / remaining
    L.command(f"set type 1 type/fraction {i} {frac:.6f} 12345")
    remaining -= composition[elem.symbol]

# 3. MEAM Potential
L.command("pair_style meam")
L.command(
    f"pair_coeff * * {potential['library']} {library_elements} "
    f"{potential['params']} {active_elements}"
)

# Masses
for i, elem in enumerate(selected, start=1):
    L.command(f"mass {i} {elem.mass}")

# 4. Initial Relaxation
print("Relaxing ground state...")
L.command("minimize 1.0e-4 1.0e-6 100 1000")

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

E, nu = get_elastic_moduli(L)
print(f"\nProperties: Young's Modulus (E) = {E:.2f} GPa, Poisson's Ratio (nu) = {nu:.3f}\n")

# ── Save Results to ML ────────────────────────────────────────
def save_to_ml_results(E_val, nu_val):
    # Try to identify the config filename from CLI arguments
    config_name = "manual_config.json"
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        config_name = os.path.basename(sys.argv[1])
    
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

save_to_ml_results(E, nu)

# 5. Production Simulation (Optional)
def run_thermal_simulation(L, config):
    """Run an NVT production simulation and collect temperature data."""
    temp = config["temperature"]
    print(f"Starting thermal simulation at {temp} K...")
    L.command(f"velocity all create {temp} 4928459")
    L.command(f"fix 1 all nvt temp {temp} {temp} 0.1")
    L.command(f"thermo {config['thermo_interval']}")
    L.command(f"dump 1 all custom {config['dump_interval']} traj.lammpstrj id type x y z")

    temp_data = []
    steps_per_iter = config["thermo_interval"]
    n_iters = config["total_steps"] // steps_per_iter

    for i in range(n_iters):
        L.command(f"run {steps_per_iter}")
        current_temp = L.get_thermo("temp")
        temp_data.append(current_temp)
        print(f"Step {(i+1)*steps_per_iter}: Temperature = {current_temp:.2f} K")
    
    return temp_data

# Skip thermal test by default as requested
# temp_data = run_thermal_simulation(L, CONFIG)

# ── Visualization ─────────────────────────────────────────────
from viz import render
render(composition, selected)
