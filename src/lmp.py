from lammps import lammps
from element import MEAM_ELEMENT_ORDER
from config import CONFIG, composition, selected, a_mean, n_repeats

EAM_DIR = "/home/kenobi/Workspaces/PHYS400/EAM"

# pair_coeff strings
library_elements = " ".join(MEAM_ELEMENT_ORDER)
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
cumulative = composition[selected[0].symbol]
for i, elem in enumerate(selected[1:], start=2):
    frac = composition[elem.symbol] / cumulative
    L.command(f"set group all type/fraction {i} {frac:.6f} 12345")
    cumulative -= composition[elem.symbol]

# 3. MEAM Potential
L.command("pair_style meam")
L.command(
    f"pair_coeff * * {EAM_DIR}/library.meam {library_elements} "
    f"{EAM_DIR}/FeMnNiTiCuCrCoAl.meam {active_elements}"
)

# Masses
for i, elem in enumerate(selected, start=1):
    L.command(f"mass {i} {elem.mass}")

# 4. Initial Relaxation
L.command("minimize 1.0e-4 1.0e-6 100 1000")

# 5. Production Setup
temp = CONFIG["temperature"]
L.command(f"velocity all create {temp} 4928459")
L.command(f"fix 1 all nvt temp {temp} {temp} 0.1")
L.command(f"thermo {CONFIG['thermo_interval']}")
L.command(f"dump 1 all custom {CONFIG['dump_interval']} traj.lammpstrj id type x y z")

# 6. Data Collection Loop
temp_data = []
steps_per_iter = CONFIG["thermo_interval"]
n_iters = CONFIG["total_steps"] // steps_per_iter

print("Starting loop...")
for i in range(n_iters):
    L.command(f"run {steps_per_iter}")
    current_temp = L.get_thermo("temp")
    temp_data.append(current_temp)
    print(f"Step {(i+1)*steps_per_iter}: Temperature = {current_temp:.2f} K")

print("\nFinal temperatures:")
print(temp_data)
