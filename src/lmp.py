from lammps import lammps

# 1. Initialization
L = lammps()
L.command("units metal")
L.command("atom_style atomic")
L.command("boundary p p p")

# 2. Create Geometry (Copper FCC)
L.command("lattice fcc 3.61")
L.command("region box block 0 5 0 5 0 5")
L.command("create_box 1 box")
L.command("create_atoms 1 box")

# 3. Potential and Mass
L.command("mass 1 63.546")
L.command("pair_style eam")
L.command("pair_coeff * * /usr/share/lammps/potentials/Cu_u3.eam")

# 4. Initial Relaxation (Minimize energy)
L.command("minimize 1.0e-4 1.0e-6 100 1000")

# 5. Production Setup
L.command("velocity all create 300.0 4928459")
L.command("fix 1 all nvt temp 300.0 300.0 0.1") # Using NVT to keep temp stable
L.command("thermo 10")
L.command("dump 1 all custom 50 traj.lammpstrj id type x y z")

# 6. Data Collection Loop
temp_data = []

print("Starting loop...")
for i in range(100): # Run 100 blocks
    L.command("run 10") # Each block is 10 steps
    current_temp = L.get_thermo("temp")
    temp_data.append(current_temp)
    print(f"Step {i*10}: Temperature = {current_temp:.2f} K")

print("\nFinal list of temperatures caught by Python:")
print(temp_data)