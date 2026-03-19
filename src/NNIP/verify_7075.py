import os
import sys
import json
import numpy as np
from lammps import lammps

# Add project root to sys.path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.MD.config import load_configuration, get_derived_quantities
from src.NNIP.logging_config import setup_logger

logger = setup_logger("verify_7075")

def get_elastic_moduli(L, delta=1e-3):
    """Estimate Young's Modulus and Poisson's Ratio via axial strain."""
    logger.info(f"Estimating elastic properties (delta={delta})...")
    
    # Baseline stress
    L.command("run 0")
    s0_xx = -L.get_thermo("pxx") * 1e-4 # bar to GPa
    s0_yy = -L.get_thermo("pyy") * 1e-4
    
    # Apply +delta strain in x
    L.command(f"change_box all x scale {1.0 + delta} remap units box")
    L.command("minimize 1e-10 1e-10 1000 10000")
    s_plus_xx = -L.get_thermo("pxx") * 1e-4
    s_plus_yy = -L.get_thermo("pyy") * 1e-4
    
    # C11, C12 estimate from axial perturbation
    c11 = (s_plus_xx - s0_xx) / delta
    c12 = (s_plus_yy - s0_yy) / delta
    
    # Revert strain
    L.command(f"change_box all x scale {1.0 / (1.0 + delta)} remap units box")
    L.command("minimize 1e-10 1e-10 1000 10000")
    
    # Formula for cubic/isotropic materials
    E = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)
    nu = c12 / (c11 + c12)
    return E, nu

def verify_7075(config_path):
    logger.info(f"\n" + "="*60)
    logger.info(f"Verifying AA 7075 Potential: {config_path}")
    logger.info("="*60)
    
    config, _ = load_configuration(config_path)
    comp, sel, a_m, n_rep, pot = get_derived_quantities(config)

    # pair_coeff strings
    library_elements = " ".join(e.symbol for e in pot["elements"])
    active_elements = " ".join(e.symbol for e in sel)

    L = lammps(cmdargs=["-log", "none", "-screen", "none"])
    L.command("units metal")
    L.command("atom_style atomic")
    L.command("boundary p p p")

    # Use a slightly larger box for better statistics if needed, 
    # but the config already specifies box_size_m.
    # 4nm is about 10x10x10 cells, which is 4000 atoms for FCC.
    
    L.command(f"lattice {sel[0].lattice_type} {a_m:.4f}")
    L.command(f"region box block 0 {n_rep} 0 {n_rep} 0 {n_rep}")
    L.command(f"create_box {len(sel)} box")
    L.command("create_atoms 1 box")

    # Seed for random distribution
    np.random.seed(42)
    
    # Randomly assign types based on composition
    remaining = 1.0
    for i, elem in enumerate(sel[1:], start=2):
        frac = comp[elem.symbol] / remaining
        L.command(f"set type 1 type/fraction {i} {frac:.6f} {np.random.randint(1, 100000)}")
        remaining -= comp[elem.symbol]

    L.command("pair_style meam")
    L.command(
        f"pair_coeff * * {pot['library']} {library_elements} "
        f"{pot['params']} {active_elements}"
    )

    for i, elem in enumerate(sel, start=1):
        L.command(f"mass {i} {elem.mass}")

    logger.info("Relaxing ground state...")
    L.command("minimize 1.0e-6 1.0e-8 1000 10000")

    E, nu = get_elastic_moduli(L)

    logger.info(f"\nResults for AA 7075 Detailed Composition:")
    logger.info(f"  Young's Modulus (E): {E:.2f} GPa")
    logger.info(f"  Poisson's Ratio (nu): {nu:.3f}")

    # Literature values for AA 7075-T6
    E_lit = 71.7
    nu_lit = 0.33

    logger.info(f"\nLiterature Comparison:")
    logger.info(f"  Target E:  {E_lit} GPa (Diff: {((E-E_lit)/E_lit)*100:.1f}%)")
    logger.info(f"  Target nu: {nu_lit} (Diff: {((nu-nu_lit)/nu_lit)*100:.1f}%)")
    
    L.close()
    
    return E, nu

if __name__ == "__main__":
    # Ensure TK_SILENT is set to avoid dialogs
    os.environ["TK_SILENT"] = "1"
    config_path = os.path.join(project_root, "src", "configs", "AL7075_detailed.json")
    verify_7075(config_path)
