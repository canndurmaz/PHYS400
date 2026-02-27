"""
elastic.py — Full 6×6 elastic tensor for Fe-V alloy via LAMMPS central finite differences.

Method: Apply each of 6 Voigt strains ±delta, minimize atoms (fixed box),
        read 6-component stress, compute Cij = dσi/dεj.
        VRH averaging → E, K, G, ν.

Units: LAMMPS metal units (pressure in bar → GPa via ×1e-4).
"""

import os
import sys
import numpy as np
from lammps import lammps

# ---------------------------------------------------------------------------
# CONFIG — edit here to change alloy / supercell / parameters
# ---------------------------------------------------------------------------
CONFIG = {
    # EAM/FS potential — header order determines type mapping
    "potential_file": "/home/kenobi/Workspaces/PHYS400/EAM/VFe_mm.eam.fs",
    "potential_elements": "V Fe",          # matches header in potential file
    # Lattice
    "lattice_const": 2.87,                 # Å, BCC Fe ≈ 2.87
    "supercell": (5, 5, 5),               # repetitions → ~250 atoms
    # Alloy composition: type 2 (Fe) → randomly assign fraction as type 1 (V)
    "host_type": 2,                        # Fe (majority)
    "dopant_type": 1,                      # V  (minority)
    "dopant_fraction": 0.10,               # 10 at.% V
    "random_seed": 42,
    # Strain perturbation magnitude (dimensionless)
    "delta": 1e-3,
    # Minimisation thresholds
    "etol": 1e-12,
    "ftol": 1e-12,
    "maxiter": 10000,
    "maxeval": 100000,
    # Output
    "frames_file": "elastic_frames.lammpstrj",
    "cij_file": "Cij.npy",
    "moduli_file": "moduli.npy",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lammps_init(silent: bool = True) -> lammps:
    """Create a silent LAMMPS instance."""
    args = ["-log", "none", "-screen", "none"] if silent else ["-log", "none"]
    return lammps(cmdargs=args)


def get_stress_gpa(L: lammps) -> np.ndarray:
    """Return 6-component Voigt stress in GPa (σ = −P)."""
    keys = ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]
    return np.array([-L.get_thermo(k) for k in keys]) * 1e-4  # bar → GPa


def minimize_atoms(L: lammps, cfg: dict) -> None:
    """CG minimise atom positions with box shape held fixed."""
    L.command(
        f"minimize {cfg['etol']} {cfg['ftol']} {cfg['maxiter']} {cfg['maxeval']}"
    )


# ---------------------------------------------------------------------------
# Ground-state builder
# ---------------------------------------------------------------------------

def build_ground_state(cfg: dict) -> lammps:
    """
    Create and fully relax the Fe-V alloy supercell.

    Relaxation protocol:
      1. iso box relax (pressure → 0)
      2. tri box relax (all stress components → 0)
    Returns a LAMMPS instance sitting at the relaxed ground state.
    """
    L = _lammps_init()

    nx, ny, nz = cfg["supercell"]
    a = cfg["lattice_const"]
    pot = cfg["potential_file"]
    elems = cfg["potential_elements"]
    seed = cfg["random_seed"]
    frac = cfg["dopant_fraction"]
    host = cfg["host_type"]
    dopant = cfg["dopant_type"]

    cmds = [
        # Setup
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        # BCC lattice with all atoms as host (Fe = type 2)
        f"lattice bcc {a}",
        f"region box block 0 {nx} 0 {ny} 0 {nz}",
        "create_box 2 box",
        f"create_atoms {host} box",
        # Assign dopant_fraction of host atoms as dopant (V = type 1)
        f"set type {host} type/fraction {dopant} {frac} {seed}",
        # Masses (Fe ≈ 55.845, V ≈ 50.942)
        "mass 1 50.942",
        "mass 2 55.845",
        # Potential
        f"pair_style eam/fs",
        f"pair_coeff * * {pot} {elems}",
        # Thermo
        "thermo 100",
        "thermo_style custom step pe pxx pyy pzz pyz pxz pxy",
        # --- Stage 1: isotropic relax ---
        "fix 1 all box/relax iso 0.0 vmax 0.001",
        f"minimize {cfg['etol']} {cfg['ftol']} {cfg['maxiter']} {cfg['maxeval']}",
        "unfix 1",
        # Convert to triclinic before shear relaxation
        "change_box all triclinic",
        # --- Stage 2: full triclinic relax ---
        "fix 2 all box/relax tri 0.0 vmax 0.001",
        f"minimize {cfg['etol']} {cfg['ftol']} {cfg['maxiter']} {cfg['maxeval']}",
        "unfix 2",
        # Reset timestep
        "reset_timestep 0",
        # run 0 to tally thermo on current timestep
        "run 0",
    ]
    for cmd in cmds:
        L.command(cmd)

    print(f"  Ground state: PE = {L.get_thermo('pe'):.6f} eV")
    return L


def _dump_state(L: lammps, fname: str, append: bool) -> None:
    """Write current configuration to a LAMMPS dump file."""
    mode = "append yes" if append else "append no"
    L.command(
        f"write_dump all custom {fname} id type x y z "
        f"modify {mode} sort id"
    )


# ---------------------------------------------------------------------------
# Strain application
# ---------------------------------------------------------------------------

def apply_normal_strain(L: lammps, axis: str, scale: float) -> None:
    """Scale a box dimension by `scale` (units box, with remap)."""
    L.command(f"change_box all {axis} scale {scale} remap units box")


def apply_shear_strain(L: lammps, tilt: str, delta_tilt: float) -> None:
    """Shift a tilt component by `delta_tilt` Å (units box, with remap)."""
    L.command(f"change_box all {tilt} delta {delta_tilt} remap units box")


# ---------------------------------------------------------------------------
# Per-column central difference
# ---------------------------------------------------------------------------

def compute_elastic_column(
    j: int,
    cfg: dict,
    frames_file: str,
    first_col: bool,
) -> np.ndarray:
    """
    Compute column j of the elastic tensor via central finite differences.

    Returns sigma_plus and sigma_minus (each 6-component Voigt, GPa).
    Also appends 3 frames to frames_file: reference, +delta, -delta.
    """
    delta = cfg["delta"]

    # ---- Build a fresh relaxed instance for this column ----
    print(f"  Building ground state for column j={j} ...")
    L = build_ground_state(cfg)

    # Retrieve reference box dimensions after relaxation
    Lx = L.get_thermo("lx")
    Ly = L.get_thermo("ly")
    Lz = L.get_thermo("lz")

    # Frame 0: reference (append if not first column)
    _dump_state(L, frames_file, append=not first_col)

    # Voigt index → (type, label, dimension char or tilt char)
    normal_axes = {0: "x", 1: "y", 2: "z"}
    shear_tilts = {3: ("yz", Lz), 4: ("xz", Lz), 5: ("xy", Ly)}

    def apply_strain(sign: float) -> None:
        d = sign * delta
        if j in normal_axes:
            ax = normal_axes[j]
            scale = 1.0 + d
            apply_normal_strain(L, ax, scale)
        else:
            tilt, L_ref = shear_tilts[j]
            apply_shear_strain(L, tilt, d * L_ref)

    def undo_strain(sign_applied: float) -> None:
        if j in normal_axes:
            ax = normal_axes[j]
            # Inverse scale to reach opposite sign
            # After applying (1+d), apply (1-d)/(1+d) to reach (1-d)
            d = sign_applied * delta
            undo_scale = (1.0 - sign_applied * delta) / (1.0 + sign_applied * delta)
            apply_normal_strain(L, ax, undo_scale)
        else:
            tilt, L_ref = shear_tilts[j]
            # Undo +d then apply -d → net shift of -2d*L_ref
            d = sign_applied * delta
            apply_shear_strain(L, tilt, -2.0 * sign_applied * delta * L_ref)

    # +delta
    apply_strain(+1.0)
    minimize_atoms(L, cfg)
    _dump_state(L, frames_file, append=True)
    sigma_plus = get_stress_gpa(L)

    # Go from +delta to -delta
    undo_strain(+1.0)
    minimize_atoms(L, cfg)

    # -delta
    _dump_state(L, frames_file, append=True)
    sigma_minus = get_stress_gpa(L)

    L.close()
    return sigma_plus, sigma_minus


# ---------------------------------------------------------------------------
# Full tensor
# ---------------------------------------------------------------------------

def compute_elastic_tensor(cfg: dict) -> tuple[np.ndarray, None]:
    """
    Compute full 6×6 Cij matrix (GPa) via 6 independent perturbations.
    Also writes elastic_frames.lammpstrj with 3 frames per column (18 total).
    """
    delta = cfg["delta"]
    frames_file = cfg["frames_file"]
    C = np.zeros((6, 6))

    # Remove stale frames file
    if os.path.exists(frames_file):
        os.remove(frames_file)

    for j in range(6):
        print(f"\n[Column {j+1}/6] Voigt index j={j}")
        sigma_plus, sigma_minus = compute_elastic_column(
            j, cfg, frames_file, first_col=(j == 0)
        )
        C[:, j] = (sigma_plus - sigma_minus) / (2.0 * delta)
        print(f"  C[:,{j}] = {C[:, j]}")

    # Symmetrize
    C = (C + C.T) / 2.0
    return C


# ---------------------------------------------------------------------------
# Voigt-Reuss-Hill averaging
# ---------------------------------------------------------------------------

def voigt_reuss_hill(C: np.ndarray) -> dict:
    """
    Compute polycrystalline moduli via VRH averaging.
    Returns dict with E, K_V, K_R, K_H, G_V, G_R, G_H, nu (all GPa except nu).
    """
    S = np.linalg.inv(C)

    # Voigt bounds
    K_V = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9.0
    G_V = (C[0,0]+C[1,1]+C[2,2] - C[0,1]-C[0,2]-C[1,2] + 3*(C[3,3]+C[4,4]+C[5,5])) / 15.0

    # Reuss bounds
    K_R = 1.0 / (S[0,0]+S[1,1]+S[2,2] + 2*(S[0,1]+S[0,2]+S[1,2]))
    G_R = 15.0 / (4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[0,2]+S[1,2]) + 3*(S[3,3]+S[4,4]+S[5,5]))

    # Hill averages
    K_H = (K_V + K_R) / 2.0
    G_H = (G_V + G_R) / 2.0

    E  = 9.0 * K_H * G_H / (3.0 * K_H + G_H)
    nu = (3.0 * K_H - 2.0 * G_H) / (2.0 * (3.0 * K_H + G_H))

    return dict(E=E, K_V=K_V, K_R=K_R, K_H=K_H, G_V=G_V, G_R=G_R, G_H=G_H, nu=nu)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

VOIGT_LABELS = ["xx", "yy", "zz", "yz", "xz", "xy"]


def print_results(C: np.ndarray, moduli: dict) -> None:
    """Pretty-print Cij matrix and polycrystalline moduli."""
    print("\n" + "="*60)
    print("  Elastic Tensor Cij (GPa)")
    print("="*60)
    header = "     " + "".join(f"  {l:>6}" for l in VOIGT_LABELS)
    print(header)
    for i in range(6):
        row = f"  {VOIGT_LABELS[i]}  " + "".join(f"  {C[i,j]:6.1f}" for j in range(6))
        print(row)

    print("\n" + "="*60)
    print("  Polycrystalline Moduli — Voigt-Reuss-Hill")
    print("="*60)
    print(f"  E   (Young's)     = {moduli['E']:7.2f} GPa")
    print(f"  K_V (bulk Voigt)  = {moduli['K_V']:7.2f} GPa")
    print(f"  K_R (bulk Reuss)  = {moduli['K_R']:7.2f} GPa")
    print(f"  K_H (bulk Hill)   = {moduli['K_H']:7.2f} GPa")
    print(f"  G_V (shear Voigt) = {moduli['G_V']:7.2f} GPa")
    print(f"  G_R (shear Reuss) = {moduli['G_R']:7.2f} GPa")
    print(f"  G_H (shear Hill)  = {moduli['G_H']:7.2f} GPa")
    print(f"  ν   (Poisson)     = {moduli['nu']:7.4f}")

    # Positive-definite check
    eigvals = np.linalg.eigvalsh(C)
    pd = np.all(eigvals > 0)
    print(f"\n  Positive-definite: {pd}  (eigenvalues: {eigvals})")

    # Hill bounds check
    ok = moduli["G_R"] <= moduli["G_H"] <= moduli["G_V"]
    print(f"  Hill bounds G_R ≤ G_H ≤ G_V: {ok}")
    print("="*60)


def save_results(C: np.ndarray, moduli: dict, cfg: dict) -> None:
    np.save(cfg["cij_file"], C)
    np.save(cfg["moduli_file"], moduli)
    print(f"\n  Saved: {cfg['cij_file']}, {cfg['moduli_file']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = CONFIG
    print("=" * 60)
    print("  Fe-V Elastic Tensor Calculator")
    print(f"  Supercell: {cfg['supercell']}, δ = {cfg['delta']}")
    print(f"  V fraction: {cfg['dopant_fraction']*100:.1f} at.%")
    print("=" * 60)

    C = compute_elastic_tensor(cfg)
    moduli = voigt_reuss_hill(C)
    print_results(C, moduli)
    save_results(C, moduli, cfg)


if __name__ == "__main__":
    main()
