from element import ELEMENTS

# ── Configuration ──────────────────────────────────────────────
CONFIG = {
    "composition": {"Cu": 0.5, "Al": 0.5},  # element symbol -> fraction (must sum to 1.0)
    "box_size_m": 5e-9,                       # box side length in meters
    "temperature": 300.0,                      # K
    "total_steps": 1000,                       # total simulation steps
    "thermo_interval": 10,                     # thermo output interval
    "dump_interval": 50,                       # trajectory dump interval
}

# ── Derived quantities ─────────────────────────────────────────
composition = CONFIG["composition"]
selected = sorted(
    [ELEMENTS[sym] for sym in composition],
    key=lambda e: e.meam_index,
)

# Weighted average lattice constant
a_mean = sum(composition[e.symbol] * e.lattice_constant for e in selected)

# Box repeats: convert box_size from meters to Angstroms, divide by lattice constant
box_angstrom = CONFIG["box_size_m"] * 1e10
n_repeats = max(1, round(box_angstrom / a_mean))
