import json
import os
import sys
import tkinter as tk
from tkinter import filedialog

from element import ELEMENTS, POTENTIALS

# -- Configuration ------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

_DEFAULTS = {
    "composition": {"Cu": 0.5, "Al": 0.5},
    "box_size_m": 5e-9,
    "temperature": 300.0,
    "total_steps": 1000,
    "thermo_interval": 10,
    "dump_interval": 50,
}

# Accept config path as CLI argument; fall back to file dialog
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    _config_path = sys.argv[1]
else:
    _root = tk.Tk()
    _root.withdraw()
    _config_path = filedialog.askopenfilename(
        title="Select configuration JSON",
        initialdir=_SRC_DIR,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
    )
    _root.destroy()

if _config_path:
    with open(_config_path) as _f:
        CONFIG = json.load(_f)
else:
    CONFIG = _DEFAULTS.copy()

# -- Derived quantities --------------------------------------------------------
# Filter out elements with zero fraction
composition = {sym: frac for sym, frac in CONFIG["composition"].items() if frac > 0}
selected = sorted(
    [ELEMENTS[sym] for sym in composition],
    key=lambda e: e.meam_index,
)

# Weighted average lattice constant
a_mean = sum(composition[e.symbol] * e.lattice_constant for e in selected)

# Box repeats: convert box_size from meters to Angstroms, divide by lattice constant
box_angstrom = CONFIG["box_size_m"] * 1e10
n_repeats = max(1, round(box_angstrom / a_mean))


# -- Potential auto-detection --------------------------------------------------
def find_potential(symbols):
    """Find a MEAM potential that covers all requested element symbols.

    Returns the matching potential dict (library, params, elements) or raises.
    """
    needed = set(symbols)
    for name, pot in POTENTIALS.items():
        available = {e.symbol for e in pot["elements"]}
        if needed <= available:
            return pot
    raise RuntimeError(
        f"No MEAM potential covers all selected elements: {sorted(needed)}. "
        f"Available potentials: "
        + ", ".join(
            f"{n} ({', '.join(e.symbol for e in p['elements'])})"
            for n, p in POTENTIALS.items()
        )
    )


potential = find_potential(composition.keys())
