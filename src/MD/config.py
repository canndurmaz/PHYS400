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

def load_configuration(path=None):
    """Load a single configuration from a path or return defaults."""
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f), os.path.basename(path)
    
    # If no path provided and not in silent mode, open a dialog for one file
    if os.environ.get("TK_SILENT") != "1":
        try:
            _root = tk.Tk()
            _root.withdraw()
            _config_path = filedialog.askopenfilename(
                title="Select configuration JSON",
                initialdir=_SRC_DIR,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            _root.destroy()
            if _config_path and os.path.isfile(_config_path):
                with open(_config_path) as _f:
                    return json.load(_f), os.path.basename(_config_path)
        except:
            pass
            
    return _DEFAULTS.copy(), "default"

def select_multiple_configs():
    """Open a dialog to select multiple configuration files."""
    if os.environ.get("TK_SILENT") == "1":
        return []
        
    try:
        _root = tk.Tk()
        _root.withdraw()
        _paths = filedialog.askopenfilenames(
            title="Select one or more configuration JSONs",
            initialdir=_SRC_DIR,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        _root.destroy()
        return list(_paths)
    except:
        return []

# -- Potential auto-detection --------------------------------------------------
def find_potential(symbols):
    """Find the best MEAM potential that covers all requested element symbols.

    Prefers the potential with the fewest extra elements (best fit).
    """
    needed = set(symbols)
    best = None
    best_extra = float("inf")
    for name, pot in POTENTIALS.items():
        available = {e.symbol for e in pot["elements"]}
        if needed <= available:
            extra = len(available) - len(needed)
            if extra < best_extra:
                best = pot
                best_extra = extra
    if best is None:
        all_available = set()
        for pot in POTENTIALS.values():
            all_available.update(e.symbol for e in pot["elements"])
        missing = needed - all_available
        if missing:
            raise RuntimeError(
                f"Elements not found in any EAM potential: {sorted(missing)}. "
                f"Available: {sorted(all_available)}"
            )
        raise RuntimeError(
            f"No single MEAM potential covers all of {sorted(needed)}. "
            f"Available potentials cover: "
            + ", ".join(
                f"{name}: {{{', '.join(sorted(e.symbol for e in pot['elements']))}}}"
                for name, pot in POTENTIALS.items()
            )
        )
    return best

def get_derived_quantities(config):
    # Filter out elements with zero fraction
    comp = {sym: frac for sym, frac in config["composition"].items() if frac > 0}

    pot = find_potential(comp.keys())

    # Use element data from the selected potential (not the global ELEMENTS dict)
    pot_elements = {e.symbol: e for e in pot["elements"]}
    sel = sorted(
        [pot_elements[sym] for sym in comp],
        key=lambda e: e.meam_index,
    )

    # Dominant element determines the lattice type and base lattice constant
    dominant = max(sel, key=lambda e: comp[e.symbol])

    # Weighted average lattice constant
    a_m = sum(comp[e.symbol] * e.lattice_constant for e in sel)

    # Box repeats: convert box_size from meters to Angstroms, divide by lattice constant
    box_ang = config["box_size_m"] * 1e10
    n_rep = max(1, round(box_ang / a_m))

    return comp, sel, a_m, n_rep, pot, dominant

# Only auto-load when run directly as a script
if __name__ == "__main__":
    CONFIG, CONFIG_NAME = load_configuration()
    composition, selected, a_mean, n_repeats, potential = get_derived_quantities(CONFIG)
