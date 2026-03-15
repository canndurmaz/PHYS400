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
    """Find a MEAM potential that covers all requested element symbols."""
    needed = set(symbols)
    for name, pot in POTENTIALS.items():
        available = {e.symbol for e in pot["elements"]}
        if needed <= available:
            return pot
    raise RuntimeError(f"No MEAM potential covers elements: {sorted(needed)}")

def get_derived_quantities(config):
    # Filter out elements with zero fraction
    comp = {sym: frac for sym, frac in config["composition"].items() if frac > 0}
    sel = sorted(
        [ELEMENTS[sym] for sym in comp],
        key=lambda e: e.meam_index,
    )

    # Weighted average lattice constant
    a_m = sum(comp[e.symbol] * e.lattice_constant for e in sel)

    # Box repeats: convert box_size from meters to Angstroms, divide by lattice constant
    box_ang = config["box_size_m"] * 1e10
    n_rep = max(1, round(box_ang / a_m))
    
    pot = find_potential(comp.keys())
    
    return comp, sel, a_m, n_rep, pot

# For backwards compatibility with single-run usage
if __name__ == "config": # If imported
    # We don't automatically load here to avoid multiple dialogs
    # Users should call the functions explicitly
    pass
else:
    # If run as a script
    CONFIG, CONFIG_NAME = load_configuration()
    composition, selected, a_mean, n_repeats, potential = get_derived_quantities(CONFIG)
