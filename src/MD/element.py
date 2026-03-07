import os
import re


class Element:
    """Represents a MEAM element with its physical properties."""

    def __init__(self, symbol, lattice_type, coord_number, atomic_number,
                 mass, lattice_constant, meam_index):
        self.symbol = symbol
        self.lattice_type = lattice_type
        self.coord_number = coord_number
        self.atomic_number = atomic_number
        self.mass = mass
        self.lattice_constant = lattice_constant
        self.meam_index = meam_index

    def __repr__(self):
        return f"Element({self.symbol}, {self.lattice_type}, a={self.lattice_constant} A)"


def parse_meam_library(path):
    """Parse a library_*.meam file, return ordered list of Elements.

    Each element block is 3 lines:
      line 1: 'Sym' 'lat' coord atomic_num mass
      line 2: alpha b0 b1 b2 b3 alat esub asub
      line 3: t0 t1 t2 t3 rozero ibar
    """
    elements = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

    i = 0
    idx = 1
    while i + 2 < len(lines):
        header = lines[i]
        params = lines[i + 1].split()
        # line 3 unused here
        # Parse header: 'Sym' 'lat' coord atomic_num mass
        parts = header.split()
        symbol = parts[0].strip("'\"")
        lattice_type = parts[1].strip("'\"")
        coord_number = int(parts[2])
        atomic_number = int(parts[3])
        mass = float(parts[4])
        lattice_constant = float(params[5])  # alat is at index 5

        elements.append(Element(
            symbol=symbol,
            lattice_type=lattice_type,
            coord_number=coord_number,
            atomic_number=atomic_number,
            mass=mass,
            lattice_constant=lattice_constant,
            meam_index=idx,
        ))
        idx += 1
        i += 3

    return elements


def scan_eam_dir(eam_dir):
    """Scan EAM directory for library_*.meam + matching *.meam pairs.

    Returns dict keyed by potential name:
      {
        "FeMnNiTiCuCrCoAl": {
          "library": "/path/library_FeMnNiTiCuCrCoAl.meam",
          "params":  "/path/FeMnNiTiCuCrCoAl.meam",
          "elements": [Element, ...],
        },
        ...
      }
    """
    potentials = {}
    for fname in os.listdir(eam_dir):
        m = re.match(r'^library_(.+)\.meam$', fname)
        if not m:
            continue
        name = m.group(1)
        library_path = os.path.join(eam_dir, fname)
        params_path = os.path.join(eam_dir, f"{name}.meam")
        if not os.path.isfile(params_path):
            continue
        elements = parse_meam_library(library_path)
        potentials[name] = {
            "library": library_path,
            "params": params_path,
            "elements": elements,
        }
    return potentials


# Default EAM directory (two levels up from src/MD/element.py)
_EAM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "EAM"))

# Scan all potentials at import time
POTENTIALS = scan_eam_dir(_EAM_DIR) if os.path.isdir(_EAM_DIR) else {}

# Merged lookup by symbol (all elements from all potentials)
ELEMENTS = {}
for _pot in POTENTIALS.values():
    for _elem in _pot["elements"]:
        if _elem.symbol not in ELEMENTS:
            ELEMENTS[_elem.symbol] = _elem
