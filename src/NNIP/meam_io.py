"""Parse and write MEAM library/params files for optimization."""

import os
import re
import tempfile
import numpy as np


# ── Library file parsing ─────────────────────────────────────────────────────

# Per-element parameter names in library file order
_LIB_LINE2 = ("alpha", "b0", "b1", "b2", "b3", "alat", "esub", "asub")
_LIB_LINE3 = ("t0", "t1", "t2", "t3", "rozero", "ibar")


def parse_library(path):
    """Parse a library_*.meam file into a dict of element data.

    Returns:
        dict[str, dict] keyed by element symbol.  Each value contains:
          header:  (lattice_type, coord, atomic_num, mass)
          params:  dict of all 14 numeric params by name
    """
    elements = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    i = 0
    while i + 2 < len(lines):
        # Line 1: 'Sym' 'lat' coord atomic_num mass
        parts = lines[i].split()
        symbol = parts[0].strip("'\"")
        lattice_type = parts[1].strip("'\"")
        coord = int(float(parts[2]))
        atomic_num = int(float(parts[3]))
        mass = float(parts[4])

        # Line 2: alpha b0 b1 b2 b3 alat esub asub
        vals2 = [float(x) for x in lines[i + 1].split()]
        # Line 3: t0 t1 t2 t3 rozero ibar
        vals3 = lines[i + 2].split()

        params = {}
        for name, val in zip(_LIB_LINE2, vals2):
            params[name] = val
        for name, raw in zip(_LIB_LINE3, vals3):
            # ibar is integer but stored as float for uniformity
            params[name] = float(raw)

        elements[symbol] = {
            "header": (lattice_type, coord, atomic_num, mass),
            "params": params,
        }
        i += 3

    return elements


# ── Params file parsing ──────────────────────────────────────────────────────

def parse_params(path):
    """Parse a *.meam params file into an ordered list of (key, value) tuples.

    Keys like 'Cmin(1,2,3)' are kept as-is.  Values are floats or strings
    (for lattce entries like 'l12').
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Try float, fall back to string (e.g. 'l12')
            try:
                val = float(val)
            except ValueError:
                val = val.strip("'\"")
            entries.append((key, val))
    return entries


# ── Writers ──────────────────────────────────────────────────────────────────

def write_library(lib_data, element_order, path):
    """Write a MEAM library file.

    Args:
        lib_data: dict from parse_library()
        element_order: list of element symbols in desired order
        path: output file path
    """
    with open(path, "w") as f:
        for sym in element_order:
            info = lib_data[sym]
            lat, coord, anum, mass = info["header"]
            p = info["params"]
            f.write(f"'{sym}' '{lat}' {coord} {anum} {mass:.6f}\n")
            f.write(
                f"{p['alpha']:.6f} {p['b0']:.6f} {p['b1']:.6f} "
                f"{p['b2']:.6f} {p['b3']:.6f} {p['alat']:.6f} "
                f"{p['esub']:.6f} {p['asub']:.6f}\n"
            )
            f.write(
                f"{p['t0']:.6f} {p['t1']:.6f} {p['t2']:.6f} "
                f"{p['t3']:.6f} {p['rozero']:.6f} {int(p['ibar'])}\n"
            )
            f.write("\n")


def write_params(entries, path):
    """Write a MEAM params file from a list of (key, value) tuples."""
    with open(path, "w") as f:
        for key, val in entries:
            if isinstance(val, str):
                f.write(f"{key}='{val}'\n")
            else:
                f.write(f"{key}={val:.6f}\n")


# ── Vector ↔ Files conversion ────────────────────────────────────────────────

def params_to_vector(lib_data, param_entries, opt_spec):
    """Flatten selected parameters into a 1-D numpy array.

    Args:
        lib_data: dict from parse_library()
        param_entries: list from parse_params()
        opt_spec: dict describing what to optimize:
            {
                "library": {"Al": ["alpha", "b0", ...], "Zn": [...]},
                "params": ["Ec(1,3)", "re(1,3)", ...]
            }

    Returns:
        (vector, names) — the parameter vector and corresponding names.
    """
    vec = []
    names = []

    # Library parameters
    for sym, param_names in opt_spec.get("library", {}).items():
        for pname in param_names:
            vec.append(lib_data[sym]["params"][pname])
            names.append(f"lib:{sym}:{pname}")

    # Params-file parameters
    param_dict = {k: v for k, v in param_entries}
    for pname in opt_spec.get("params", []):
        vec.append(param_dict[pname])
        names.append(f"par:{pname}")

    return np.array(vec, dtype=np.float64), names


def vector_to_files(vec, names, base_lib, base_params, out_dir):
    """Write temporary MEAM files from an optimization vector.

    Args:
        vec: parameter vector (np array)
        names: list of parameter names from params_to_vector()
        base_lib: dict from parse_library() (will be deep-copied internally)
        base_params: list from parse_params() (will be copied internally)
        out_dir: directory to write files into

    Returns:
        (lib_path, params_path) — paths to the written files
    """
    import copy
    lib = copy.deepcopy(base_lib)
    entries = list(base_params)  # shallow copy of list of tuples

    for val, name in zip(vec, names):
        if name.startswith("lib:"):
            _, sym, pname = name.split(":")
            lib[sym]["params"][pname] = float(val)
        elif name.startswith("par:"):
            pkey = name[4:]  # strip "par:"
            entries = [(k, float(val) if k == pkey else v) for k, v in entries]

    element_order = list(base_lib.keys())
    lib_path = os.path.join(out_dir, "library.meam")
    params_path = os.path.join(out_dir, "params.meam")

    write_library(lib, element_order, lib_path)
    write_params(entries, params_path)

    return lib_path, params_path
