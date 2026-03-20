#!/usr/bin/env python3
"""Generalized MEAM potential merger.

Combines elements from multiple source files and literature data into a single 
multi-element potential based on a JSON configuration.

Usage:
    python merge_potentials.py --config src/configs/meam_merge_7075.json
"""

import argparse
import json
import math
import os
import re
import sys

# Allow running as script or module
if __name__ == "__main__" and __package__ is None:
    # Add project root to sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Correct the import to match project structure
from src.NNIP.meam_io import parse_library, parse_params, write_library, write_params


# ── Atomic number → standard symbol lookup ───────────────────────────────────
_Z_TO_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 40: "Zr", 41: "Nb",
    42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In",
    50: "Sn", 51: "Sb", 73: "Ta", 74: "W", 75: "Re", 77: "Ir", 78: "Pt",
    79: "Au", 82: "Pb",
}


_KNOWN_SYMBOLS = set(_Z_TO_SYMBOL.values())


def _canonical_symbol(lib_name, header):
    """Map a library element name to a standard chemical symbol.

    Some MEAM library files use non-standard names (e.g. 'SiS', 'AlS') to
    distinguish parameter sets from different publications.  We resolve the
    standard symbol via the atomic number stored in the header.

    If the raw name is already a recognized element symbol, trust it — some
    library files store dummy atomic numbers (e.g. Z=1 for all elements).
    """
    if lib_name in _KNOWN_SYMBOLS:
        return lib_name
    # Non-standard name — try resolving via atomic number
    std = _Z_TO_SYMBOL.get(header[2])
    if std:
        return std
    return lib_name


# ── Nearest-neighbor distance helpers ─────────────────────────────────────────

def nn_distance(lattice, alat):
    """Return nearest-neighbor distance for a given lattice type and parameter."""
    lat = lattice.lower()
    if lat == "fcc":
        return alat / math.sqrt(2)
    elif lat == "bcc":
        return alat * math.sqrt(3) / 2
    elif lat in ("hcp", "hex"):
        return alat  # a parameter is the nn distance in HCP
    elif lat in ("dia", "diamond"):
        return alat * math.sqrt(3) / 4
    elif lat == "b1":
        return alat / 2
    elif lat == "b2":
        return alat * math.sqrt(3) / 2
    else:
        return alat  # fallback


# ── Index remapping ──────────────────────────────────────────────────────────

_KEY_RE = re.compile(r"^(\w+)\(([^)]+)\)$")


def _parse_key(key):
    """Parse 'name(i,j,...)' into (name, [indices]) or (key, None) for globals."""
    m = _KEY_RE.match(key)
    if not m:
        return key, None
    name = m.group(1)
    indices = [int(x.strip()) for x in m.group(2).split(",")]
    return name, indices


def _make_key(name, indices):
    """Reconstruct 'name(i,j,...)' from name and index list."""
    return f"{name}({','.join(str(i) for i in indices)})"


def remap_params(entries, index_map, drop_indices):
    """Remap indexed params entries from source to new element ordering.

    Args:
        entries: list of (key, value) from parse_params()
        index_map: {old_index: new_index} for kept elements
        drop_indices: set of old indices to skip

    Returns:
        dict of {new_key: value}
    """
    result = {}
    for key, val in entries:
        name, indices = _parse_key(key)

        if indices is None:
            # Global param
            result[key] = val
            continue

        # Skip entries involving dropped elements
        if any(idx in drop_indices for idx in indices):
            continue

        # Skip if any index is not in the map
        if any(idx not in index_map for idx in indices):
            continue

        new_indices = [index_map[idx] for idx in indices]

        # For 2-index pair params, normalize to i < j
        if len(new_indices) == 2:
            new_indices.sort()

        new_key = _make_key(name, new_indices)
        result[new_key] = val

    return result


# ── Mixing rules for missing binary pairs ────────────────────────────────────

def generate_missing_pair(i, j, target_elements, lib_data, self_screen):
    """Generate MEAM cross-term params for a missing binary pair using mixing rules."""
    sym_i, sym_j = target_elements[i - 1], target_elements[j - 1]
    lib_i, lib_j = lib_data[sym_i], lib_data[sym_j]

    # Cohesive energy: arithmetic mean
    ec_ij = (lib_i["params"]["esub"] + lib_j["params"]["esub"]) / 2

    # Equilibrium distance: average nn distances
    lat_i, lat_j = lib_i["header"][0], lib_j["header"][0]
    nn_i = nn_distance(lat_i, lib_i["params"]["alat"])
    nn_j = nn_distance(lat_j, lib_j["params"]["alat"])
    re_ij = (nn_i + nn_j) / 2

    # Rose equation parameter: arithmetic mean
    alpha_ij = (lib_i["params"]["alpha"] + lib_j["params"]["alpha"]) / 2

    # Choose reference lattice based on pair structures
    lat_pair = {lib_i["header"][0].lower(), lib_j["header"][0].lower()}
    if lat_pair == {"fcc"}:
        lattce = "l12"
    elif lat_pair == {"bcc"}:
        lattce = "b2"
    else:
        lattce = "b1"

    entries = [
        (f"lattce({i},{j})", lattce),
        (f"Ec({i},{j})", round(ec_ij, 6)),
        (f"re({i},{j})", round(re_ij, 6)),
        (f"alpha({i},{j})", round(alpha_ij, 4)),
        (f"attrac({i},{j})", 0.0),
        (f"repuls({i},{j})", 0.0),
        (f"nn2({i},{j})", 1),
    ]

    # Screening params: use geometric mean of self-screening values
    cmin_ii = self_screen.get(f"Cmin({i},{i},{i})", 0.8)
    cmin_jj = self_screen.get(f"Cmin({j},{j},{j})", 0.8)
    cmax_ii = self_screen.get(f"Cmax({i},{i},{i})", 2.8)
    cmax_jj = self_screen.get(f"Cmax({j},{j},{j})", 2.8)

    cmin_mix = round(math.sqrt(cmin_ii * cmin_jj), 4)
    cmax_mix = round(math.sqrt(cmax_ii * cmax_jj), 4)

    # Four Cmin/Cmax triplets per pair
    entries.extend([
        (f"Cmin({i},{i},{j})", round(cmin_ii, 4)),
        (f"Cmin({j},{j},{i})", round(cmin_jj, 4)),
        (f"Cmin({i},{j},{i})", round(cmin_mix, 4)),
        (f"Cmin({i},{j},{j})", round(cmin_mix, 4)),
        (f"Cmax({i},{i},{j})", round(cmax_ii, 4)),
        (f"Cmax({j},{j},{i})", round(cmax_jj, 4)),
        (f"Cmax({i},{j},{i})", round(cmax_mix, 4)),
        (f"Cmax({i},{j},{j})", round(cmax_mix, 4)),
    ])

    return entries


# ── Ordering helper ──────────────────────────────────────────────────────────

def _order_params(params_dict):
    """Order params entries logically: self → binary → ternary → global."""
    self_entries = []
    pair_entries = []
    screen_entries = []
    ternary_entries = []
    global_entries = []

    for key, val in params_dict.items():
        name, indices = _parse_key(key)
        if indices is None:
            global_entries.append((key, val))
            continue

        unique_idx = set(indices)
        if len(unique_idx) == 1:
            self_entries.append((key, val))
        elif len(indices) == 2:
            pair_entries.append((key, val))
        elif len(indices) == 3 and len(unique_idx) == 2:
            screen_entries.append((key, val))
        elif len(indices) == 3 and len(unique_idx) == 3:
            ternary_entries.append((key, val))
        else:
            global_entries.append((key, val))

    def _sort_key(item):
        _, indices = _parse_key(item[0])
        return (indices or [0], item[0])

    self_entries.sort(key=_sort_key)
    pair_entries.sort(key=_sort_key)
    screen_entries.sort(key=_sort_key)
    ternary_entries.sort(key=_sort_key)

    return self_entries + pair_entries + screen_entries + ternary_entries + global_entries


# ── Main merge logic ─────────────────────────────────────────────────────────

def merge_from_config(config_path, eam_dir, output_dir):
    with open(config_path, "r") as f:
        config = json.load(f)

    target_elements = config["target_elements"]
    source_files = config["source_files"]
    lit_elements = config.get("literature_elements", {})
    global_params = config.get("global_params", {})
    output_prefix = config.get("output_prefix", "merged")

    merged_lib = {}
    merged_params = {}

    # 1. Process source files
    for src_file in source_files:
        lib_path = os.path.join(eam_dir, src_file)
        # Find matching params file (e.g. library_AlMgZn.meam -> AlMgZn.meam)
        # Or library.meam -> params.meam? The user might specify them differently.
        # Let's assume the params file is the lib filename minus 'library_' prefix.
        if src_file.startswith("library_"):
            par_file = src_file[len("library_"):]
        else:
            par_file = src_file.replace("library", "params") # Fallback
        
        par_path = os.path.join(eam_dir, par_file)
        
        if not os.path.exists(lib_path):
            print(f"Warning: Library file not found: {lib_path}")
            continue
        
        src_lib = parse_library(lib_path)
        src_elements = list(src_lib.keys())
        
        # Build index map: 1-based source index -> 1-based target index
        index_map = {}
        drop_indices = set()
        for i, sym in enumerate(src_elements, start=1):
            if sym in target_elements:
                index_map[i] = target_elements.index(sym) + 1
                # Add to merged library if not already there
                if sym not in merged_lib:
                    merged_lib[sym] = src_lib[sym]
            else:
                drop_indices.add(i)

        if os.path.exists(par_path):
            src_par_entries = parse_params(par_path)
            remapped = remap_params(src_par_entries, index_map, drop_indices)
            merged_params.update(remapped)
        else:
            print(f"Warning: Params file not found: {par_path}")

    # 2. Add literature elements
    for sym, data in lit_elements.items():
        if sym in target_elements:
            merged_lib[sym] = {
                "header": tuple(data["header"]),
                "params": data["params"]
            }
            # Add literature self_params if any
            for key, val in data.get("self_params", []):
                merged_params[key] = val

    # 3. Identify missing pairs
    all_pairs = set()
    n = len(target_elements)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            all_pairs.add((i, j))

    known_pairs = set()
    for key in merged_params:
        if key.startswith("Ec("):
            _, indices = _parse_key(key)
            if indices and len(indices) == 2:
                known_pairs.add(tuple(sorted(indices)))

    missing_pairs = all_pairs - known_pairs

    # 4. Generate mixing rules for missing pairs
    # Collect self-screening for mixing rules
    self_screen = {}
    for key, val in merged_params.items():
        if isinstance(val, (int, float)):
            name, indices = _parse_key(key)
            if indices and name in ("Cmin", "Cmax") and len(indices) == 3:
                if indices[0] == indices[1] == indices[2]:
                    self_screen[key] = val

    for i, j in sorted(missing_pairs):
        pair_entries = generate_missing_pair(i, j, target_elements, merged_lib, self_screen)
        for key, val in pair_entries:
            merged_params[key] = val

    # 5. Apply global params
    merged_params.update(global_params)

    # 6. Write output
    os.makedirs(output_dir, exist_ok=True)
    lib_out = os.path.join(output_dir, f"library_{output_prefix}.meam")
    par_out = os.path.join(output_dir, f"{output_prefix}.meam")

    write_library(merged_lib, target_elements, lib_out)
    write_params(_order_params(merged_params), par_out)

    print(f"Merged potential written to {output_dir}")
    print(f"  Library: {lib_out}")
    print(f"  Params:  {par_out}")
    print(f"Elements: {' '.join(target_elements)}")


def auto_merge(elements, eam_dir, output_dir):
    """Auto-discover and merge MEAM potentials for the requested elements.

    Scans all library_*.meam files in eam_dir, pulls each element from the
    first source that has it, remaps params, generates missing cross-terms,
    and writes merged output files.  No config file needed.

    Args:
        elements: list of element symbols in desired order
        eam_dir: directory containing library_*.meam + matching *.meam files
        output_dir: directory to write merged output

    Returns:
        (lib_path, params_path) — paths to the written files

    Raises:
        FileNotFoundError: if not all elements are available across source files
    """
    # 1. Discover all library files and their elements
    # Some files use non-standard names (e.g. 'SiS' for Si) — resolve via atomic number.
    sources = []  # [(lib_path, par_path, lib_data, raw_names, alias_map)]
    for fname in sorted(os.listdir(eam_dir)):
        if not (fname.startswith("library_") and fname.endswith(".meam")):
            continue
        lib_path = os.path.join(eam_dir, fname)
        par_file = fname[len("library_"):]
        par_path = os.path.join(eam_dir, par_file)
        try:
            lib_data = parse_library(lib_path)
        except Exception:
            continue
        raw_names = list(lib_data.keys())
        # alias_map: canonical_symbol -> raw_name_in_file
        alias_map = {}
        for raw in raw_names:
            canon = _canonical_symbol(raw, lib_data[raw]["header"])
            alias_map[canon] = raw
        sources.append((lib_path, par_path, lib_data, raw_names, alias_map))

    # 2. Check that all requested elements are available (using canonical names)
    available = set()
    for _, _, _, _, alias_map in sources:
        available.update(alias_map.keys())

    missing = set(elements) - available
    if missing:
        raise FileNotFoundError(
            f"Elements {sorted(missing)} not found in any library_*.meam file in {eam_dir}. "
            f"Available elements: {sorted(available)}"
        )

    # 3. Build merged library — pull each element from the first source that has it
    #    Re-key aliased names to standard symbols so the merged file is clean.
    merged_lib = {}
    for sym in elements:
        for _, _, lib_data, _, alias_map in sources:
            if sym in alias_map:
                raw = alias_map[sym]
                merged_lib[sym] = lib_data[raw]
                break

    # 4. Build merged params — remap from each source
    merged_params = {}
    for lib_path, par_path, lib_data, src_elements, alias_map in sources:
        if not os.path.exists(par_path):
            continue

        # Map raw source names to canonical symbols for this source
        src_canonical = [_canonical_symbol(raw, lib_data[raw]["header"]) for raw in src_elements]

        # Only process if this source contributes at least one element
        contributing = [c for c in src_canonical if c in elements]
        if not contributing:
            continue

        src_par_entries = parse_params(par_path)

        # Build index map: source 1-based index -> target 1-based index
        index_map = {}
        drop_indices = set()
        for i, canon in enumerate(src_canonical, start=1):
            if canon in elements:
                index_map[i] = elements.index(canon) + 1
            else:
                drop_indices.add(i)

        remapped = remap_params(src_par_entries, index_map, drop_indices)
        # Only update keys not already set (first source wins)
        for k, v in remapped.items():
            if k not in merged_params:
                merged_params[k] = v

    # 5. Generate missing cross-term pairs
    n = len(elements)
    all_pairs = {(i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1)}

    known_pairs = set()
    for key in merged_params:
        if key.startswith("Ec("):
            _, indices = _parse_key(key)
            if indices and len(indices) == 2:
                known_pairs.add(tuple(sorted(indices)))

    # Collect self-screening for mixing rules
    self_screen = {}
    for key, val in merged_params.items():
        if isinstance(val, (int, float)):
            name, indices = _parse_key(key)
            if indices and name in ("Cmin", "Cmax") and len(indices) == 3:
                if indices[0] == indices[1] == indices[2]:
                    self_screen[key] = val

    for i, j in sorted(all_pairs - known_pairs):
        pair_entries = generate_missing_pair(i, j, elements, merged_lib, self_screen)
        for key, val in pair_entries:
            merged_params[key] = val

    # 6. Add default global params if not already set
    defaults = {"rc": 6.0, "delr": 0.1, "ialloy": 2, "emb_lin_neg": 1, "bkgd_dyn": 1}
    for k, v in defaults.items():
        if k not in merged_params:
            merged_params[k] = v

    # 7. Write output
    os.makedirs(output_dir, exist_ok=True)
    prefix = "".join(elements)
    lib_out = os.path.join(output_dir, f"library_{prefix}.meam")
    par_out = os.path.join(output_dir, f"{prefix}.meam")

    write_library(merged_lib, elements, lib_out)
    write_params(_order_params(merged_params), par_out)

    print(f"Auto-merged potential written to {output_dir}")
    print(f"  Library: {lib_out}")
    print(f"  Params:  {par_out}")
    print(f"  Elements: {' '.join(elements)}")

    return lib_out, par_out


def main():
    parser = argparse.ArgumentParser(description="Generalized MEAM Merger")
    parser.add_argument("--config", required=True, help="Path to merge config JSON")
    parser.add_argument("--eam-dir", default=None, help="Source EAM directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    # Default paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    eam_dir = os.path.abspath(args.eam_dir or os.path.join(project_root, "EAM"))
    output_dir = os.path.abspath(args.output_dir or eam_dir)

    merge_from_config(args.config, eam_dir, output_dir)


if __name__ == "__main__":
    main()
