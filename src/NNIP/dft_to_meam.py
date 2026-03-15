#!/usr/bin/env python3
"""Initialize MEAM potential files from DFT reference data.

Takes DFT results (lattice constants, cohesive energies, formation energies)
and overlays them onto merged MEAM base files, producing initial library
and params files suitable for NN optimization.
"""

import json
import math
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.NNIP.meam_io import parse_library, parse_params, write_library, write_params
from src.NNIP.merge_potentials import merge_from_config, nn_distance


def _rose_alpha(E_coh, B_GPa, omega, mass_amu):
    """Estimate MEAM alpha parameter from Rose equation.

    alpha = sqrt(9 * B * Omega / E_coh)

    where B is in eV/A^3 and Omega is atomic volume in A^3.
    """
    B_eV_A3 = B_GPa * 0.00624  # GPa to eV/A^3
    if E_coh <= 0 or B_eV_A3 <= 0 or omega <= 0:
        return 4.5  # fallback
    return math.sqrt(9.0 * B_eV_A3 * omega / E_coh)


def _atomic_volume(lattice, a_lat):
    """Atomic volume for a given lattice type."""
    lat = lattice.lower()
    if lat == "fcc":
        return a_lat ** 3 / 4.0
    elif lat == "bcc":
        return a_lat ** 3 / 2.0
    elif lat in ("hcp", "hex"):
        return a_lat ** 3 * math.sqrt(2)  # approximate
    elif lat in ("dia", "diamond"):
        return a_lat ** 3 / 8.0
    return a_lat ** 3 / 4.0  # fallback


def initialize_meam_from_dft(dft_results, elements, eam_dir, merge_config_path=None, output_dir=None):
    """Update MEAM library/params with DFT-derived values.

    Args:
        dft_results: dict from dft_reference.py (or path to dft_results.json)
        elements: list of element symbols in desired order
        eam_dir: directory containing base MEAM files
        merge_config_path: path to merge config JSON (if base files need generating)
        output_dir: output directory for initialized files

    Returns:
        (lib_path, params_path) — paths to the written files
    """
    if isinstance(dft_results, str):
        with open(dft_results) as f:
            dft_results = json.load(f)

    if output_dir is None:
        output_dir = os.path.join(eam_dir, "dft_initialized")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate or load base merged potential
    prefix = "".join(elements)

    lib_file = os.path.join(eam_dir, f"library_{prefix}.meam")
    par_file = os.path.join(eam_dir, f"{prefix}.meam")

    if not os.path.exists(lib_file) and merge_config_path:
        print("Base merged files not found, running merge...")
        merge_from_config(merge_config_path, eam_dir, eam_dir)

    if not os.path.exists(lib_file):
        # Try to find any library file that covers the elements
        for fname in os.listdir(eam_dir):
            if fname.startswith("library_") and fname.endswith(".meam"):
                candidate = os.path.join(eam_dir, fname)
                lib_data = parse_library(candidate)
                if all(e in lib_data for e in elements):
                    lib_file = candidate
                    par_file = os.path.join(eam_dir, fname[len("library_"):])
                    break

    if not os.path.exists(lib_file):
        raise FileNotFoundError(
            f"No MEAM library file found covering {elements} in {eam_dir}. "
            "Run merge_potentials.py first."
        )

    lib_data = parse_library(lib_file)
    param_entries = parse_params(par_file)

    # Step 2: Update library parameters from DFT results
    dft_elements = dft_results.get("elements", {})
    for sym in elements:
        if sym not in dft_elements:
            continue
        if sym not in lib_data:
            continue

        dft = dft_elements[sym]
        lib_entry = lib_data[sym]
        params = lib_entry["params"]

        # Update esub (cohesive energy)
        if "E_coh" in dft:
            params["esub"] = dft["E_coh"]

        # Update alat (lattice constant)
        if "a_lat" in dft:
            params["alat"] = dft["a_lat"]

        # Update alpha from Rose equation
        if "E_coh" in dft and "B_GPa" in dft and "a_lat" in dft:
            lattice = dft.get("lattice", lib_entry["header"][0])
            omega = _atomic_volume(lattice, dft["a_lat"])
            alpha = _rose_alpha(dft["E_coh"], dft["B_GPa"], omega, lib_entry["header"][3])
            params["alpha"] = round(alpha, 6)

    # Step 3: Update cross-term parameters from binary formation energies
    binary_pairs = dft_results.get("binary_pairs", {})
    param_dict = {k: v for k, v in param_entries}

    for pair_key, pair_data in binary_pairs.items():
        parts = pair_key.split("-")
        if len(parts) != 2:
            continue
        sym_i, sym_j = parts

        if sym_i not in elements or sym_j not in elements:
            continue

        idx_i = elements.index(sym_i) + 1
        idx_j = elements.index(sym_j) + 1
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i

        # Update Ec(i,j) from formation energy
        if "E_form" in pair_data:
            # Ec(i,j) = average of pure Ec values + formation energy contribution
            esub_i = lib_data[sym_i]["params"]["esub"]
            esub_j = lib_data[sym_j]["params"]["esub"]
            ec_ij = (esub_i + esub_j) / 2.0 + pair_data["E_form"]
            param_dict[f"Ec({idx_i},{idx_j})"] = round(ec_ij, 6)

        # Update re(i,j) from DFT lattice constants
        if sym_i in dft_elements and sym_j in dft_elements:
            a_i = dft_elements[sym_i].get("a_lat", lib_data[sym_i]["params"]["alat"])
            a_j = dft_elements[sym_j].get("a_lat", lib_data[sym_j]["params"]["alat"])
            lat_i = dft_elements[sym_i].get("lattice", lib_data[sym_i]["header"][0])
            lat_j = dft_elements[sym_j].get("lattice", lib_data[sym_j]["header"][0])
            nn_i = nn_distance(lat_i, a_i)
            nn_j = nn_distance(lat_j, a_j)
            re_ij = (nn_i + nn_j) / 2.0
            param_dict[f"re({idx_i},{idx_j})"] = round(re_ij, 6)

        # Update alpha(i,j) from average of pure alphas
        alpha_i = lib_data[sym_i]["params"]["alpha"]
        alpha_j = lib_data[sym_j]["params"]["alpha"]
        param_dict[f"alpha({idx_i},{idx_j})"] = round((alpha_i + alpha_j) / 2.0, 4)

    # Rebuild param entries list preserving order, adding new keys at end
    updated_entries = []
    seen_keys = set()
    for key, val in param_entries:
        if key in param_dict:
            updated_entries.append((key, param_dict[key]))
        else:
            updated_entries.append((key, val))
        seen_keys.add(key)

    # Add any new keys not in original
    for key, val in param_dict.items():
        if key not in seen_keys:
            updated_entries.append((key, val))

    # Write output files
    lib_out = os.path.join(output_dir, f"library_{prefix}.meam")
    par_out = os.path.join(output_dir, f"{prefix}.meam")

    write_library(lib_data, elements, lib_out)
    write_params(updated_entries, par_out)

    print(f"DFT-initialized MEAM files written to {output_dir}")
    print(f"  Library: {lib_out}")
    print(f"  Params:  {par_out}")

    return lib_out, par_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFT-to-MEAM Initializer")
    parser.add_argument("--dft-results", required=True, help="Path to dft_results.json")
    parser.add_argument("--elements", nargs="+", required=True, help="Element symbols in order")
    parser.add_argument("--eam-dir", default=None, help="EAM directory")
    parser.add_argument("--merge-config", default=None, help="Path to merge config JSON")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    eam_dir = args.eam_dir or os.path.join(project_root, "EAM")
    initialize_meam_from_dft(
        args.dft_results, args.elements, eam_dir,
        merge_config_path=args.merge_config,
        output_dir=args.output_dir,
    )
