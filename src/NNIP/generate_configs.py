#!/usr/bin/env python3
"""Generate diverse atomic configurations for DFT training data.

Reads settings from config.json (elements, strains, alloy compositions, etc.).
Run with --config to specify a different config file.

Output: Extended XYZ files in data/training/configs/
"""

import argparse
import itertools
import json
import os
import random

import numpy as np
from ase.build import bulk
from ase.io import write

PROJECT_DIR = "/home/kenobi/Workspaces/PHYS400"
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG_DIR = os.path.join(PROJECT_DIR, "data", "training", "configs")


def load_config(path):
    with open(path) as f:
        return json.load(f)


def make_supercell_atoms(atoms, n):
    """Create an n×n×n supercell."""
    return atoms.repeat((n, n, n))


def apply_volumetric_strain(atoms, strain):
    """Apply isotropic volumetric strain. strain=0.05 means +5%."""
    strained = atoms.copy()
    factor = (1.0 + strain) ** (1.0 / 3.0)
    strained.set_cell(strained.get_cell() * factor, scale_atoms=True)
    return strained


def apply_shear_strain(atoms, gamma=0.05):
    """Apply shear strain in xy plane."""
    strained = atoms.copy()
    cell = strained.get_cell().copy()
    cell[0, 1] += gamma * cell[1, 1]
    strained.set_cell(cell, scale_atoms=True)
    return strained


def create_random_alloy(base_element, elements_dict, elements, fractions, supercell_size=2):
    """Create a random substitutional alloy supercell."""
    props = elements_dict[base_element]
    atoms = bulk(base_element, props["structure"], a=props["a"])
    atoms = make_supercell_atoms(atoms, supercell_size)

    n_atoms = len(atoms)
    symbols = list(atoms.get_chemical_symbols())
    cumulative = np.cumsum(fractions)
    for i in range(n_atoms):
        r = random.random()
        for j, c in enumerate(cumulative):
            if r < c:
                symbols[i] = elements[j]
                break
    atoms.set_chemical_symbols(symbols)
    return atoms


def create_vacancy(atoms, n_vacancies=1):
    """Remove n_vacancies atoms from the structure."""
    indices = list(range(len(atoms)))
    remove = sorted(random.sample(indices, min(n_vacancies, len(atoms) - 1)), reverse=True)
    vacancy = atoms.copy()
    for idx in remove:
        del vacancy[idx]
    return vacancy


def rattled_config(atoms, stdev=0.1):
    """Add Gaussian noise to atomic positions."""
    rattled = atoms.copy()
    rattled.positions += np.random.normal(0, stdev, rattled.positions.shape)
    return rattled


# ── Generators ──────────────────────────────────────────────


def generate_pure_element_configs(cfg):
    """Generate pure element configurations at various strains."""
    pure_cfg = cfg.get("pure", {})
    if not pure_cfg.get("enabled", True):
        return []

    elements = cfg["elements"]
    sc_size = pure_cfg.get("supercell_size", 2)
    vol_strains = pure_cfg.get("volumetric_strains", [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05])
    shear_strains = pure_cfg.get("shear_strains", [0.02, 0.05])

    configs = []
    for elem, props in elements.items():
        base = bulk(elem, props["structure"], a=props["a"])
        sc = make_supercell_atoms(base, sc_size)

        for strain in vol_strains:
            if strain == 0.0:
                config = sc.copy()
                config.info["config_type"] = f"pure_{elem}_eq"
            else:
                config = apply_volumetric_strain(sc, strain)
                config.info["config_type"] = f"pure_{elem}_vol_{strain:+.2f}"
            configs.append(config)

        for gamma in shear_strains:
            config = apply_shear_strain(sc, gamma)
            config.info["config_type"] = f"pure_{elem}_shear_{gamma:.2f}"
            configs.append(config)

    print(f"  Pure element configs: {len(configs)}")
    return configs


def generate_alloy_configs(cfg):
    """Generate random binary, ternary, and custom alloy supercells."""
    elements = cfg["elements"]
    elem_list = list(elements.keys())
    configs = []

    # Binary alloys
    bin_cfg = cfg.get("binary", {})
    if bin_cfg.get("enabled", True):
        fracs = bin_cfg.get("fractions", [0.25, 0.50, 0.75])
        sc_sizes = bin_cfg.get("supercell_sizes", [2, 3])

        if bin_cfg.get("all_pairs", True):
            pairs = list(itertools.combinations(elem_list, 2))
        else:
            pairs = []

        for e1, e2 in pairs:
            for frac in fracs:
                for sc_size in sc_sizes:
                    config = create_random_alloy(e1, elements, [e1, e2], [1 - frac, frac], sc_size)
                    config.info["config_type"] = f"binary_{e1}{e2}_{frac:.2f}_sc{sc_size}"
                    configs.append(config)

    # Ternary alloys
    ter_cfg = cfg.get("ternary", {})
    if ter_cfg.get("enabled", True):
        rng = random.Random(ter_cfg.get("seed", 42))
        n_random = ter_cfg.get("n_random", 30)
        sc_sizes = ter_cfg.get("supercell_sizes", [2, 3])

        for _ in range(n_random):
            elems = rng.sample(elem_list, 3)
            raw = [rng.random() for _ in range(3)]
            total = sum(raw)
            fracs = [r / total for r in raw]
            sc_size = rng.choice(sc_sizes)
            config = create_random_alloy(elems[0], elements, elems, fracs, sc_size)
            frac_str = "_".join(f"{f:.2f}" for f in fracs)
            config.info["config_type"] = f"ternary_{''.join(elems)}_{frac_str}_sc{sc_size}"
            configs.append(config)

    # Custom Al-rich alloys
    al_cfg = cfg.get("al_alloys", {})
    if al_cfg.get("enabled", True):
        sc_sizes = al_cfg.get("supercell_sizes", [2, 3])
        for comp in al_cfg.get("compositions", []):
            elems = comp["elements"]
            fracs = comp["fractions"]
            for sc_size in sc_sizes:
                config = create_random_alloy(elems[0], elements, elems, fracs, sc_size)
                comp_str = "".join(f"{e}{f:.2f}" for e, f in zip(elems, fracs))
                config.info["config_type"] = f"alalloy_{comp_str}_sc{sc_size}"
                configs.append(config)

    print(f"  Alloy configs: {len(configs)}")
    return configs


def generate_vacancy_configs(cfg):
    """Generate vacancy configurations from supercells."""
    vac_cfg = cfg.get("vacancy", {})
    if not vac_cfg.get("enabled", True):
        return []

    elements = cfg["elements"]
    max_vac = vac_cfg.get("max_vacancies", 2)
    configs = []

    # Pure element vacancies
    for elem, props in elements.items():
        base = bulk(elem, props["structure"], a=props["a"])
        sc = make_supercell_atoms(base, 2)
        for nv in range(1, max_vac + 1):
            config = create_vacancy(sc, nv)
            config.info["config_type"] = f"vacancy_{elem}_{nv}v"
            configs.append(config)

    # Alloy vacancies
    for comp in vac_cfg.get("alloy_compositions", []):
        elems = comp["elements"]
        fracs = comp["fractions"]
        sc = create_random_alloy(elems[0], elements, elems, fracs, supercell_size=2)
        for nv in range(1, max_vac + 1):
            config = create_vacancy(sc, nv)
            comp_str = "".join(elems)
            config.info["config_type"] = f"vacancy_{comp_str}_{nv}v"
            configs.append(config)

    print(f"  Vacancy configs: {len(configs)}")
    return configs


def generate_rattled_configs(cfg):
    """Generate thermally displaced (rattled) configurations."""
    rat_cfg = cfg.get("rattled", {})
    if not rat_cfg.get("enabled", True):
        return []

    elements = cfg["elements"]
    temps = rat_cfg.get("temperatures", {"300K": 0.05, "600K": 0.10, "1000K": 0.15})
    n_snap = rat_cfg.get("snapshots_per_temp", 3)
    configs = []

    # Pure element rattled
    for elem, props in elements.items():
        base = bulk(elem, props["structure"], a=props["a"])
        sc = make_supercell_atoms(base, 2)
        for temp_label, stdev in temps.items():
            for i in range(n_snap):
                config = rattled_config(sc, stdev)
                config.info["config_type"] = f"rattled_{elem}_{temp_label}_{i}"
                configs.append(config)

    # Rattled alloys
    for comp in rat_cfg.get("alloy_compositions", []):
        elems = comp["elements"]
        fracs = comp["fractions"]
        sc = create_random_alloy(elems[0], elements, elems, fracs, supercell_size=2)
        for temp_label, stdev in temps.items():
            for i in range(n_snap):
                config = rattled_config(sc, stdev)
                comp_str = "".join(elems)
                config.info["config_type"] = f"rattled_{comp_str}_{temp_label}_{i}"
                configs.append(config)

    print(f"  Rattled configs: {len(configs)}")
    return configs


# ── Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate DFT training configurations")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Path to config JSON (default: {DEFAULT_CONFIG})")
    parser.add_argument("--output", default=CONFIG_DIR,
                        help=f"Output directory (default: {CONFIG_DIR})")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Set global seed
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Elements: {', '.join(cfg['elements'].keys())}")
    print()

    all_configs = []
    all_configs.extend(generate_pure_element_configs(cfg))
    all_configs.extend(generate_alloy_configs(cfg))
    all_configs.extend(generate_vacancy_configs(cfg))
    all_configs.extend(generate_rattled_configs(cfg))

    print(f"\nTotal configurations: {len(all_configs)}")

    # Write configs
    manifest = []
    for i, config in enumerate(all_configs):
        config_type = config.info.get("config_type", f"config_{i:04d}")
        fname = f"{i:04d}_{config_type}.xyz"
        filepath = os.path.join(output_dir, fname)
        write(filepath, config, format="extxyz")
        manifest.append({
            "index": i,
            "file": fname,
            "config_type": config_type,
            "n_atoms": len(config),
            "elements": sorted(set(config.get_chemical_symbols())),
        })

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Written {len(all_configs)} configurations to {output_dir}")
    print(f"Manifest: {manifest_path}")

    # Summary
    from collections import Counter
    type_prefix = Counter()
    for m in manifest:
        prefix = m["config_type"].split("_")[0]
        type_prefix[prefix] += 1
    print("\nBreakdown by type:")
    for t, count in sorted(type_prefix.items()):
        print(f"  {t}: {count}")

    atom_counts = [m["n_atoms"] for m in manifest]
    print(f"\nAtom counts: min={min(atom_counts)}, max={max(atom_counts)}, "
          f"median={sorted(atom_counts)[len(atom_counts)//2]}")


if __name__ == "__main__":
    main()
