# LAMMPS MEAM Simulation — Configurable HEA Alloys

## Prerequisites

- Python 3.12+
- OVITO (installed in venv)
- LAMMPS with MEAM package (`maxelt >= 8`)

## Rebuilding LAMMPS with maxelt=8

The default Ubuntu LAMMPS package has `maxelt=5`, which is too small for the 8-element FeMnNiTiCuCrCoAl MEAM potential. Rebuild from source:

```bash
# 1. Get the source
git clone -b patch_7Feb2024 --depth 1 https://github.com/lammps/lammps.git
cd lammps/src

# 2. Increase maxelt from 5 to 8
sed -i 's/maxelt = 5/maxelt = 8/' MEAM/meam.h

# 3. Build as a shared library
cd ../
mkdir build && cd build
cmake ../cmake -D BUILD_SHARED_LIBS=yes \
               -D PKG_MEAM=yes \
               -D CMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install

# 4. Make the new library available
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Architecture

### `element.py` — Element Class & Predefined Instances

Defines a reusable `Element` class with properties parsed from `library.meam`:
- `symbol`, `lattice_type`, `coord_number`, `atomic_number`, `mass`, `lattice_constant`, `meam_index`

Provides 8 predefined instances for the FeMnNiTiCuCrCoAl MEAM potential:

| Element | meam_index | Lattice | Mass (amu) | a (A) |
|---------|-----------|---------|------------|-------|
| Fe | 1 | bcc | 55.845 | 2.8285 |
| Mn | 2 | hcp | 54.940 | 2.4708 |
| Ni | 3 | fcc | 58.6934 | 3.5214 |
| Ti | 4 | hcp | 47.880 | 2.9200 |
| Cu | 5 | fcc | 63.546 | 3.6200 |
| Cr | 6 | bcc | 51.960 | 2.8810 |
| Co | 7 | hcp | 58.933 | 2.5000 |
| Al | 8 | fcc | 26.9815 | 4.0500 |

Exports `ELEMENTS` (dict lookup by symbol) and `MEAM_ELEMENT_ORDER` (list in meam_index order).

### `config.py` — Shared Configuration

Single source of truth for all simulation parameters. Both `lmp.py` and `viz.py` import from here, so changing the composition in one place updates everything.

```python
CONFIG = {
    "composition": {"Cu": 0.5, "Al": 0.5},  # element fractions (must sum to 1.0)
    "box_size_m": 5e-9,                       # box side length in meters
    "temperature": 300.0,                      # K
    "total_steps": 1000,                       # total simulation steps
    "thermo_interval": 10,                     # thermo output interval
    "dump_interval": 50,                       # trajectory dump interval
}
```

Also computes derived quantities used by both scripts:
- `selected` — list of `Element` objects sorted by `meam_index`
- `a_mean` — weighted average lattice constant
- `n_repeats` — number of lattice repeats per box side

To change the alloy, edit `CONFIG["composition"]` (e.g., `{"Fe": 0.7, "Al": 0.3}`). Everything else updates automatically.

### `lmp.py` — LAMMPS Simulation

Imports config from `config.py` and auto-generates all LAMMPS commands:
- **`pair_coeff`** — lists all 8 MEAM elements for correct index mapping, active elements at the end
- **`mass` commands** — one per atom type
- **`set type/fraction`** — assigns random compositions sequentially

### `viz.py` — OVITO Visualization

Imports config from `config.py` and renders a trajectory animation for any composition:
- **Per-element colors** — assigned via an OVITO modifier using a predefined color map (Fe=brown, Cu=copper, Al=light blue, etc.)
- **Smaller spheres** — radius 0.8 for clearer visualization
- **Dynamic legend** — auto-generated labels showing each element symbol and its fraction
- **Auto-named output** — filename derived from composition (e.g., `Cu50Al50_vibration.mp4`)

## Running

```bash
cd src && bash run.sh
```

This runs `lmp.py` (LAMMPS simulation) followed by `viz.py` (OVITO rendering).
