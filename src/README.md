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

### `element.py` — Dynamic Potential Parsing

Instead of hardcoding element properties, this module dynamically scans the `EAM/` directory for MEAM potential pairs (`library_*.meam` and `*.meam`).

- **`parse_meam_library(path)`**: Extracts element symbols, lattice types, masses, and lattice constants directly from the MEAM library file.
- **`scan_eam_dir(dir)`**: Identifies all available potentials (e.g., `AlMgZn`, `FeMnNiTiCuCrCoAl`) and their constituent elements.
- **`ELEMENTS`**: A merged lookup dictionary containing all elements discovered across all available potentials.

### `config.py` — Potential Auto-Detection & Shared Config

The single source of truth for simulation parameters. It handles loading configuration JSON files and deriving physical quantities.

- **`find_potential(symbols)`**: Automatically selects the first MEAM potential in `EAM/` that contains all elements requested in the composition.
- **`potential`**: Exports a dictionary containing the paths to the matching `library` and `params` files, along with the required element ordering.
- **Derived Quantities**: Calculates `a_mean` (weighted lattice constant) and `n_repeats` based on the selected alloy.

### `gui.py` — Web Configuration Tool

Provides a modern, browser-based interface to configure simulations without editing JSON manually.
- **Dynamic Elements**: Polls `element.py` to show all elements available in the `EAM/` directory.
- **Interactive Preview**: Features a 3D animated cube that reflects the selected alloy composition.
- **Validation**: Ensures percentages sum to 100% and generates valid `config.json` files.

### `lmp.py` — LAMMPS Simulation

Constructs and executes the simulation using the dynamic potential info:
- **`pair_coeff`**: Auto-generated using the specific library and parameter files detected by `config.py`.
- **Composition Mapping**: Handles the mapping between LAMMPS atom types and MEAM library indices for any supported potential.

### `viz.py` — OVITO Visualization

Renders high-quality animations of the simulation results.
- **Expanded Color Map**: Includes specialized colors for common elements (Fe, Cu, Al, Mg, Zn, etc.).
- **Deterministic Fallback**: Uses a hash-based generator to assign unique, consistent colors to any unknown elements found in new MEAM potentials.

## Running

### Using the GUI (Recommended)
```bash
python3 src/gui.py
```
This opens a web interface. Save your configuration, then run the simulation.

### Running a specific config
```bash
python3 src/lmp.py path/to/your_config.json
```
This runs the LAMMPS simulation followed by the OVITO rendering for the specified configuration.
