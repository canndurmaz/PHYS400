# PHYS400 — Computational Materials Science

Molecular dynamics (MD) simulations of metallic alloys using **LAMMPS**, coupled with **TensorFlow Machine Learning (ML)** for elastic property prediction.

---

## 1. Installation & Environment Setup

This project requires a specific version of LAMMPS and a Python 3.12 environment.

### Python Environment
1. Create and activate the virtual environment:
   ```bash
   python3 -m venv phys
   source phys/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install lammps ovito numpy tensorflow matplotlib
   ```

### Custom LAMMPS Build
The default LAMMPS build often limits MEAM potentials to 5 elements. To support the 8-element potentials used in this project, you must rebuild from source with `maxelt=8`.

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

# 4. Export the library path
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 2. Project Structure

```
PHYS400/
├── EAM/              # Interatomic potential library (MEAM files)
├── src/
│   ├── configs/      # Input alloy compositions (.json)
│   ├── MD/           # Molecular Dynamics module (LAMMPS)
│   └── ML/           # Machine Learning module (TensorFlow)
├── phys/             # Python Virtual Environment
└── README.md         # This file
```

---

## 3. Workflows

### A. MD Simulation & Data Generation
This workflow runs a physical simulation to calculate the Young's Modulus and Poisson's Ratio of an alloy.

1.  **Configure Alloy**: Use the web-based GUI to create a composition.
    ```bash
    python3 src/MD/gui.py
    ```
2.  **Run Simulation**: Execute the MD engine. This automatically calculates elastic properties and appends them to `src/ML/results.json`.
    ```bash
    # Run the default config
    ./src/MD/run.sh
    # OR run a specific config
    ./phys/bin/python3 src/MD/lmp.py src/configs/AL7075_simple.json
    ```

### B. Machine Learning Training
This workflow trains a Neural Network on the data generated in the MD step.

1.  **Batch Training**: Reads all results from `results.json`, trains the model, and saves it to `alloy_model.keras`.
    ```bash
    ./src/ML/run_nn.sh
    ```
    *This script also generates predictions for randomized alloys in `src/ML/predict.json`.*

### C. Standalone Prediction
Use the trained model to instantly estimate properties for any new composition without running a full MD simulation.

1.  **Predict from JSON**:
    ```bash
    ./phys/bin/python3 src/ML/predict_from_model.py path/to/composition.json
    ```

---

## 4. Element Feature Mapping
The following index order is strictly maintained for both MD integration and ML feature vectors:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Symbol** | Al | Co | Cr | Cu | Fe | Mg | Mn | Ni | Ti | Zn |

---

## 5. Modules Documentation
- [**MD Module (LAMMPS)**](src/MD/README.md): Details on potential parsing, relaxation protocols, and the static deformation method.
- [**ML Module (TensorFlow)**](src/ML/README.md): Details on model architecture, training parameters, and data structures.
