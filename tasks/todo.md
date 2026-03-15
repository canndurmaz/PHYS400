# NNIP: Neural Network Interatomic Potential for 9-Element Al Alloys

## Phase 1: Download Pseudopotentials
- [x] Download PBE PAW pseudopotentials for: Zn, Mg, Mn, Cu, Si, Cr, Fe, Ti
- [x] Store in `/home/kenobi/Workspaces/PHYS400/pseudopotentials/`
- [x] Validate each with a single-element SCF test (bulk crystal) — all 9 pass
- **File**: `src/NNIP/download_pseudos.py`

## Phase 2: Generate DFT Training Data
- [x] Write config generator: pure bulk, strained, alloy supercells, vacancies, rattled
- [ ] Run QE SCF on each configuration (energy, forces, stress) — pipeline validated on 3 configs
- [x] Generated 470 initial configurations (81 pure, 266 alloy, 24 vacancy, 99 rattled)
- [ ] Store in DeePMD-kit raw format (convert_to_deepmd.py ready)
- **Files**: `src/NNIP/generate_configs.py`, `src/NNIP/run_dft.py`, `src/NNIP/convert_to_deepmd.py`
- **Output**: `data/training/`

## Phase 3: Install DeePMD-kit
- [x] Install `deepmd-kit` 3.1.2 in phys venv
- [x] Verify `dp --version`
- [ ] Build/install LAMMPS DeePMD plugin (system LAMMPS lacks it)
- [ ] Verify `pair_style deepmd` in LAMMPS — using ASE calculator fallback
- **File**: `src/NNIP/install_deepmd.sh`

## Phase 4: Train the Neural Network Potential
- [ ] Prepare data in DeePMD format (type.raw, set.000/)
- [ ] Write training input JSON (se_e2_a descriptor, 3x240 fitting net)
- [ ] Train: `dp train input.json`
- [ ] Freeze: `dp freeze -o model.pb`
- **Files**: `src/NNIP/input.json`, `src/NNIP/train.sh`
- **Output**: `models/model.pb`

## Phase 5: Validate the Model
- [ ] `dp test` on validation set
- [ ] Plot energy/force parity (predicted vs DFT)
- [ ] MAE targets: energy < 5 meV/atom, forces < 100 meV/Å
- [ ] Test on held-out compositions
- **File**: `src/NNIP/validate.py`

## Phase 6: Deploy in LAMMPS
- [ ] LAMMPS input with `pair_style deepmd model.pb`
- [ ] Test MD on Al alloy supercell
- [ ] Compare elastic properties with MEAM reference
- [ ] ASE calculator wrapper
- **Files**: `src/NNIP/run_md.py`, `src/NNIP/elastic.py`, `src/NNIP/calculator.py`

## Phase 7: Documentation
- [ ] `src/NNIP/README.md`
- [ ] Update project `README.md`
