# DFT-Driven N-Element MEAM Potential Pipeline

## Implementation Status

- [x] `src/NNIP/element_selector.py` — Web GUI for element selection
- [x] `src/NNIP/dft_reference.py` — DFT reference generator (ASE + QE)
- [x] `src/NNIP/dft_to_meam.py` — DFT-to-MEAM initializer
- [x] `src/NNIP/nn_optimizer.py` — Refactored multi-target NN optimizer
- [x] `src/NNIP/pipeline.py` — Pipeline orchestrator with CLI
- [x] `src/NNIP/run_pipeline.sh` — Shell wrapper with logging
- [x] `src/NNIP/README.md` — Full workflow documentation
- [x] Folder restructure: all code under `src/NNIP/` (removed `meam_opt/`)
- [x] DFT binary coverage: all 10 lattice-pair combinations covered (55 pairs for 11 elements)

## Verification Checklist

- [x] All files pass syntax check
- [x] All imports resolve correctly
- [x] Pipeline CLI: `--help`, `--elements`, `--skip-dft`, `--samples` work
- [x] Element selector scans pseudopotentials/ (18 elements found)
- [x] Training targets load from `src/ML/results.json` (5 entries)
- [x] All binary lattice combinations covered (fcc/bcc/hcp/diamond x fcc/bcc/hcp/diamond)
- [ ] GUI opens and selection works (requires display)
- [ ] DFT Al: a_lat ~ 4.04 A, E_coh ~ 3.39 eV (requires QE run)
- [ ] Initial MEAM files load in LAMMPS
- [ ] NN loss decreases during training
- [ ] Optimized MEAM matches results.json targets within 10%

## Pipeline Usage

```bash
# Full pipeline with GUI
./src/NNIP/run_pipeline.sh

# Skip GUI, specify elements directly
./src/NNIP/run_pipeline.sh Al Cu Zn Mg

# Skip DFT (use existing results), more samples
./src/NNIP/run_pipeline.sh --skip-dft --samples 50 Al Cu Zn Mg

# Python directly
python src/NNIP/pipeline.py --elements Al Cu --skip-optimize --skip-verify
```
