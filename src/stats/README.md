# `src/stats/` -- report-asset generators

This folder contains the Python scripts that regenerate every
auto-derivable figure and LaTeX table consumed by the interim report
under `reports/interim/`. The scripts read only the canonical project
data files (`src/ML/results.json`, `src/NNIP/dft_results.json`,
`src/NNIP/pipeline_summary.json`) and write into the report tree.

## What each script produces

| Script                  | Figures (`reports/interim/figures/`)                                   | Tables (`reports/interim/sections/`) |
| ----------------------- | ----------------------------------------------------------------------- | ------------------------------------- |
| `dataset_stats.py`      | `element_frequency.png`, `dataset_distributions.png`, `E_nu_scatter.png` | `_auto_md_stats.tex`                  |
| `dft_stats.py`          | `formation_energy_heatmap.png`                                          | `_auto_dft_elements.tex`              |
| `meam_init_stats.py`    | --                                                                       | `_auto_meam_init.tex`                 |
| `stage5_stats.py`       | `stage5_verification_bars.png`                                          | `_auto_pipeline_timings.tex`, `_auto_nn_validation.tex` |

All LaTeX tables are produced via the `tabulate` library
(`tablefmt="latex_raw"`) and wrapped in a complete `\begin{table}` /
`\caption` / `\label` block by the helper in `_common.py`, so the report
can pick them up with `\input{sections/_auto_<name>}`.

## How to run

```
./run_all.sh
```

(uses the project venv at `../../phys/bin/python3`). Or run an
individual generator, e.g. `python dataset_stats.py`.

The ML sanity-check branch (`src/ML/nn_alloy.py`) has its own
report-export step embedded in the script -- it copies its plots into
`reports/interim/figures/` and writes
`sections/_auto_ml_metrics.tex` directly, so there is no separate
generator here.
