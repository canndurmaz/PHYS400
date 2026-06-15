#!/bin/bash
# Build the PHYS400 final report.
#
# The auto-generated tables (sections/_auto_*.tex) are committed alongside
# the prose, so a plain run just compiles. Set REGEN=1 to refresh them from
# the canonical project data first: the stats helper rebuilds the MD/DFT/MEAM
# and composition-surrogate tables, and the two presentation generators
# rebuild the literature-comparison tables. A missing data artifact warns and
# is skipped, so the report still builds from the committed tables.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ "${REGEN:-0}" == "1" ]]; then
  echo "== refreshing stats tables (MD / DFT / MEAM / composition NN) =="
  "$HERE/../../src/stats/run_all.sh" || echo "   (skipped: stats inputs not present)"
  echo "== refreshing literature-comparison tables =="
  "$HERE/../../phys/bin/python3" "$HERE/../finalPresentation/generate_literature_figure.py" \
    || echo "   (skipped: literature_comparison.json not present)"
  # The literature generator writes into the presentation's sections/; mirror
  # the fresh copies into this report.
  cp "$HERE/../finalPresentation/sections/_auto_literature_al.tex" "$HERE/sections/" 2>/dev/null || true
  cp "$HERE/../finalPresentation/sections/_auto_literature_al_detail.tex" "$HERE/sections/" 2>/dev/null || true
  cp "$HERE/../finalPresentation/sections/_auto_literature_family.tex" "$HERE/sections/" 2>/dev/null || true
  for t in md_stats dft_elements meam_init ml_metrics; do
    cp "$HERE/../interim/sections/_auto_${t}.tex" "$HERE/sections/" 2>/dev/null || true
  done
fi

cd "$HERE"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
