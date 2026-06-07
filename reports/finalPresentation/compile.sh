#!/bin/bash
# Compile the final-presentation beamer deck.
# Usage:  ./compile.sh                     -- full pdflatex/bibtex/pdflatex x2
#         REGEN=1 ./compile.sh             -- also re-run generate_nnip_figures.py
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tmp

if [[ "${REGEN:-0}" == "1" ]]; then
  # Each generator depends on a JSON artifact produced elsewhere in the
  # pipeline; a missing input should warn and skip, not abort the whole
  # compile (so the deck still builds from the committed PNGs).
  echo "== regenerating NNIP figures =="
  ../../phys/bin/python3 generate_nnip_figures.py \
    || echo "   (skipped: NNIP diagnostics not present)"
  echo "== regenerating literature-comparison tables =="
  ../../phys/bin/python3 generate_literature_figure.py \
    || echo "   (skipped: literature_comparison.json not present)"
fi

DOC=0900-2587772-meam-nn-final
# bibtex needs the .bib next to its working dir, so copy it into tmp
cp references.bib tmp/

pdflatex -interaction=nonstopmode -output-directory=tmp $DOC.tex >/dev/null
( cd tmp && bibtex $DOC >/dev/null ) || true
pdflatex -interaction=nonstopmode -output-directory=tmp $DOC.tex >/dev/null
pdflatex -interaction=nonstopmode -output-directory=tmp $DOC.tex >/dev/null

cp tmp/$DOC.pdf .
echo "Done: $DOC.pdf"
