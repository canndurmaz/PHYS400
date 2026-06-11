#!/bin/bash
# Compile the A1 poster. Figures are committed PNGs copied from
# reports/finalPresentation/figures; regenerate them there if needed.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tmp

pdflatex -interaction=nonstopmode -output-directory=tmp poster.tex >/dev/null
pdflatex -interaction=nonstopmode -output-directory=tmp poster.tex >/dev/null

cp tmp/poster.pdf .
echo "Done: poster.pdf"
