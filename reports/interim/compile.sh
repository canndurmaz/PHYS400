#!/bin/bash
# Refresh auto-generated figures and LaTeX tables from the canonical
# project data, then build the report.
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/../../src/stats/run_all.sh"

cd "$HERE"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
