#!/bin/bash
cd "$(dirname "$0")"
mkdir -p tmp

# TEXINPUTS: let pdflatex find .bib and images via parent dirs
export TEXINPUTS=".:../proposal/:../../src/NNIP/:"
export BIBINPUTS=".:../proposal/:"

pdflatex -interaction=nonstopmode -output-directory=tmp slideshow.tex
bibtex tmp/slideshow
pdflatex -interaction=nonstopmode -output-directory=tmp slideshow.tex
pdflatex -interaction=nonstopmode -output-directory=tmp slideshow.tex
cp tmp/slideshow.pdf .
echo "Done: slideshow.pdf"
