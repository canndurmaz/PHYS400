#!/bin/bash
cd "$(dirname "$0")"
mkdir -p tmp

pdflatex -interaction=nonstopmode -output-directory=tmp 0900-2587772-meam-nn-mock.tex
pdflatex -interaction=nonstopmode -output-directory=tmp 0900-2587772-meam-nn-mock.tex

cp tmp/0900-2587772-meam-nn-mock.pdf .
echo "Done: 0900-2587772-meam-nn-mock.pdf"
