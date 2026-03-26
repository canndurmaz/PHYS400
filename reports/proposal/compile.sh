#!/bin/bash
pdflatex -interaction=nonstopmode Proposal.tex
bibtex Proposal
pdflatex -interaction=nonstopmode Proposal.tex
pdflatex -interaction=nonstopmode Proposal.tex
