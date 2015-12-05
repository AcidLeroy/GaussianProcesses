#!/bin/bash

${MATLABROOT?"Please set your MATLABROOT"}/bin/matlab -nojvm -nosplash < hw3.m
pdflatex --output-directory tex_output/ hw3.tex
bibtex tex_output/hw3
pdflatex --output-directory tex_output/ hw3.tex
pdflatex --output-directory tex_output/ hw3.tex

open -a preview tex_output/hw3.pdf
