#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

lualatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
lualatex -interaction=nonstopmode -halt-on-error paper.tex
lualatex -interaction=nonstopmode -halt-on-error paper.tex
