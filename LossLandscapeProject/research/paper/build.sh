#!/usr/bin/env bash
# Builds main.pdf (named preprint, for arXiv) and main_anon.pdf (anonymous TMLR review version).
set -e
cd "$(dirname "$0")"
build() {  # $1 = output name, $2 = extra definitions
  pdflatex -interaction=nonstopmode -jobname="$1" "$2\input{main.tex}" >/dev/null || true
  bibtex "$1" >/dev/null
  pdflatex -interaction=nonstopmode -jobname="$1" "$2\input{main.tex}" >/dev/null || true
  pdflatex -interaction=nonstopmode -jobname="$1" "$2\input{main.tex}" >/dev/null || true
  grep -E "^!|undefined|multiply defined" "$1.log" | sort -u || true
  echo "$1.pdf: $(pdfinfo "$1.pdf" | grep Pages)"
  rm -f "$1".aux "$1".bbl "$1".blg "$1".log "$1".out
}
build main ""
build main_anon "\def\ANON{1}"
