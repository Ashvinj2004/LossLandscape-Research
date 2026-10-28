#!/usr/bin/env bash
# Build the ICML-format paper: regenerate the appendix from ../main.tex, then pdflatex + bibtex.
# usage: bash build.sh   (from paper/icml/; writes main_icml.pdf here)
set -e
cd "$(dirname "$0")"
python3 build_appendix.py
B=$(mktemp -d)
cp ../refs.bib "$B/"
pdflatex -interaction=nonstopmode -output-directory="$B" main_icml.tex >/dev/null || true
sed -i 's#{../refs}#{refs}#' "$B/main_icml.aux"
(cd "$B" && bibtex main_icml >/dev/null)
pdflatex -interaction=nonstopmode -output-directory="$B" main_icml.tex >/dev/null || true
pdflatex -interaction=nonstopmode -output-directory="$B" main_icml.tex >/dev/null || true
grep -E "^!|undefined|multiply defined" "$B/main_icml.log" | sort -u || true
cp "$B/main_icml.pdf" .
for p in $(seq 1 15); do
  if pdftotext -f "$p" -l "$p" main_icml.pdf - 2>/dev/null | grep -q "^References"; then
    echo "main text + start of references: pages 1-$p (limit: 8 pages before references)"; break
  fi
done
echo "total pages: $(pdfinfo main_icml.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
