#!/bin/bash
# Build a self-contained arXiv source package from tex/ and figures/, compile it from scratch, and tar it.
# arXiv does not run BibTeX, so the generated main.bbl is included; all paths are local to the package.
set -euo pipefail
cd "$(dirname "$0")"
OUT=arxiv; rm -rf "$OUT" arxiv-source.tar.gz; mkdir -p "$OUT/figures"
cp tex/*.tex "$OUT"/
cp figures/wall.pdf figures/production.pdf figures/adaptive.pdf "$OUT/figures/"
cp references.bib "$OUT/references.bib"
sed -i 's#\.\./figures/#figures/#g' "$OUT"/*.tex
sed -i 's#\\bibliography{\.\./references}#\\bibliography{references}#' "$OUT/main.tex"
( cd "$OUT" && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
  && sed -i 's/agents?, 2026/agents? 2026/' main.bbl \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null )
grep -E "undefined|Overfull|^!" "$OUT/main.log" && { echo "build problems above"; exit 1; } || true
cp "$OUT/main.pdf" arxiv-preview.pdf
( cd "$OUT" && rm -f *.aux *.log *.out *.blg *.fls *.fdb_latexmk main.pdf references.bib \
  && tar -czf ../arxiv-source.tar.gz main.bbl *.tex figures )
# verify the tarball alone compiles (as arXiv will): no BibTeX, only the shipped .bbl
T=$(mktemp -d); tar -xzf arxiv-source.tar.gz -C "$T"
( cd "$T" && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null )
echo "undefined refs in clean build: $(grep -cE "undefined" "$T/main.log" || true)"
pdfinfo "$T/main.pdf" | grep Pages
ls -la arxiv-source.tar.gz
