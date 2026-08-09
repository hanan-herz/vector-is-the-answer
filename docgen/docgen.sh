#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

INPUTS=("$@")
if [ "${#INPUTS[@]}" -eq 0 ]; then
  echo "usage: $0 <file.md> [file.md ...]" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for INPUT in "${INPUTS[@]}"; do
  OUT="${INPUT%.*}.pdf"
  pandoc "$INPUT" -f gfm -t html5 -o "$TMP/latent.html"
  { printf '<!doctype html><meta charset="utf-8">\n'; cat "$TMP/latent.html"; } > "$TMP/full.html"
  weasyprint -s docgen/latent.css "$TMP/full.html" "$OUT"
  echo "built $OUT"
done