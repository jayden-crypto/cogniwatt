#!/bin/zsh
# Package the project for Adit. Owner: R1.
# Ships source + tests + docs + the ONE reference policy + result CSVs/figures.
# Excludes the 15 experiment checkpoints and training logs - Adit builds his
# own measurements; the full experiment archive stays with Jayden.
cd "/Users/jayden/Desktop/rl edge ai"
OUT="$HOME/Desktop/cogniwatt_handoff.zip"
rm -f "$OUT"
zip -r -q "$OUT" \
  cogniwatt scripts tests docs data README.md \
  results/eval results/figures \
  results/runs/ff_2m \
  -x '*__pycache__*' '*.pyc' '.DS_Store' '*/.DS_Store' \
     'results/figures/_embedded.json' 'data/*.parquet'
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "top-level:"; unzip -l "$OUT" | awk '{print $4}' | grep -v '^$' | cut -d/ -f1 | sort -u
