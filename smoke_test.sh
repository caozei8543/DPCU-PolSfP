#!/usr/bin/env bash
# Automated self-contained pipeline verification; entirely SYNTHETIC.
# Do not quote any resulting numbers as published benchmark performance.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python -m pytest -q
dpcu-polsfp demo-data --out data/demo --size 32
dpcu-polsfp train --train data/demo/train.csv --val data/demo/val.csv \
  --out weights/toy_reference.pt --epochs 2 --width 8 --batch-size 1
dpcu-polsfp evaluate --manifest data/demo/test.csv --checkpoint weights/toy_reference.pt \
  --out results/toy_eval.json
python scripts/decision_analysis.py --manifest data/demo/test.csv \
  --checkpoint weights/toy_reference.pt --out results/toy_decisions.json
dpcu-polsfp sensitivity --manifest data/demo/test.csv \
  --checkpoint weights/toy_reference.pt --out results/toy_sensitivity.json
echo 'PASS: all codepaths above executed; SYNTHETIC ONLY, not paper reproduction.'
