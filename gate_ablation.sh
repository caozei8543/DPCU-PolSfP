#!/usr/bin/env bash
# Train one reference architecture for each gate. Requires real author data.
# Usage: bash scripts/gate_ablation.sh data/train.csv data/val.csv
set -euo pipefail
train="${1:?training manifest required}"; val="${2:?validation manifest required}"
for gate in local fixed dolp softmax_confidence learned entropy context; do
  dpcu-polsfp train --train "$train" --val "$val" --gate "$gate" --seed 42 --out "weights/${gate}_seed42.pt"
done
