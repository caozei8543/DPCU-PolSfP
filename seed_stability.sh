#!/usr/bin/env bash
# Paired gate runs from same data split; results from author data, never from demo.
set -euo pipefail
train="${1:?train manifest}";val="${2:?val manifest}"
for seed in 42 43 44 45 46; do
  for gate in learned entropy; do
    dpcu-polsfp train --train "$train" --val "$val" --gate "$gate" --seed "$seed" --out "weights/${gate}_seed${seed}.pt"
  done
done
