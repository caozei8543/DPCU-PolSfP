# Verification report for this reference repository

Recorded in the controlled assistant execution environment, November 2026.

## Checks completed

- Python source `compileall` passed.
- 11 unit tests passed: paper Fresnel DoLP numeric values, Stokes consistency, LUT validity, azimuth candidates, 90° Stokes/normal transforms, crop/augmentation, 7 gate forward paths, training gradient propagation, candidate oracle decomposition, connected-component Poisson integration, Chamfer self-distance, effective model optimizer.
- Reference toy-scene training ran 2 CPU epochs; wrote a `weights_only` checkpoint and read it back; evaluation returned standard angular metrics with oracle decomposition.
- One-checkpoint decision reversal and 5-bin ambiguity analysis wrote JSON.
- Twenty-five-condition physical sensitivity ran with fixed model weights, wrote JSON and produced a 350-DPI heatmap.
- Forward profiling and train-only physical calibration executed, wrote JSON.
- Synthetic diagnostic visualization saved successfully.

## What these tests do NOT establish

The original paper author's network weights, software modules, 8.9M parameter count, input calibration, original acquisition devices, raw LWIR/MWIR data, registered meshes, or the published performance figures were **not available** to these checks. The synthetic sample has the same scene in all splits to test pipeline plumbing only. No acceptance-level scientific reproduction is claimed.

## Requirements for scientific release

Check all items in `docs/REPRODUCIBILITY_CHECKLIST.md`, compare outputs on author's own original data and weights, then update this report with actual hashes, execution logs, environment and dates. Do not alter historical test claims without real evidence.
