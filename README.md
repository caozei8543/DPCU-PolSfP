# DPCU-PolSfP: Manuscript-derived reference code

> **Important provenance statement**: This is a **newly written, runnable reference implementation** based on the manuscript *Physics-Anchored Uncertainty-Guided Shape-from-Polarization for Infrared Polarimetric 3D Imaging*. It **is not** a copy of the authors' actual training/evaluation source. No author-provided experimental images, ground-truth meshes, weights, hardware-control software, per-scene outputs, or numeric benchmark results were supplied when this repository was constructed. **It does not substantiate the numerical claims (6.58° / 7.08° / 1.91% / 1.25%) without author verification.**
>
> Authors: **replace this notice only after comparing the repository against the true original source and reproducing the manuscript results from real datasets and checkpoints**. Reviewers must not mistake synthetic demonstration data or randomly trained reference checkpoints for published evidence.

## Purpose

A transparent software starting point for examination of the components described in the paper. Includes differentiable polarimetric physics, candidate-restricted inference, entropy-based routing, losses, calibrated LUT inversion, reconstruction metrics, masked normal integration, reproducible command-line examples and tests.

### Implemented components vs manuscript

| Paper section | Repository | Provenance and limitations |
|---|---|---|
| Eq. (1) Stokes / DoLP / AoP | `physics.py:stokes` | Exact four-angle linear-analyzer equations |
| Eq. (2) emission/reflection | `physics.py:signed_polarization` | Exact effective real-η Fresnel assumption; real absorbing materials need spectral/complex index calibration |
| Eq. (3) inversion & ±10° | `physics.py:ZenithLookup`, `model.py` | 0°–75°, 0.1° LUT; correction from reference conv head |
| Eq. (4) two candidates | `physics.py:normal_candidates` | Two unit normals differing by π azimuth, fixed 90° phase convention |
| Eqs. (5–6) entropy routing | `model.py` | Normalized local entropy, local/context logit blending |
| Eq. (7) candidate supervision | `losses.py` | CE + expected cosine distance, λn=0.5 |
| Eq. (8) DoLP reprojection | `losses.py` | Heteroscedastic penalty + residual floor 1e-3, bounded log scale |
| Eq. (9) ST Gumbel | `model.py` | Hard forward/soft backward in train; deterministic argmax in eval |
| Eq. (10) integrability | `losses.py` | Local curl; optional depth loss disabled as in manuscript |
| Masked Poisson / LSMR | `geometry.py` | Orthographic local least squares, 4-connected, per-component gauge |
| Normal/oracle/depth/Chamfer | `metrics.py`, `geometry.py` | Defined metric implementation, consistent masks, explicit units |
| Gate ablations | `scripts/gate_ablation.sh` | Reference gate variants, not author-fitted models |
| Five-seed studies | `scripts/seed_stability.sh` | Reference setup; requires author datasets |
| Decision reversals | `scripts/decision_analysis.py` | Paired within-checkpoint analysis |
| 5×5 physical sensitivity | CLI `sensitivity` | Frozen checkpoint; physical inversion re-evaluated |
| Code for FLIR SC5500-M acquisition | **not supplied** | Requires the camera SDK and actual rotation-stage SDK / sync logs |
| Exact backbone (8.9 M params) | **not supplied** | Present CNN (`width=32`) is an explicit implementation choice, NOT the manuscript's verified architecture |
| Official competing methods | **not bundled** | Evaluate separately using respective official licenses and saved normal maps |
| Real ThermoPol16/MWIR data & models | **not supplied** | See `docs/DATASET_CARD.md`, author must add with permission |

The core algorithms can be tested now, but **claim-level reproducibility requires the original model, exact data partitions, calibrations, camera coordinate conventions, and acquisition records**.

## Installation

A Conda configuration is also supplied as `environment.yml` for teams using Conda.

Python 3.10+; tested in the creation environment against installed PyTorch, NumPy and SciPy:

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[test]'
python -m pytest -q
# Optional end-to-end toy smoke test, synthetic-only:
# bash scripts/smoke_test.sh
```

Do not install or publish artifacts named `ours.pth` / real measurement files without confirming their provenance and permissions.

## Reproducible technical sanity check (SYNTHETIC ONLY)

```bash
dpcu-polsfp demo-data --out data/demo --size 64
dpcu-polsfp train --train data/demo/train.csv --val data/demo/val.csv \
    --out weights/toy_reference.pt --epochs 2 --batch-size 1 --width 8 --device cpu
dpcu-polsfp evaluate --manifest data/demo/test.csv \
    --checkpoint weights/toy_reference.pt --out results/toy_eval.json --device cpu
python scripts/decision_analysis.py --manifest data/demo/test.csv \
    --checkpoint weights/toy_reference.pt --out results/toy_decisions.json
```

**Warning**: The toy dataset has a single idealized sphere reused across train/val/test and the resulting score is **not** meaningful generalization accuracy. It is for code-path verification only. There are **no paper reproduction numbers bundled**.

## Using the authors' actual datasets

See [DATASET_CARD](docs/DATASET_CARD.md) for NPZ and CSV manifest conventions and the necessary calibration inputs. You must keep the paper's 60 synthetic training / 26 synthetic validation samples separate from all 16 real ThermoPol16 test objects. All nine MWIR configurations must remain entirely held out for zero-shot evaluation; the five sequences per condition are not independent specimens. Do not tune parameters on the test set.

Create `data/train.csv`, `data/val.csv`, `data/test.csv` with columns `id,path`. Every `.npz` must contain `i0,i45,i90,i135`, with linear-calibrated arrays (`H×W`, same radiometric units); optional `mask`, required `normals` for supervised work and optional `depth`. Axis convention: camera x right, y image down, positive z toward the observed target (positive normal z for visible facing surfaces). **Ensure the actual laboratory convention is transformed accordingly** before using any benchmark comparison.

```bash
# Run calibration on TRAIN ONLY, using training reference normals
# (the true authors' calibration script/ground-truth zenith convention may differ)
dpcu-polsfp calibrate --train data/train.csv --out results/calibration.json

# Fit a baseline reference CNN; this is not automatically the actual paper model
dpcu-polsfp train --train data/train.csv --val data/val.csv --epochs 100 \
    --width 32 --batch-size 8 --crop 512 --augment --out weights/author_to_verify.pt --device cuda

# Normal metrics, invalid LUT fraction, oracle and selection decomposition
dpcu-polsfp evaluate --manifest data/test.csv --checkpoint weights/author_to_verify.pt \
    --device cuda --out results/test.json

# Physical η and RER sensitivity without test-time training
dpcu-polsfp sensitivity --checkpoint weights/author_to_verify.pt \
    --manifest data/test.csv --device cuda --out results/parameter_grid.json
```

**Caution:** before treating `calibrate` outputs as the paper's coefficients, the authors must reconcile the fitted normalization, true zenith conventions, and experiment used to derive η=1.82/RER=1.65. The reference network's default coefficients are those manuscript numbers and are held fixed during inference; the calibration command does **not** modify the checkpoint automatically.

### Quantitative 3D geometry

Metric depth evaluation is available only when **real reference depth** and **camera intrinsics and range** are supplied. The depth integration uses a local orthographic approximation with `dx = range_m/fx` and `dy = range_m/fy`. It applies one global z-offset alignment, no rotational or scale fit, then computes depth RMSE and bidirectional Euclidean Chamfer distance normalized by the reference visible-surface bounding-box diagonal. The reference model can differ from the published implementation; inspect the solver and ground truth definitions before claiming exact paper replication.

```bash
dpcu-polsfp evaluate --manifest data/mwir_eval.csv --checkpoint weights/verified_original.pt \
    --geometry --fx 250 --fy 250 --cx 159.5 --cy 127.5 --range-m 1.2 \
    --out results/mwir_geometry.json --device cuda
```

**The shown intrinsics (`fx=fy=250`) are placeholders, NOT SC5500-M calibration values.** Supply measured camera intrinsics. Do **not** quote results produced with placeholders.

### Comparison with official external baselines

The manuscript cites DeepSfP, SfPWild and Kitazawa et al.'s official `SfPTER-ICCP2025` baseline. Their original code and weights are neither copied nor redistributed here. After independently obtaining official predictions and converting normals to the same camera frame, use:

```bash
python scripts/external_baseline_eval.py --manifest data/external_predictions.csv \
    --out results/official_method_comparison.json
```

`data/external_predictions.csv` has `id,path,pred_path`, with each prediction NPZ containing `normals` of shape H×W×3 or 3×H×W. Validate consistent masks, valid pixels, radiometric processing and conditions before making claims of a fair comparison.

## Reproduction checklist for *paper* results

See the [automated test report](docs/TEST_REPORT.md), [REPRODUCIBILITY_CHECKLIST](docs/REPRODUCIBILITY_CHECKLIST.md) and [AUTHOR_ACTION_REQUIRED](docs/AUTHOR_ACTION_REQUIRED.md). In particular, author must provide actual training scripts/config, **original trained checkpoint, 8.9 M backbone definition**, the registered scan geometry and calibration records, and quantitative outputs by condition. Merely publishing a plausible replacement model would **not** satisfy an audit of the published experiment.

## Security / scientific integrity

- `.npz` loads use `allow_pickle=False`; never accept untrusted serialized PyTorch checkpoints.
- The only provided demo data are synthetic and labeled as such; no private MWIR images are distributed.
- Training validation and test file manifests are explicit; no automatic test-data hyperparameter tuning.
- This reference code is **not an equipment photograph, instrument driver, calibrated sensor simulator, or evidence that experiments were actually executed**.
- Record `pip freeze`, a commit hash, seed, exact datasets, checkpoints, and hardware in the authors' experiment log.

## License

See [LICENSE_NOTICE.md](LICENSE_NOTICE.md). A final open-source license must be selected by the authors after checking third-party rights.

## GitHub publication

See [GITHUB_PUBLISH_GUIDE](docs/GITHUB_PUBLISH_GUIDE.md). Upload extracted repository files (rather than a single ZIP as one GitHub file), configure the public link, and confirm code/data rights and the explicit reference-implementation disclaimer before claiming it meets a journal reproducibility request.
