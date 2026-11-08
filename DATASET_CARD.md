# Dataset contracts, curation, and privacy

## Dataset provenance

The actual ThermoPolSynth/ThermoPol16 data and independently collected MWIR nine-condition dataset **are not included** in this assistant-generated repository. You must obtain permission from dataset owners before redistribution. This module is not an acquisition driver.

## Each scene NPZ

```
images/target_01.npz:
  i0      float32 [H,W] calibrated linear intensity at 0°
  i45     float32 [H,W] calibrated linear intensity at 45°
  i90     float32 [H,W] calibrated linear intensity at 90°
  i135    float32 [H,W] calibrated linear intensity at 135°
  normals float32 [H,W,3] or [3,H,W] camera-frame unit normal, SUPERVISED ONLY
  mask    uint8   [H,W] independent valid foreground mask
  depth   float32 [H,W] METRIC Z for quantitative geometry, if available
```

The radiometric calibration must include non-uniformity, detector-offset, four-angle transmission corrections, camera registration and angular phase calibration. Input image values must be comparable between training and evaluation. Their radiance scale is not inferred automatically. Brightness clipping or gamma coding changes Stokes interpretation. `normals` are not algorithm predictions; they are independent ground truth.

## CSV manifest

```
id,path
sample_0001,images/target_01.npz
sample_0002,images/target_02.npz
```

Paths are relative to the CSV file (or absolute). For external predictions: `id,path,pred_path`.

## Documented manuscript partitions

- ThermoPolSynth: 86 synthetic samples, split 60 train/26 validation with seed 42.
- ThermoPol16: 16 real LWIR held-out test cases; do not include in training/calibration/tuning.
- MWIR C1--C9: 9 configurations; 5 temporal sequences per configuration (not independent target specimens). All models zero-shot from LWIR train; no test-time adaptation.

The actual sample identities/split list have not been provided. Reconstructing a random 60/26 split without exact IDs **does not** establish identical experimental protocol.

## Original MWIR hardware as reported in the paper

FLIR SC5500-M, cooled InSb 320×256 at 30 μm pixel pitch; 27 mm F/3 optics; Thorlabs WP25M-IRA rotating wire-grid polarizer; analyzer angles 0°,45°,90°,135°, 1.5 ms integration time, 8 averaged frames per angle; about 1.2 m range. Camera nominal 2.5--5.1 μm and polarizer nominal 3--5 μm. Author must provide actual passband / filters and calibrated intrinsics.

Original reference geometry is described as Artec Space Spider, registered with camera by calibration target + ICP; source meshes and camera transforms must be released or described precisely. Registration residuals and thermal expansion differences must be disclosed.

## Validity and fairness

- Report a defined pixel mask and do not silently remove low-DoLP regions.
- Report invalid DoLP-inversion ratio BEFORE clamping.
- Geometry: evaluate visible surfaces at the common scale and coordinate frame, with at most one global depth offset.
- Five temporal captures of one specimen must not be described as five independent materials/specimens.
- Never upload personally identifiable data, proprietary scans or third-party weights without permission.
