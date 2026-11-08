# Paper reproduction / reviewer checklist

Tick only after verification from the ACTUAL author source and files.

- [ ] Name, affiliation, ORCID and corresponding-author permission confirmed.
- [ ] Exact 60/26 ThermoPolSynth split with seed 42, complete training list/checksums published.
- [ ] ThermoPol16 evaluation list and foreground masks preserved, including all 16 objects.
- [ ] MWIR C1--C9 five sequences per condition / true target identities / temperature logs documented.
- [ ] Real calibrated MWIR `I0,I45,I90,I135` and camera wavelength response provided where permitted.
- [ ] Calibration values η=1.82, RER=1.65, AoP offset 90° matched to actual calibration code and data.
- [ ] Author's real backbone/head definitions, exact 8.9M parameter count and actual inference order verified.
- [ ] 100 epoch optimizer/scheduler/augmentation/loss weights matched to authentic run logs.
- [ ] Original checkpoints with hash/size/training details attached.
- [ ] ThermoPol16 paper metric 6.58° / threshold 89.72% independently regenerated from provided checkpoint.
- [ ] MWIR four-condition and nine-condition MAEs reproduced from real held-out images and scan normals.
- [ ] Five seed gate comparison and 95% paired CI reproduced from actual model outputs.
- [ ] Candidate oracle and selection errors consistent per pixel with ground-truth conventions.
- [ ] Physical η/RER 5×5 sweep reproduced with fixed weights and no test-tuning.
- [ ] Real meshes, camera intrinsics, reference depth, scan alignment and 10k point sampling seeds provided.
- [ ] Depth RMSE 1.91%D and Chamfer 1.25%D reproduced with aligned visible surfaces.
- [ ] Baseline code provenance, official checkpoints, coordinate transforms, masks and exact timing protocols documented.
- [ ] Runtime end-to-end 180.0±6.5 ms reproduced on RTX 4090 including Poisson integration.
- [ ] Scripts do not leak real benchmark test samples into training/validation/calibration.
- [ ] Figure generation scripts and underlying intermediate arrays for all quantitative figures available.
- [ ] Third-party dataset redistribution rights and license confirmed.
- [ ] GitHub repository can be opened without authentication and install/CI steps succeed on fresh environment.

**A checked code repository is not equivalent to valid experimental evidence without the underlying real datasets and checkpoints.**
