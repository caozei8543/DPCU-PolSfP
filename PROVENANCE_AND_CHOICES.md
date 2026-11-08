# Implementation choices not uniquely determined by the manuscript

| Topic | Explicit in manuscript | Reference-code choice |
|---|---|---|
| Feature encoder | four raw measurements + S0,DoLP,cos2AoP,sin2AoP | three 3×3 convolution blocks with GroupNorm+SiLU; width=32 |
| Context | geometry-aware context features from candidate normal maps | two dilated convolution layers |
| Bounded correction | ±10° via tanh and clamp to 0°–75° | one 1×1 CNN head |
| Log residual scale | range −7 to 3 | one 1×1 CNN head |
| Fusion gate | entropy-based logit interpolation | direct implementation; optional alternate gates for controls |
| Augmentation | polarization-consistent flips and 90° rotations | exact spin-2 Stokes transforms and x/y normal-vector transforms |
| Polarimetry | measured calibrated images; spectral response depends on hardware | real-index dielectric effective Fresnel law, directionally opaque Kirchhoff simplification |
| Normal integration | foreground-masked Poisson LSMR with Neumann-type constraints | undirected 4-connectivity, averaged neighbor gradients, one gauge per component |
| Chamfer | symmetric mean Euclidean nearest-neighbor normalised by D | defined with no square and no ICP, all visible pixels unless sampled explicitly |
| Train/validation | 60/26 synthetic, 16 held-out LWIR; seed 42 | manifests must be supplied externally; no inferable IDs |
| Official model params | 8.9M claimed for DPCU | reference CNN is not that model and no pretrained weights are provided |

The default evaluation `geometry.py` uses **all valid visible-surface points** by default, while the manuscript specifies 10,000 sampled points. To claim paper-level Chamfer reproduction, implement/verify its identical sampling with fixed RNG and actual scanner geometry; especially on 320×256 maps, sampling affects reported distances.

Training by default uses batch size 1 for out-of-the-box mixed dataset sizes; the manuscript uses batch 8 and 512×512 crops. The reference data loader supports `--crop 512 --augment` (geometric transforms, joint ±5% radiance gain and independent channel noise σ=0.005), but the exact author preprocessing must still be independently verified.

The `physics.py` reference assumes positive real index and a fixed polarization mode. It cannot simulate a mode switch, complex refraction in absorbing metals, or wavelength-dependent effective RER without user-provided calibration measurements.

**Identifiability caveat:** fitting η and RER to DoLP alone can produce near-equivalent parameter pairs; low Huber loss is not proof of independently measured optical constants. Original calibration measurements and priors are needed to justify η=1.82/RER=1.65.
