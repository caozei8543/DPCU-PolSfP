# What the corresponding author must replace before public release as "the paper code"

This software did not originate from the authors' actual experiments and must be explicitly verified before citing as the official implementation. **Do not remove provenance warnings solely to appear reviewer-ready.**

## Items absent from the manuscript/upload

1. **Original network and trained parameters.** The paper names local and geometry-aware context branches but does not provide complete layers, kernel sizes, receptive fields, channels or weight count. The reference model uses a lightweight 8-channel input CNN that does NOT match the claimed 8.9M parameter design. Provide the real architecture and checkpoint(s).
2. **Original objective implementation.** Confirm mask definitions, real training units, resampling, correlation of temporal images, ST-Gumbel temperature, loss normalization, phase offset and channel normalization. Our supplied values match the text where specified; other choices are explicitly reference choices.
3. **Real data and scanner registration.** Supply permitted originals or a legal release subset; publish the nine-condition metadata and actual calibrated camera geometry. Never substitute synthetic demo data for MWIR acquisitions.
4. **Actual experiment scripts and raw logs.** Include exact training/validation split IDs, calibration inputs, five-seed model runs, gate comparisons, parameter sweeps and per-scene metrics, not just final values copied from manuscript.
5. **Official baselines.** Provide commands to obtain compatible release of Kitazawa et al.'s model and checkpoint, log its exact commit. Do not republish models without rights.
6. **Camera acquisition and rotation stage.** Driver/API source and calibration procedure not supplied; actual control software may be vendor-specific and must be documented with operating safety checks.
7. **Numerical performance consistency.** Confirm how original 4-condition normal statistics and ground truth meshes were generated, including differences between earlier draft's scanner designation and revised draft.
8. **No invented evidence.** Do not fabricate 6.58°/7.08° outcomes, checkpoint file contents, equipment metadata, condition measurements or GitHub activity.
9. **Publishable metadata**: add license, citation metadata, DOI if assigned, authorship approval, dataset licenses, useful issue templates and explicit contact instructions.

## Reviewer-facing disclosure template (adapt to actual status)

"We provide a manuscript-derived reference implementation for the mathematical framework and evaluation pipeline. The source code supplied here is not yet demonstrated to reproduce the original trained system; the original model weights, calibrated data, and per-condition experimental artifacts will be released [only if/when author actually releases them]."

For the reviewer demand **"Code and acquired dataset be presented on some internet repository"**, code alone is insufficient. Public raw data/subset and precise acquisition protocol should be linked or the restriction should be explicitly justified to the journal. Do not assert availability without a working URL.
