# Code/figure mapping

- Figure 1 and Figure 2 are method diagrams; not computer-generated experiment outputs. Source editable diagrams must be supplied by authors.
- Figure 3 physics/geometry consistency: run author ablation configurations and compare normal/depth outputs using `scripts/visualize_scene.py` plus original surfaces.
- Figure 4 acquisition schematic/photograph: **real photo and author-created vector schematic only**; not a neural network output.
- Figure 5 ThermoPol16 method comparison: requires official external baseline predictions and paper model normal maps; `scripts/external_baseline_eval.py` computes unified normal errors but does not generate third-party prediction files.
- Figure 6 real MWIR normal/depth: use `scripts/visualize_scene.py`; corresponding physical metric data requires real scans/intrinsics.
- Figure 7 uncertainty distribution: `scripts/visualize_scene.py` renders DoLP/AoP/entropy/errors.
- Figure 8 entropy-stratified errors: `scripts/decision_analysis.py` produces numerics for five bins; plotting and design remains author-curated.
- Figure 9 decision reversal: `scripts/decision_analysis.py`; diagram remains a conceptual figure.
- Figure 10 parameter sensitivity: `dpcu-polsfp sensitivity` + `scripts/plot_sensitivity.py`.
- Figure 11 nine MWIR conditions: `scripts/plot_conditions.py` takes REAL per-condition output CSV.
- Figure 12 failure cases: `scripts/visualize_scene.py` helps inspect per-pixel errors; selecting actual failure samples requires original source data.

The figure numbering assumes the revision containing added experimental Fig. 4. Do not hardcode paper values into plots: they must be computed from the original measurements.
