# Physical acquisition: detailed protocol checklist

This is documentation of the method described in the manuscript, **not code that controls FLIR or Thorlabs equipment**. Do not execute any instrument motions based only on this document.

1. Warm up and radiometrically calibrate the cooled FLIR SC5500-M InSb detector. Record dark/offset, nonuniformity maps and thermal stability.
2. Log camera model, serial if permitted, actual lens/f-number, exposure 1.5 ms, integration mode, hardware trigger, real optical passband and FPA temperature. The camera nominal 2.5--5.1 µm is not the same thing as the actual combined system response.
3. Install an appropriately mounted Thorlabs WP25M-IRA wire-grid polarizer before the camera lens; log stage make/model, encoder zero, polarizer transmission and extinction ratio. Zero the 0°,45°,90°,135° analyzer convention against a reference polarizer.
4. At an approximately 1.2 m target range, sequentially acquire 8 frames for each analyzer setting. Record full rotation-stabilization and readout times; **1.5 ms per exposure is not the complete four-angle capture cycle**.
5. Average each group of 8 linear radiometric images at fixed analyzer angle. Apply angle-dependent transmission corrections; spatially register images if needed. Save the ORIGINAL calibrated four-angle channels alongside metadata.
6. Record target material, sample ID, roughness/profile measurement, target temperature, ambient/background source temperatures, exact range, placement, motion and per-sequence timestamp.
7. For each C1--C9 condition capture five full sequences, which are repeats of one configuration rather than independent objects. Confirm target motion/negligible temperature drift during sequential capture.
8. Independently scan cooled objects with recorded Artec Space Spider geometry (if that is the actual ground truth), rigidly align mesh to camera via calibration target + ICP, log reprojection/registration residuals, camera intrinsics and coordinate-frame conversion.
9. Render mask, normals and depth using calibrated camera intrinsics; verify all sample IDs match their corresponding 4-angle captures. Never use real held-out MWIR scans as training targets for claimed zero-shot tests.
10. Run masked quantitative evaluation with one common reference geometry and mask per method, archived inference outputs, and no undocumented scale/rotational fitting.

**Reviewer requirement:** Physical arrangement photograph must be a genuine record of actual apparatus, not an AI-generated photograph or composite presented as evidence. Any illustrative schematic should be labeled as such.
