# Ablations -- STDC-Seg, Cityscapes val (IoU %%, delta vs full model in points)

generated 2026-09-27 10:41 by gen_ablation.py; noise floor = range across I0, Baseline2, Baseline: bnd_r1 0.32, bnd_r3 0.47, thin_r1 0.42, thin_r3 0.55, full 1.23

| Model variation | Status | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 | Full mIoU |
|---|---|---|---|---|---|---|
| Full model (RSR + BPM) | done | 39.77 | 49.82 | 38.62 | 48.39 | 77.25 |
| Baseline (no RSR, no BPM) | done | 35.80 (-3.96) | 44.91 (-4.91) | 33.16 (-5.46) | 41.84 (-6.55) | 75.82 (-1.43) |
| w/o RSR (BPM only) | done | 37.78 (-1.98) | 47.68 (-2.14) | 35.43 (-3.19) | 45.03 (-3.36) | 77.04 (-0.21) |
| w/o BPM (RSR only) | done | 37.57 (-2.20) | 47.08 (-2.75) | 35.40 (-3.22) | 44.53 (-3.86) | 76.60 (-0.65) |
| RSR w/o high-res features | done | 38.44 (-1.33) | 48.55 (-1.27) | 37.26 (-1.36) | 46.97 (-1.42) | 77.34 (+0.09) |
| RSR at stride 8 instead of 4 | done | 37.94 (-1.83) | 47.87 (-1.95) | 36.26 (-2.36) | 45.93 (-2.46) | 76.55 (-0.70) |
| RSR w/o residual | pending |  |  |  |  |  |
| RSR w/o logit input | pending |  |  |  |  |  |
| RSR w/o gate | pending |  |  |  |  |  |
| BPM w/ unweighted OHEM rank | done | 38.82 (-0.95) | 48.50 (-1.32) | 37.73 (-0.89) | 47.08 (-1.31) | 76.83 (-0.42) |
| BPM w/o OHEM (w=3) | done | 38.72 (-1.05) | 48.51 (-1.32) | 37.05 (-1.58) | 46.44 (-1.95) | 76.74 (-0.51) |
| w/o OHEM, w=1 | pending |  |  |  |  |  |
| w/o detail-head loss | pending |  |  |  |  |  |
| w/o auxiliary heads | pending |  |  |  |  |  |
