# Evaluation digest -- key results for the paper

generated 2026-11-02 19:02 from resultData/cache/eval by gen_digest.py

Paper: "DUET" -- two plug-and-play additions for real-time semantic segmentation that improve
BOUNDARY and THIN-STRUCTURE accuracy: RSR (Residual Sub-grid Refinement: a stride-4 refinement head,
+0.086M params) fixes output resolution; BPM (Boundary-Priority Mining: boundary-weighted OHEM loss,
zero inference cost) fixes the training objective. "Ours" = RSR + BPM.

How to read the numbers:
- All values are IoU in %; deltas in percentage points.
- Full mIoU = standard full-image mIoU. It is a REGRESSION GUARD, not the claim: on Cityscapes its
  run-to-run spread (0.88 pts between the 2 fresh-clone baseline runs) exceeds several effects.
- Bnd r=N = IoU restricted to pixels within N px of a ground-truth class boundary (a trimap IoU).
  PRIMARY metric. Thin r=N = mean IoU of thin classes inside that band.
- Thin classes: Cityscapes/SYNTHIA = pole, traffic light, traffic sign, rider, motorcycle, bicycle.
  RUGD = pole, sign, fence, bicycle, log (same rule: structures thinner than one stride-8 cell).
- STDC-Seg = STDC2 backbone, scale 0.75 eval. HRNet = HRNetV2-W48, eval at native full resolution
  (Cityscapes 1024x2048, SYNTHIA 760x1280, RUGD 550x688). HRNet uses one recipe on all three datasets
  and both arms (reduced 120-epoch schedule, crop 512x1024, batch 3, OHEM); only dataset-dictated fields
  differ (class count, base size, no Cityscapes class weights). Compare within a host/dataset only.
- Cityscapes STDC baseline = 2-run mean of the 2 fresh-clone baseline runs (Baseline, Baseline2). I0 is
  excluded: it was trained with the bnd_weight flag (lambda=1), not from a fresh clone of the original
  STDC-Seg repo. Every other row is a single run (n=1).
- Both hosts are scored on the same SYNTHIA (940) and RUGD (733) val images.

## 1. Main results: baseline vs Ours on every dataset and host

Scale 0.75 for STDC, native full resolution for HRNet (each host's training-time val protocol). RUGD means use only classes with >= 0.1% of val pixels (RUGD val is two videos; rarer classes give meaningless IoU); "all cls" is the unfiltered mean that the training log reports.

| Dataset | Method | n | Full mIoU | Full mIoU (all cls) | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|---|---|---|
| Cityscapes | STDC-Seg | 2 | 75.56 | 75.56 | 35.79 | 44.89 | 33.25 | 41.91 |
| Cityscapes | STDC-Seg + Ours | 1 | 77.25 (+1.69) | 77.25 (+1.69) | 39.77 (+3.98) | 49.82 (+4.93) | 38.62 (+5.38) | 48.39 (+6.48) |
| Cityscapes | HRNet-W48 | 1 | 75.62 | 75.62 | 40.88 | 50.89 | 41.70 | 51.60 |
| Cityscapes | HRNet-W48 + Ours | 1 | 77.22 (+1.60) | 77.22 (+1.60) | 43.24 (+2.36) | 53.75 (+2.85) | 45.28 (+3.58) | 55.81 (+4.21) |
| SYNTHIA | STDC-Seg | 1 | 72.56 | 72.56 | 37.41 | 47.64 | 31.40 | 40.45 |
| SYNTHIA | STDC-Seg + Ours | 1 | 77.19 (+4.63) | 77.19 (+4.63) | 43.67 (+6.26) | 55.31 (+7.67) | 38.56 (+7.16) | 48.93 (+8.48) |
| SYNTHIA | HRNet-W48 | 1 | 85.82 | 85.82 | 60.90 | 71.50 | 55.05 | 65.10 |
| SYNTHIA | HRNet-W48 + Ours | 1 | 88.46 (+2.65) | 88.46 (+2.65) | 66.80 (+5.90) | 76.25 (+4.75) | 62.55 (+7.50) | 71.50 (+6.40) |
| RUGD | STDC-Seg | 1 | 53.14 | 41.56 | 25.22 | 30.51 | 17.16 | 20.28 |
| RUGD | STDC-Seg + Ours | 1 | 54.62 (+1.48) | 43.75 (+2.19) | 26.65 (+1.43) | 33.13 (+2.62) | 19.23 (+2.07) | 23.21 (+2.92) |
| RUGD | HRNet-W48 | 1 | 58.82 | 48.52 | 31.95 | 38.90 | 19.09 | 24.03 |
| RUGD | HRNet-W48 + Ours | 1 | 60.00 (+1.18) | 50.15 (+1.64) | 35.65 (+3.70) | 42.56 (+3.66) | 24.58 (+5.49) | 30.03 (+6.00) |

Gain of Ours over its own baseline, every host x dataset (points; the column a cross-host "consistent improvement" claim rests on):

| Dataset | Host | d Full mIoU | d Full mIoU (all cls) | d Bnd r=1 | d Bnd r=3 | d Thin r=1 | d Thin r=3 |
|---|---|---|---|---|---|---|---|
| Cityscapes | STDC-Seg | +1.69 | +1.69 | +3.98 | +4.93 | +5.38 | +6.48 |
| Cityscapes | HRNet-W48 | +1.60 | +1.60 | +2.36 | +2.85 | +3.58 | +4.21 |
| SYNTHIA | STDC-Seg | +4.63 | +4.63 | +6.26 | +7.67 | +7.16 | +8.48 |
| SYNTHIA | HRNet-W48 | +2.65 | +2.65 | +5.90 | +4.75 | +7.50 | +6.40 |
| RUGD | STDC-Seg | +1.48 | +2.19 | +1.43 | +2.62 | +2.07 | +2.92 |
| RUGD | HRNet-W48 | +1.18 | +1.64 | +3.70 | +3.66 | +5.49 | +6.00 |

Cityscapes STDC-Seg effect size as a multiple of run-to-run noise (|delta| / range across the 2 fresh-clone baseline runs):

| Metric | Delta | Run-to-run noise (range) | x noise |
|---|---|---|---|
| Full mIoU | +1.69 | 0.88 | 1.9 |
| Bnd r=1 | +3.98 | 0.32 | 12.3 |
| Bnd r=3 | +4.93 | 0.47 | 10.4 |
| Thin r=1 | +5.38 | 0.31 | 17.3 |
| Thin r=3 | +6.48 | 0.55 | 11.8 |

## 2. Run-to-run noise floor (Cityscapes, STDC-Seg baseline, 2 fresh-clone runs)

| Run | Note | Full mIoU | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|---|
| Baseline | fresh-clone baseline run | 76.00 | 35.95 | 45.13 | 33.40 | 42.18 |
| Baseline2 | fresh-clone baseline run | 75.12 | 35.63 | 44.65 | 33.09 | 41.63 |
| range | noise floor (Baseline vs Baseline2) | 0.88 | 0.32 | 0.47 | 0.31 | 0.55 |
| I0 | excluded: trained with the bnd_weight flag (lambda=1), not a fresh clone | 76.35 | 35.83 | 44.95 | 32.99 | 41.70 |

I0 is shown for reference only; it is not in the baseline mean or the range.

## 3. Component analysis (Cityscapes, STDC-Seg, gains vs 2-run baseline mean)

| Configuration | Run | Full mIoU | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|---|
| Baseline | 2-run mean | 75.56 | 35.79 | 44.89 | 33.25 | 41.91 |
| + BPM only | I1 | 77.04 (+1.49) | 37.78 (+1.99) | 47.68 (+2.79) | 35.43 (+2.18) | 45.03 (+3.12) |
| + RSR only | H1 | 76.60 (+1.04) | 37.57 (+1.78) | 47.08 (+2.18) | 35.40 (+2.16) | 44.53 (+2.62) |
| + RSR + BPM (Ours) | HI1 | 77.25 (+1.69) | 39.77 (+3.98) | 49.82 (+4.93) | 38.62 (+5.38) | 48.39 (+6.48) |
| sum of single-arm gains | I1 + H1 (additive prediction) | +2.53 | +3.77 | +4.97 | +4.34 | +5.74 |
| Ours minus additive prediction | > noise = super-additive | -0.84 | +0.21 | -0.04 | +1.04 | +0.74 |

Reading: each arm alone gives about half the boundary gain; together they are additive on overall boundary IoU and super-additive only on thin classes (single runs). Do not use the sign of res_scale as evidence of interaction (the output is res_scale * conv, so the sign is not identifiable).

## 4. Ablations (Cityscapes, STDC-Seg; every row = full model with one thing changed; delta vs full model)

Primary columns are boundary/thin. Full-mIoU ablation deltas are all below its 0.88-pt run-to-run noise, so full mIoU cannot rank ablations; report it as the guard.

| Model variation | Run | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 | Full mIoU | abs(d Bnd r=1) / noise |
|---|---|---|---|---|---|---|---|
| Baseline (no RSR, no BPM) | 2-run mean | 35.79 (-3.98) | 44.89 (-4.93) | 33.25 (-5.38) | 41.91 (-6.48) | 75.56 (-1.69) | 12.3 |
| Full model (RSR + BPM) | HI1 | 39.77 | 49.82 | 38.62 | 48.39 | 77.25 |  |
| w/o RSR (BPM only) | I1 | 37.78 (-1.98) | 47.68 (-2.14) | 35.43 (-3.19) | 45.03 (-3.36) | 77.04 (-0.21) | 6.1 |
| w/o BPM (RSR only) | H1 | 37.57 (-2.20) | 47.08 (-2.75) | 35.40 (-3.22) | 44.53 (-3.86) | 76.60 (-0.65) | 6.8 |
| RSR w/o high-res features | ABL-M1-noHR | 38.44 (-1.33) | 48.55 (-1.27) | 37.26 (-1.36) | 46.97 (-1.42) | 77.34 (+0.09) | 4.1 |
| RSR at stride 8 instead of 4 | ABL-M2-s8 | 37.94 (-1.83) | 47.87 (-1.95) | 36.26 (-2.36) | 45.93 (-2.46) | 76.55 (-0.70) | 5.6 |
| BPM w/ unweighted OHEM rank | ABL-L1-unwRank | 38.82 (-0.95) | 48.50 (-1.32) | 37.73 (-0.89) | 47.08 (-1.31) | 76.83 (-0.42) | 2.9 |
| BPM w/o OHEM (w=3) | ABL-L2-noOHEM | 38.72 (-1.05) | 48.51 (-1.32) | 37.05 (-1.58) | 46.44 (-1.95) | 76.74 (-0.51) | 3.2 |

Not yet trained (no checkpoint): RSR w/o residual, RSR w/o logit input, RSR w/o gate, w/o OHEM, w=1, w/o detail-head loss, w/o auxiliary heads, w=2, w=5, w=8, r=1, r=2, r=5.

## 5. Thin classes, boundary band r=1

Cityscapes:

| Method | Pole | Tr.Light | Sign | Rider | M.bike | Bike | Thin mean |
|---|---|---|---|---|---|---|---|
| STDC-Seg (2-run mean) | 33.32 | 33.73 | 36.68 | 32.64 | 25.75 | 37.38 | 33.25 |
| STDC-Seg + BPM only | 37.63 | 36.14 | 40.29 | 34.66 | 27.48 | 36.40 | 35.43 |
| STDC-Seg + RSR only | 38.41 | 34.86 | 38.10 | 33.59 | 28.67 | 38.80 | 35.40 |
| STDC-Seg + Ours | 44.83 | 39.21 | 43.89 | 34.27 | 28.66 | 40.88 | 38.62 |
| HRNet-W48 | 49.00 | 42.77 | 48.12 | 37.87 | 30.41 | 42.02 | 41.70 |
| HRNet-W48 + Ours | 52.85 | 46.38 | 51.63 | 41.74 | 34.54 | 44.53 | 45.28 |

Thin classes, boundary band r=1 (SYNTHIA):

| Method | Pole | Tr.Light | Sign | Rider | M.bike | Bike | Thin mean |
|---|---|---|---|---|---|---|---|
| STDC-Seg | 31.22 | 32.50 | 26.16 | 34.65 | 39.25 | 24.59 | 31.40 |
| STDC-Seg + Ours | 43.83 | 41.45 | 33.29 | 39.13 | 43.57 | 30.07 | 38.56 |
| HRNet-W48 | 61.87 | 53.22 | 53.14 | 56.63 | 60.88 | 44.59 | 55.05 |
| HRNet-W48 + Ours | 70.33 | 62.03 | 60.78 | 63.16 | 66.82 | 52.19 | 62.55 |

Thin classes, boundary band r=1 (RUGD); classes marked (excl.) are below the 0.1% support filter and are not in the thin mean:

| Method | pole (excl.) | sign (excl.) | fence | log | Thin mean |
|---|---|---|---|---|---|
| STDC-Seg | 15.67 | 0.00 | 17.89 | 16.43 | 17.16 |
| STDC-Seg + Ours | 27.62 | 0.00 | 22.06 | 16.40 | 19.23 |
| HRNet-W48 | 34.61 | 0.91 | 18.28 | 19.90 | 19.09 |
| HRNet-W48 + Ours | 43.28 | 0.00 | 22.08 | 27.08 | 24.58 |

## 6. Robustness checks

SYNTHIA single arm (BPM only; this run's flags were not logged, identity inferred from its weights having no RSR module):

| Method | Full mIoU | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|
| STDC-Seg | 72.56 | 37.41 | 47.64 | 31.40 | 40.45 |
| + BPM only | 72.90 (+0.34) | 38.17 (+0.76) | 48.53 (+0.89) | 31.81 (+0.41) | 41.20 (+0.75) |
| + RSR + BPM (Ours) | 77.19 (+4.63) | 43.67 (+6.26) | 55.31 (+7.67) | 38.56 (+7.16) | 48.93 (+8.48) |

Input-size check: scale 0.75 on SYNTHIA/RUGD is not a multiple of 32 (the network stride), which distorts predictions. Re-evaluated at the nearest 32-divisible size; the improvement holds at both. (SYNTHIA: every model scores ~5.8 mIoU higher at 576x960.)

| Dataset | Input | Base full | Ours full | d Full mIoU | d Bnd r=1 | d Bnd r=3 | d Thin r=1 | d Thin r=3 |
|---|---|---|---|---|---|---|---|---|
| synthia | scale 0.75 (570x960) | 72.56 | 77.19 | +4.63 | +6.26 | +7.67 | +7.16 | +8.48 |
| synthia | 32-divisible (576x960) | 78.38 | 82.16 | +3.79 | +7.43 | +7.13 | +8.23 | +8.60 |
| rugd | scale 0.75 (412x516) | 53.14 | 54.62 | +1.48 | +1.43 | +2.62 | +2.07 | +2.92 |
| rugd | 32-divisible (416x512) | 54.77 | 55.99 | +1.22 | +1.02 | +1.83 | +2.11 | +2.26 |

## 7. Cost (Tesla T4; training = one full iteration at the training batch -- STDC 16x512x1024, HRNet 3x512x1024; inference = batch 1, 1024x2048 image -> label map)

Params/GMACs are exact and are the primary cost claim. Throughput is pooled over 30 interleaved rounds on a thermally throttling T4; the % columns are paired per-round ratios vs the baseline (about +/-3% noise). Memory = peak allocated tensors.

| Method | Params (M) | GMACs | Train it/s | d train | Train mem (GB) | Infer img/s | d infer | Infer mem (GB) |
|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 16.079 | 66.8 | 0.530 |  | 10.81 | 11.82 |  | 0.46 |
| STDC-Seg + Ours | 16.166 | 73.1 | 0.489 | -8.5% | 11.39 | 11.28 | -4.9% | 0.46 |
| HRNet-W48 | 65.859 | 747.5 | 0.468 |  | 8.21 | 1.34 |  | 1.86 |
| HRNet-W48 + Ours | 65.946 | 792.5 | 0.452 | -4.8% | 8.74 | 1.27 | -6.6% | 1.86 |

## 8. Caveats to state in the paper (so reviewers do not find them first)

- All "Ours" rows and all non-Cityscapes/HRNet rows are single runs (n=1); HI1 seed replicates are pending. HRNet has no seed-noise estimate of its own (one run per arm per dataset, ~4-8 GPU-days each), so treat HRNet full-mIoU gains under ~0.88 pts (the STDC Cityscapes run-to-run range) as noise-level.
- Full-mIoU gains: Cityscapes STDC +1.69 is only ~1.9x run-to-run noise (range of 2 runs); claim "no regression", not "improves mIoU".
- Checkpoints are the best-val-mIoU checkpoint for every arm (same rule for all), evaluated on val.
- HRNet-W48 Cityscapes baseline (75.6) is below the published number because of the reduced 120-epoch schedule; both arms share it. SYNTHIA/RUGD use the same recipe unchanged (no per-dataset tuning). On RUGD the 512x1024 training crop is wider than the 550x688 images and is padded with ignore, identically in both arms.
- HRNet full mIoU on SYNTHIA is much higher than STDC-Seg's (different backbone, native-resolution input vs scale 0.75); compare hosts only through their deltas.
- RUGD: several classes are rare in val; means are support-filtered (>= 0.1% of val pixels). RUGD thin mean covers only fence and log (bicycle absent; pole, sign below threshold -- pole still shows the largest STDC-Seg RUGD boundary gain, 15.7 -> 27.6, but is excluded from the mean).
- Cost is not zero for a real-time method (section 7). The strongest defence is an equal-latency comparison (baseline at a larger input scale vs Ours); not in this digest yet.

