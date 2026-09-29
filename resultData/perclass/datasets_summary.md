# All datasets -- full mIoU and boundary metrics (%, deltas in points vs host baseline)

generated 2026-09-27 21:26 by gen_perclass.py. Full mIoU = regression guard; boundary/thin = primary. RUGD means are support-filtered (>= 0.1% of val px); "all classes" is what train.py logs.

| Dataset | Method | Full mIoU | Full mIoU (all classes) | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|---|---|
| cityscapes | STDC-Seg | 75.82 | 75.82 | 35.80 | 44.91 | 33.16 | 41.84 |
| cityscapes | STDC-Seg + Ours | 77.25 (+1.43) | 77.25 (+1.43) | 39.77 (+3.96) | 49.82 (+4.91) | 38.62 (+5.46) | 48.39 (+6.55) |
| cityscapes | HRNet-W48 | 75.62 | 75.62 | 40.88 | 50.89 | 41.70 | 51.60 |
| cityscapes | HRNet-W48 + Ours | 77.22 (+1.60) | 77.22 (+1.60) | 43.24 (+2.36) | 53.75 (+2.85) | 45.28 (+3.58) | 55.81 (+4.21) |
| synthia | STDC-Seg | 72.56 | 72.56 | 37.41 | 47.64 | 31.40 | 40.45 |
| synthia | STDC-Seg + Ours | 77.19 (+4.63) | 77.19 (+4.63) | 43.67 (+6.26) | 55.31 (+7.67) | 38.56 (+7.16) | 48.93 (+8.48) |
| synthia | STDC-Seg [Synthia-I1] | 72.90 (+0.34) | 72.90 (+0.34) | 38.17 (+0.76) | 48.53 (+0.89) | 31.81 (+0.41) | 41.20 (+0.75) |
| rugd | STDC-Seg | 53.14 | 41.56 | 25.22 | 30.51 | 17.16 | 20.28 |
| rugd | STDC-Seg + Ours | 54.62 (+1.48) | 43.75 (+2.19) | 26.65 (+1.43) | 33.13 (+2.62) | 19.23 (+2.07) | 23.21 (+2.92) |

## synthia: scale 0.75 vs 32-divisible input

| Input | Method | Full mIoU | Full mIoU (all classes) | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|---|---|
| scale 0.75 | STDC-Seg | 72.56 | 72.56 | 37.41 | 47.64 | 31.40 | 40.45 |
| scale 0.75 | STDC-Seg + Ours | 77.19 | 77.19 | 43.67 | 55.31 | 38.56 | 48.93 |
| scale 0.75 | delta (Ours - baseline) | +4.63 | +4.63 | +6.26 | +7.67 | +7.16 | +8.48 |
| 32-div 576x960 | STDC-Seg | 78.38 | 78.38 | 44.71 | 57.15 | 35.74 | 46.50 |
| 32-div 576x960 | STDC-Seg + Ours | 82.16 | 82.16 | 52.14 | 64.27 | 43.97 | 55.10 |
| 32-div 576x960 | delta (Ours - baseline) | +3.79 | +3.79 | +7.43 | +7.13 | +8.23 | +8.60 |

## rugd: scale 0.75 vs 32-divisible input

| Input | Method | Full mIoU | Full mIoU (all classes) | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|---|---|
| scale 0.75 | STDC-Seg | 53.14 | 41.56 | 25.22 | 30.51 | 17.16 | 20.28 |
| scale 0.75 | STDC-Seg + Ours | 54.62 | 43.75 | 26.65 | 33.13 | 19.23 | 23.21 |
| scale 0.75 | delta (Ours - baseline) | +1.48 | +2.19 | +1.43 | +2.62 | +2.07 | +2.92 |
| 32-div 416x512 | STDC-Seg | 54.77 | 38.86 | 26.36 | 32.46 | 18.54 | 22.70 |
| 32-div 416x512 | STDC-Seg + Ours | 55.99 | 44.77 | 27.37 | 34.29 | 20.65 | 24.95 |
| 32-div 416x512 | delta (Ours - baseline) | +1.22 | +5.91 | +1.02 | +1.83 | +2.11 | +2.26 |
