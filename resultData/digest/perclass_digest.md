# Per-class digest -- per-class IoU (%) for the paper tables

generated 2026-09-29 18:19 from resultData/cache/eval by gen_digest.py

Paper: "DUET" -- two plug-and-play additions for real-time semantic segmentation that improve
BOUNDARY and THIN-STRUCTURE accuracy: RSR (Residual Sub-grid Refinement: a stride-4 refinement head,
+0.086M params) fixes output resolution; BPM (Boundary-Priority Mining: boundary-weighted OHEM loss,
zero inference cost) fixes the training objective. "Ours" = RSR + BPM.

How to read the numbers:
- All values are IoU in %; deltas in percentage points.
- Full mIoU = standard full-image mIoU. It is a REGRESSION GUARD, not the claim: on Cityscapes its
  seed-to-seed spread (1.23 pts over 3 identical baseline runs) exceeds several effects.
- Bnd r=N = IoU restricted to pixels within N px of a ground-truth class boundary (a trimap IoU).
  PRIMARY metric. Thin r=N = mean IoU of thin classes inside that band.
- Thin classes: Cityscapes/SYNTHIA = pole, traffic light, traffic sign, rider, motorcycle, bicycle.
  RUGD = pole, sign, fence, bicycle, log (same rule: structures thinner than one stride-8 cell).
- STDC-Seg = STDC2 backbone, scale 0.75 eval. HRNet = HRNetV2-W48, full-res eval (reduced 120-epoch
  schedule, identical for both arms). Compare within a host/dataset only.
- Cityscapes STDC baseline = mean of 3 identical-config seeds; every other row is a single run (n=1).

Per-class table layout follows a standard "Method x class + mIoU" table. "Delta" rows are Ours minus baseline. For Cityscapes STDC, the "seed range" row is the spread of that class across 3 identical baseline seeds; a delta marked * exceeds it (i.e. is larger than seed noise). Classes absent from a val set are omitted.

## Cityscapes

### Full image

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg (3-seed mean) | 98.11 | 84.70 | 91.85 | 56.76 | 59.56 | 56.57 | 65.63 | 74.58 | 91.58 | 62.70 | 94.25 | 78.59 | 59.02 | 94.47 | 78.39 | 85.15 | 76.94 | 57.75 | 74.02 | 75.82 |
| STDC-Seg + Ours | 98.28 | 85.96 | 92.27 | 52.20 | 60.02 | 63.84 | 69.57 | 77.44 | 92.23 | 66.50 | 94.78 | 80.81 | 60.08 | 95.00 | 79.20 | 84.33 | 77.50 | 61.90 | 75.83 | 77.25 |
| HRNet-W48 | 98.35 | 86.17 | 92.73 | 55.08 | 61.53 | 67.35 | 72.58 | 79.91 | 92.81 | 65.28 | 94.53 | 82.02 | 61.15 | 95.23 | 77.85 | 77.40 | 38.72 | 61.04 | 77.06 | 75.62 |
| HRNet-W48 + Ours | 98.34 | 86.23 | 93.26 | 61.99 | 62.93 | 70.38 | 74.95 | 82.11 | 93.07 | 66.11 | 94.58 | 84.17 | 64.12 | 95.70 | 75.87 | 80.26 | 38.02 | 66.24 | 78.89 | 77.22 |

### Full image -- deltas (points)

| Row | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg: Ours - base | +0.17* | +1.26* | +0.42* | -4.56* | +0.46 | +7.26* | +3.95* | +2.86* | +0.65* | +3.80* | +0.54* | +2.22* | +1.06 | +0.54* | +0.81 | -0.81 | +0.56 | +4.16 | +1.81* | +1.43 |
| STDC-Seg seed range | 0.04 | 0.17 | 0.07 | 4.51 | 1.38 | 0.73 | 1.00 | 0.57 | 0.20 | 3.16 | 0.13 | 0.13 | 1.34 | 0.11 | 4.11 | 3.61 | 7.76 | 8.74 | 0.42 |  |
| HRNet-W48: Ours - base | -0.01 | +0.06 | +0.53 | +6.91 | +1.41 | +3.03 | +2.37 | +2.20 | +0.26 | +0.83 | +0.05 | +2.15 | +2.97 | +0.47 | -1.97 | +2.86 | -0.70 | +5.20 | +1.84 | +1.60 |

### Boundary band r=1

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg (3-seed mean) | 46.82 | 41.45 | 41.36 | 21.42 | 24.75 | 33.18 | 32.94 | 36.72 | 44.15 | 29.90 | 47.36 | 40.03 | 31.98 | 48.61 | 30.33 | 37.57 | 27.58 | 27.04 | 37.10 | 35.80 |
| STDC-Seg + Ours | 49.39 | 45.56 | 44.88 | 23.50 | 26.52 | 44.83 | 39.21 | 43.89 | 47.44 | 33.92 | 52.23 | 45.16 | 34.27 | 52.13 | 32.35 | 39.45 | 31.33 | 28.66 | 40.88 | 39.77 |
| HRNet-W48 | 47.72 | 48.13 | 46.00 | 23.40 | 27.11 | 49.00 | 42.77 | 48.12 | 48.66 | 33.88 | 54.52 | 48.29 | 37.87 | 54.03 | 35.18 | 39.28 | 20.31 | 30.41 | 42.02 | 40.88 |
| HRNet-W48 + Ours | 49.32 | 49.22 | 48.76 | 27.93 | 30.53 | 52.85 | 46.38 | 51.63 | 50.08 | 35.80 | 56.80 | 51.31 | 41.74 | 54.57 | 35.05 | 42.73 | 17.80 | 34.54 | 44.53 | 43.24 |

### Boundary band r=1 -- deltas (points)

| Row | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg: Ours - base | +2.57* | +4.11* | +3.52* | +2.08* | +1.78* | +11.65* | +6.27* | +7.17* | +3.29* | +4.02* | +4.88* | +5.13* | +2.29* | +3.51* | +2.02* | +1.88* | +3.75 | +1.62 | +3.78* | +3.96 |
| STDC-Seg seed range | 0.42 | 0.94 | 0.18 | 1.47 | 1.77 | 0.58 | 2.39 | 0.14 | 0.24 | 1.59 | 0.09 | 1.75 | 2.21 | 0.88 | 0.23 | 1.83 | 5.12 | 4.56 | 1.15 |  |
| HRNet-W48: Ours - base | +1.60 | +1.09 | +2.76 | +4.54 | +3.42 | +3.85 | +3.60 | +3.51 | +1.42 | +1.92 | +2.28 | +3.02 | +3.87 | +0.54 | -0.13 | +3.46 | -2.51 | +4.13 | +2.51 | +2.36 |

### Boundary band r=3 -- deltas (points)

| Row | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg: Ours - base | +3.46* | +4.73* | +5.08* | +2.40* | +2.47* | +12.89* | +7.73* | +8.33* | +4.65* | +4.72* | +5.93* | +6.15* | +3.47* | +3.98* | +3.09* | +2.29* | +5.08 | +2.19 | +4.70* | +4.91 |
| STDC-Seg seed range | 0.32 | 0.66 | 0.30 | 1.80 | 1.62 | 0.76 | 2.54 | 0.36 | 0.21 | 1.83 | 0.01 | 1.46 | 1.63 | 0.44 | 0.37 | 1.95 | 6.64 | 4.97 | 1.01 |  |
| HRNet-W48: Ours - base | +1.86 | +1.70 | +2.94 | +5.32 | +3.64 | +3.97 | +4.05 | +4.17 | +1.68 | +2.23 | +1.51 | +3.82 | +4.45 | +1.58 | +0.87 | +4.51 | -2.67 | +5.53 | +3.09 | +2.85 |

Summary: STDC-Seg + Ours improves 19/19 classes at boundary r=1 (largest: Pole +11.7, Sign +7.2, Tr.Light +6.3). HRNet-W48 + Ours improves 17/19 classes at boundary r=1 (largest: Wall +4.5, M.bike +4.1, Rider +3.9); decreases: Truck -0.1, Train -2.5. Rare classes (truck, bus, train, motorcycle) swing several points between identical seeds; make no per-class claims about them.

## SYNTHIA (STDC-Seg, n=1)

SYNTHIA labels use Cityscapes classes; terrain, truck and train do not occur in val.

### Full image

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Sky | Person | Rider | Car | Bus | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 89.60 | 88.69 | 94.06 | 78.63 | 56.70 | 59.47 | 61.64 | 61.25 | 83.63 | 95.58 | 71.75 | 58.04 | 80.04 | 88.54 | 63.97 | 29.34 | 72.56 |
| STDC-Seg + Ours | 91.54 | 90.43 | 94.84 | 81.47 | 67.15 | 68.26 | 69.27 | 67.54 | 84.92 | 96.69 | 78.32 | 63.59 | 85.66 | 90.87 | 68.69 | 35.80 | 77.19 |
| Ours - base | +1.93 | +1.75 | +0.78 | +2.84 | +10.46 | +8.79 | +7.62 | +6.29 | +1.29 | +1.11 | +6.57 | +5.55 | +5.62 | +2.32 | +4.72 | +6.46 | +4.63 |

### Boundary band r=1

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Sky | Person | Rider | Car | Bus | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 40.02 | 41.43 | 43.53 | 36.23 | 37.66 | 31.22 | 32.50 | 26.16 | 45.25 | 47.75 | 41.90 | 34.65 | 37.93 | 38.50 | 39.25 | 24.59 | 37.41 |
| STDC-Seg + Ours | 45.29 | 47.21 | 47.94 | 41.01 | 41.90 | 43.83 | 41.45 | 33.29 | 46.70 | 60.90 | 51.17 | 39.13 | 41.83 | 43.49 | 43.57 | 30.07 | 43.67 |
| Ours - base | +5.27 | +5.78 | +4.40 | +4.78 | +4.25 | +12.61 | +8.95 | +7.13 | +1.45 | +13.16 | +9.27 | +4.48 | +3.90 | +4.99 | +4.31 | +5.48 | +6.26 |

## RUGD (STDC-Seg, n=1)

Only classes with >= 0.1% of val pixels are shown/averaged: asphalt, building, bush, dirt, fence, grass, gravel, log, mulch, picnic-table, rock, sky, tree, vehicle. Excluded rare classes: concrete (0.079%), container (0.015%), person (0.001%), pole (0.083%), sign (0.003%), water (0.064%).

### Full image

| Method | asphalt | building | bush | dirt | fence | grass | gravel | log | mulch | picnic-table | rock | sky | tree | vehicle | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 84.24 | 34.56 | 38.97 | 0.00 | 25.59 | 85.16 | 77.13 | 29.57 | 80.22 | 54.16 | 39.56 | 53.99 | 88.74 | 52.14 | 53.14 |
| STDC-Seg + Ours | 93.21 | 35.70 | 25.53 | 0.00 | 28.56 | 85.33 | 82.36 | 27.36 | 81.01 | 59.25 | 37.17 | 58.99 | 89.25 | 60.96 | 54.62 |
| Ours - base | +8.97 | +1.14 | -13.44 | +0.00 | +2.97 | +0.17 | +5.23 | -2.21 | +0.79 | +5.09 | -2.39 | +5.00 | +0.51 | +8.82 | +1.48 |

### Boundary band r=1

| Method | asphalt | building | bush | dirt | fence | grass | gravel | log | mulch | picnic-table | rock | sky | tree | vehicle | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 29.02 | 17.11 | 21.28 | 0.00 | 17.89 | 37.46 | 36.54 | 16.43 | 36.00 | 28.89 | 20.61 | 16.71 | 50.54 | 24.62 | 25.22 |
| STDC-Seg + Ours | 30.11 | 16.02 | 21.30 | 0.00 | 22.06 | 41.76 | 35.65 | 16.40 | 35.78 | 34.24 | 13.61 | 21.43 | 53.42 | 31.29 | 26.65 |
| Ours - base | +1.09 | -1.09 | +0.02 | +0.00 | +4.17 | +4.30 | -0.89 | -0.03 | -0.23 | +5.35 | -7.00 | +4.72 | +2.88 | +6.67 | +1.43 |

Excluded but notable -- pole (0.083% of val px): full 20.09 -> 28.10, boundary r=1 15.67 -> 27.62.

