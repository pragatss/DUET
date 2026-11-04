# Per-class digest -- per-class IoU (%) for the paper tables

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

Per-class table layout follows a standard "Method x class + mIoU" table. "Delta" rows are Ours minus baseline. For Cityscapes STDC, the "run range" row is the spread of that class across the 2 fresh-clone baseline runs (Baseline, Baseline2); a delta marked * exceeds it (i.e. is larger than run-to-run noise). Classes absent from a val set are omitted.

## Cityscapes

### Full image

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg (2-run mean) | 98.10 | 84.74 | 91.85 | 55.78 | 59.50 | 56.72 | 65.91 | 74.73 | 91.64 | 63.66 | 94.21 | 78.60 | 58.74 | 94.46 | 77.03 | 84.80 | 75.32 | 55.85 | 73.93 | 75.56 |
| STDC-Seg + Ours | 98.28 | 85.96 | 92.27 | 52.20 | 60.02 | 63.84 | 69.57 | 77.44 | 92.23 | 66.50 | 94.78 | 80.81 | 60.08 | 95.00 | 79.20 | 84.33 | 77.50 | 61.90 | 75.83 | 77.25 |
| HRNet-W48 | 98.35 | 86.17 | 92.73 | 55.08 | 61.53 | 67.35 | 72.58 | 79.91 | 92.81 | 65.28 | 94.53 | 82.02 | 61.15 | 95.23 | 77.85 | 77.40 | 38.72 | 61.04 | 77.06 | 75.62 |
| HRNet-W48 + Ours | 98.34 | 86.23 | 93.26 | 61.99 | 62.93 | 70.38 | 74.95 | 82.11 | 93.07 | 66.11 | 94.58 | 84.17 | 64.12 | 95.70 | 75.87 | 80.26 | 38.02 | 66.24 | 78.89 | 77.22 |

### Full image -- deltas (points)

| Row | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg: Ours - base | +0.19* | +1.22* | +0.42* | -3.58* | +0.52 | +7.11* | +3.66* | +2.70* | +0.59* | +2.84* | +0.57* | +2.20* | +1.34* | +0.55* | +2.18* | -0.47 | +2.17 | +6.05 | +1.90* | +1.69 |
| STDC-Seg run range (2 runs) | 0.02 | 0.10 | 0.07 | 3.11 | 1.38 | 0.56 | 0.31 | 0.20 | 0.04 | 0.58 | 0.06 | 0.13 | 0.98 | 0.11 | 0.05 | 3.61 | 5.83 | 6.11 | 0.30 |  |
| HRNet-W48: Ours - base | -0.01 | +0.06 | +0.53 | +6.91 | +1.41 | +3.03 | +2.37 | +2.20 | +0.26 | +0.83 | +0.05 | +2.15 | +2.97 | +0.47 | -1.97 | +2.86 | -0.70 | +5.20 | +1.84 | +1.60 |

### Boundary band r=1

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg (2-run mean) | 46.93 | 41.53 | 41.33 | 21.02 | 25.10 | 33.32 | 33.73 | 36.68 | 44.08 | 30.41 | 47.38 | 40.61 | 32.64 | 48.37 | 30.25 | 37.60 | 25.91 | 25.75 | 37.38 | 35.79 |
| STDC-Seg + Ours | 49.39 | 45.56 | 44.88 | 23.50 | 26.52 | 44.83 | 39.21 | 43.89 | 47.44 | 33.92 | 52.23 | 45.16 | 34.27 | 52.13 | 32.35 | 39.45 | 31.33 | 28.66 | 40.88 | 39.77 |
| HRNet-W48 | 47.72 | 48.13 | 46.00 | 23.40 | 27.11 | 49.00 | 42.77 | 48.12 | 48.66 | 33.88 | 54.52 | 48.29 | 37.87 | 54.03 | 35.18 | 39.28 | 20.31 | 30.41 | 42.02 | 40.88 |
| HRNet-W48 + Ours | 49.32 | 49.22 | 48.76 | 27.93 | 30.53 | 52.85 | 46.38 | 51.63 | 50.08 | 35.80 | 56.80 | 51.31 | 41.74 | 54.57 | 35.05 | 42.73 | 17.80 | 34.54 | 44.53 | 43.24 |
| Baseline (single run) | 46.84 | 41.99 | 41.42 | 20.74 | 25.80 | 33.49 | 33.72 | 36.70 | 44.11 | 30.48 | 47.41 | 40.58 | 32.40 | 48.53 | 30.24 | 38.52 | 26.01 | 26.43 | 37.69 | 35.95 |
| Baseline2 (single run) | 47.02 | 41.06 | 41.24 | 21.31 | 24.40 | 33.14 | 33.74 | 36.67 | 44.06 | 30.34 | 47.34 | 40.64 | 32.87 | 48.22 | 30.27 | 36.69 | 25.80 | 25.07 | 37.07 | 35.63 |

### Boundary band r=1 -- deltas (points)

| Row | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg: Ours - base | +2.46* | +4.03* | +3.55* | +2.48* | +1.42* | +11.52* | +5.48* | +7.21* | +3.36* | +3.51* | +4.86* | +4.56* | +1.64* | +3.75* | +2.09* | +1.84* | +5.43* | +2.91* | +3.50* | +3.98 |
| STDC-Seg run range (2 runs) | 0.18 | 0.94 | 0.18 | 0.57 | 1.41 | 0.36 | 0.02 | 0.03 | 0.05 | 0.14 | 0.07 | 0.06 | 0.48 | 0.31 | 0.02 | 1.83 | 0.21 | 1.36 | 0.61 |  |
| HRNet-W48: Ours - base | +1.60 | +1.09 | +2.76 | +4.54 | +3.42 | +3.85 | +3.60 | +3.51 | +1.42 | +1.92 | +2.28 | +3.02 | +3.87 | +0.54 | -0.13 | +3.46 | -2.51 | +4.13 | +2.51 | +2.36 |

### Boundary band r=3 -- deltas (points)

| Row | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Terrain | Sky | Person | Rider | Car | Truck | Bus | Train | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg: Ours - base | +3.36* | +4.63* | +5.07* | +2.84* | +2.15* | +12.74* | +6.93* | +8.25* | +4.65* | +4.17* | +5.93* | +5.68* | +2.94* | +4.07* | +3.12* | +2.15* | +7.00* | +3.57* | +4.47* | +4.93 |
| STDC-Seg run range (2 runs) | 0.01 | 0.66 | 0.30 | 0.98 | 1.35 | 0.58 | 0.29 | 0.27 | 0.21 | 0.37 | 0.01 | 0.11 | 0.12 | 0.34 | 0.37 | 1.95 | 1.76 | 1.64 | 0.65 |  |
| HRNet-W48: Ours - base | +1.86 | +1.70 | +2.94 | +5.32 | +3.64 | +3.97 | +4.05 | +4.17 | +1.68 | +2.23 | +1.51 | +3.82 | +4.45 | +1.58 | +0.87 | +4.51 | -2.67 | +5.53 | +3.09 | +2.85 |

Summary: STDC-Seg + Ours improves 19/19 classes at boundary r=1 (largest: Pole +11.5, Sign +7.2, Tr.Light +5.5). HRNet-W48 + Ours improves 17/19 classes at boundary r=1 (largest: Wall +4.5, M.bike +4.1, Rider +3.9); decreases: Truck -0.1, Train -2.5. Rare classes (truck, bus, train, motorcycle) swing several points between the baseline runs; make no per-class claims about them.

## SYNTHIA (STDC-Seg and HRNet-W48, n=1)

SYNTHIA labels use Cityscapes classes; terrain, truck and train do not occur in val.

### Full image

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Sky | Person | Rider | Car | Bus | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 89.60 | 88.69 | 94.06 | 78.63 | 56.70 | 59.47 | 61.64 | 61.25 | 83.63 | 95.58 | 71.75 | 58.04 | 80.04 | 88.54 | 63.97 | 29.34 | 72.56 |
| STDC-Seg + Ours | 91.54 | 90.43 | 94.84 | 81.47 | 67.15 | 68.26 | 69.27 | 67.54 | 84.92 | 96.69 | 78.32 | 63.59 | 85.66 | 90.87 | 68.69 | 35.80 | 77.19 |
| HRNet-W48 | 96.95 | 95.92 | 96.37 | 90.22 | 86.33 | 78.87 | 76.66 | 78.54 | 86.27 | 96.90 | 88.24 | 77.70 | 95.37 | 94.83 | 82.56 | 51.37 | 85.82 |
| HRNet-W48 + Ours | 97.58 | 96.75 | 96.81 | 93.35 | 89.35 | 83.48 | 82.40 | 84.01 | 87.37 | 97.18 | 90.94 | 81.40 | 95.93 | 95.18 | 85.50 | 58.20 | 88.46 |
| STDC-Seg: Ours - base | +1.93 | +1.75 | +0.78 | +2.84 | +10.46 | +8.79 | +7.62 | +6.29 | +1.29 | +1.11 | +6.57 | +5.55 | +5.62 | +2.32 | +4.72 | +6.46 | +4.63 |
| HRNet-W48: Ours - base | +0.63 | +0.84 | +0.44 | +3.14 | +3.03 | +4.61 | +5.74 | +5.47 | +1.10 | +0.28 | +2.70 | +3.70 | +0.56 | +0.35 | +2.94 | +6.84 | +2.65 |

### Boundary band r=1

| Method | Road | S.walk | Build. | Wall | Fence | Pole | Tr.Light | Sign | Veget. | Sky | Person | Rider | Car | Bus | M.bike | Bike | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 40.02 | 41.43 | 43.53 | 36.23 | 37.66 | 31.22 | 32.50 | 26.16 | 45.25 | 47.75 | 41.90 | 34.65 | 37.93 | 38.50 | 39.25 | 24.59 | 37.41 |
| STDC-Seg + Ours | 45.29 | 47.21 | 47.94 | 41.01 | 41.90 | 43.83 | 41.45 | 33.29 | 46.70 | 60.90 | 51.17 | 39.13 | 41.83 | 43.49 | 43.57 | 30.07 | 43.67 |
| HRNet-W48 | 67.58 | 68.08 | 59.08 | 63.64 | 67.56 | 61.87 | 53.22 | 53.14 | 50.77 | 69.26 | 68.98 | 56.63 | 66.62 | 62.49 | 60.88 | 44.59 | 60.90 |
| HRNet-W48 + Ours | 72.62 | 73.38 | 63.02 | 70.31 | 74.08 | 70.33 | 62.03 | 60.78 | 53.25 | 72.69 | 75.57 | 63.16 | 71.15 | 67.47 | 66.82 | 52.19 | 66.80 |
| STDC-Seg: Ours - base | +5.27 | +5.78 | +4.40 | +4.78 | +4.25 | +12.61 | +8.95 | +7.13 | +1.45 | +13.16 | +9.27 | +4.48 | +3.90 | +4.99 | +4.31 | +5.48 | +6.26 |
| HRNet-W48: Ours - base | +5.05 | +5.30 | +3.94 | +6.67 | +6.52 | +8.47 | +8.81 | +7.64 | +2.47 | +3.43 | +6.59 | +6.53 | +4.52 | +4.98 | +5.94 | +7.60 | +5.90 |

## RUGD (STDC-Seg and HRNet-W48, n=1)

Only classes with >= 0.1% of val pixels are shown/averaged: asphalt, building, bush, dirt, fence, grass, gravel, log, mulch, picnic-table, rock, sky, tree, vehicle. Excluded rare classes: concrete (0.079%), container (0.015%), person (0.001%), pole (0.083%), sign (0.003%), water (0.064%).

### Full image

| Method | asphalt | building | bush | dirt | fence | grass | gravel | log | mulch | picnic-table | rock | sky | tree | vehicle | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 84.24 | 34.56 | 38.97 | 0.00 | 25.59 | 85.16 | 77.13 | 29.57 | 80.22 | 54.16 | 39.56 | 53.99 | 88.74 | 52.14 | 53.14 |
| STDC-Seg + Ours | 93.21 | 35.70 | 25.53 | 0.00 | 28.56 | 85.33 | 82.36 | 27.36 | 81.01 | 59.25 | 37.17 | 58.99 | 89.25 | 60.96 | 54.62 |
| HRNet-W48 | 90.23 | 38.23 | 29.83 | 0.00 | 25.10 | 85.18 | 80.60 | 37.92 | 82.46 | 69.47 | 55.53 | 71.88 | 91.67 | 65.32 | 58.82 |
| HRNet-W48 + Ours | 91.68 | 36.39 | 33.12 | 0.00 | 26.58 | 85.44 | 82.49 | 37.17 | 81.37 | 72.41 | 54.38 | 76.58 | 92.23 | 70.10 | 60.00 |
| STDC-Seg: Ours - base | +8.97 | +1.14 | -13.44 | +0.00 | +2.97 | +0.17 | +5.23 | -2.21 | +0.79 | +5.09 | -2.39 | +5.00 | +0.51 | +8.82 | +1.48 |
| HRNet-W48: Ours - base | +1.45 | -1.83 | +3.29 | +0.00 | +1.47 | +0.25 | +1.89 | -0.75 | -1.09 | +2.93 | -1.15 | +4.71 | +0.56 | +4.78 | +1.18 |

### Boundary band r=1

| Method | asphalt | building | bush | dirt | fence | grass | gravel | log | mulch | picnic-table | rock | sky | tree | vehicle | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 29.02 | 17.11 | 21.28 | 0.00 | 17.89 | 37.46 | 36.54 | 16.43 | 36.00 | 28.89 | 20.61 | 16.71 | 50.54 | 24.62 | 25.22 |
| STDC-Seg + Ours | 30.11 | 16.02 | 21.30 | 0.00 | 22.06 | 41.76 | 35.65 | 16.40 | 35.78 | 34.24 | 13.61 | 21.43 | 53.42 | 31.29 | 26.65 |
| HRNet-W48 | 31.92 | 22.63 | 21.60 | 0.00 | 18.28 | 43.68 | 38.40 | 19.90 | 35.28 | 44.66 | 27.62 | 48.90 | 59.33 | 35.13 | 31.95 |
| HRNet-W48 + Ours | 32.32 | 21.94 | 23.93 | 0.00 | 22.08 | 44.38 | 38.62 | 27.08 | 37.88 | 49.30 | 29.44 | 64.22 | 66.16 | 41.78 | 35.65 |
| STDC-Seg: Ours - base | +1.09 | -1.09 | +0.02 | +0.00 | +4.17 | +4.30 | -0.89 | -0.03 | -0.23 | +5.35 | -7.00 | +4.72 | +2.88 | +6.67 | +1.43 |
| HRNet-W48: Ours - base | +0.40 | -0.69 | +2.32 | +0.00 | +3.80 | +0.70 | +0.22 | +7.18 | +2.61 | +4.64 | +1.82 | +15.32 | +6.83 | +6.65 | +3.70 |

Excluded but notable, STDC-Seg -- pole (0.083% of val px): full 20.09 -> 28.10, boundary r=1 15.67 -> 27.62.

Excluded but notable, HRNet-W48 -- pole (0.083% of val px): full 39.21 -> 43.55, boundary r=1 34.61 -> 43.28.

