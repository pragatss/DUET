# Per-class IoU (%) -- rugd val

generated 2026-09-27 21:26 by gen_perclass.py

## Full image

| Method | asphalt | bicycle | bridge | building | bush | concrete | container | dirt | fence | grass | gravel | log | mulch | person | picnic-table | pole | rock | rock-bed | sand | sign | sky | tree | vehicle | water | mIoU | Thin | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 84.24 |  |  | 34.56 | 38.97 | 0.41 | 14.48 | 0.00 | 25.59 | 85.16 | 77.13 | 29.57 | 80.22 | 0.00 | 54.16 | 20.09 | 39.56 |  |  | 0.00 | 53.99 | 88.74 | 52.14 | 52.21 | 53.14 | 27.58 |  |
| STDC-Seg + Ours | 93.21 |  |  | 35.70 | 25.53 | 4.55 | 22.53 | 0.00 | 28.56 | 85.33 | 82.36 | 27.36 | 81.01 | 0.00 | 59.25 | 28.10 | 37.17 |  |  | 0.00 | 58.99 | 89.25 | 60.96 | 55.24 | 54.62 | 27.96 | +1.48 |

## Boundary band r=1

| Method | asphalt | bicycle | bridge | building | bush | concrete | container | dirt | fence | grass | gravel | log | mulch | person | picnic-table | pole | rock | rock-bed | sand | sign | sky | tree | vehicle | water | mIoU | Thin | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 29.02 |  |  | 17.11 | 21.28 | 0.28 | 9.17 | 0.00 | 17.89 | 37.46 | 36.54 | 16.43 | 36.00 | 0.00 | 28.89 | 15.67 | 20.61 |  |  | 0.00 | 16.71 | 50.54 | 24.62 | 12.61 | 25.22 | 17.16 |  |
| STDC-Seg + Ours | 30.11 |  |  | 16.02 | 21.30 | 6.96 | 14.22 | 0.00 | 22.06 | 41.76 | 35.65 | 16.40 | 35.78 | 0.00 | 34.24 | 27.62 | 13.61 |  |  | 0.00 | 21.43 | 53.42 | 31.29 | 23.30 | 26.65 | 19.23 | +1.43 |

## Boundary band r=3

| Method | asphalt | bicycle | bridge | building | bush | concrete | container | dirt | fence | grass | gravel | log | mulch | person | picnic-table | pole | rock | rock-bed | sand | sign | sky | tree | vehicle | water | mIoU | Thin | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STDC-Seg | 35.92 |  |  | 21.83 | 23.56 | 0.36 | 12.02 | 0.00 | 20.41 | 43.57 | 40.50 | 20.15 | 40.97 | 0.00 | 38.76 | 20.06 | 23.51 |  |  | 0.00 | 26.50 | 59.64 | 31.86 | 17.14 | 30.51 | 20.28 |  |
| STDC-Seg + Ours | 37.43 |  |  | 21.82 | 23.53 | 7.07 | 20.44 | 0.00 | 25.95 | 48.43 | 41.37 | 20.46 | 41.52 | 0.00 | 46.57 | 32.23 | 17.30 |  |  | 0.00 | 34.86 | 63.29 | 41.24 | 29.05 | 33.13 | 23.21 | +2.62 |

## Ours - baseline (points)

| Host | Region | asphalt | bicycle | bridge | building | bush | concrete | container | dirt | fence | grass | gravel | log | mulch | person | picnic-table | pole | rock | rock-bed | sand | sign | sky | tree | vehicle | water | mIoU | Thin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stdc | full | +8.97 |  |  | +1.14 | -13.44 | +4.14 | +8.04 | +0.00 | +2.97 | +0.17 | +5.23 | -2.21 | +0.79 | +0.00 | +5.09 | +8.01 | -2.39 |  |  | +0.00 | +5.00 | +0.51 | +8.82 | +3.02 | +1.48 | +0.38 |
| stdc | bnd_r1 | +1.09 |  |  | -1.09 | +0.02 | +6.68 | +5.05 | +0.00 | +4.17 | +4.30 | -0.89 | -0.03 | -0.23 | +0.00 | +5.35 | +11.95 | -7.00 |  |  | +0.00 | +4.72 | +2.88 | +6.67 | +10.70 | +1.43 | +2.07 |
| stdc | bnd_r3 | +1.52 |  |  | -0.00 | -0.03 | +6.72 | +8.42 | +0.00 | +5.54 | +4.86 | +0.87 | +0.31 | +0.55 | +0.00 | +7.81 | +12.16 | -6.21 |  |  | +0.00 | +8.36 | +3.65 | +9.38 | +11.91 | +2.62 | +2.92 |

Thin classes: pole, sign, fence, bicycle, log.
Means use only classes with >= 0.1% of val GT pixels: asphalt, building, bush, dirt, fence, grass, gravel, log, mulch, picnic-table, rock, sky, tree, vehicle.
Absent from val (blank, excluded): bicycle, bridge, rock-bed, sand.
