# Figure A rows, fig_qualitative_2026-11-14

Generated 2026-11-14 18:02 by `gen_qual_rows.py`. All paths are relative to this folder. The earlier rows in `../fig_qualitative/` are untouched.

## How the rows were chosen

The method improves the overall segmentation, with a specific focus on thin classes, so every dataset gets one row of each:

* **overall**: the largest drop in full-frame error (share of labelled pixels whose predicted class differs from the GT), Ours vs baseline.
* **thin**: the most thin-class pixels fixed: thin pixels newly labelled with the right class minus new false thin pixels, as a share of the frame, so the gain is big enough to see in a full-frame panel. Only images where the per-image thin IoU (thin pixels labelled right / union of GT and predicted thin pixels) rises >= 5 points for STDC-Seg and >= 2 for HRNet-W48, and GT thin pixels cover >= 0.5% of the frame.
* Score = STDC-Seg gain + 0.5 x HRNet-W48 gain (STDC-Seg is the main comparison; the HRNet term makes both column pairs show the effect).
* Every candidate, both kinds: the error drops AND thin IoU does not fall for both networks, and STDC-Seg's thin IoU goes up. So an "overall" row also improves thin classes, and a "thin" row also improves the whole frame.
* Rows of one dataset come from different cities / SYNTHIA frame ranges, or RUGD frames >= 50 apart.
* Thin classes: Cityscapes pole, tlight, tsign, rider, moto, bicycle; SYNTHIA pole, tlight, tsign, rider, moto, bicycle; RUGD pole, fence. (The dataset's thin classes that also gained in the per-class table, bnd r1 >= +1.0 and full >= 0.)

## Rows

| Row | Dataset | Kind | Image | STDC-Seg error | STDC-Seg thin IoU | HRNet-W48 error | HRNet-W48 thin IoU | Chosen | Folder |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cityscapes | overall | `frankfurt_000001_066092` | 14.6% -> 7.5% | 52.4 -> 60.6 | 15.9% -> 13.1% | 72.4 -> 76.0 | rank 1 of 259 | `row1_cityscapes_overall__frankfurt_000001_066092/` |
| 2 | Cityscapes | thin | `lindau_000053_000019` | 3.3% -> 2.3% | 69.6 -> 83.3 | 3.5% -> 2.9% | 69.9 -> 73.8 | rank 2 of 118 | `row2_cityscapes_thin__lindau_000053_000019/` |
| 3 | SYNTHIA | overall | `0002716` | 9.6% -> 6.9% | 61.9 -> 70.2 | 6.5% -> 5.1% | 81.1 -> 85.1 | rank 1 of 900 | `row3_synthia_overall__0002716/` |
| 4 | SYNTHIA | thin | `0000815` | 8.2% -> 6.3% | 45.0 -> 66.7 | 4.1% -> 3.3% | 68.5 -> 74.5 | rank 1 of 731 | `row4_synthia_thin__0000815/` |
| 5 | RUGD | overall | `park-8_00141` | 31.5% -> 4.7% | 9.6 -> 26.9 | 4.3% -> 2.8% | 21.0 -> 38.0 | rank 1 of 123 | `row5_rugd_overall__park-8_00141/` |
| 6 | RUGD | thin | `park-8_01161` | 15.0% -> 11.1% | 11.5 -> 39.2 | 11.2% -> 10.3% | 11.1 -> 17.3 | rank 1 of 76 | `row6_rugd_thin__park-8_01161/` |

Error: lower is better. Thin IoU: higher is better. Baseline -> Ours.

## Alternatives (best first; swap one in with `--pick <dataset>:<kind>:<image>`)

* Cityscapes overall: `frankfurt_000001_066092`, `frankfurt_000001_078803`, `frankfurt_000001_012738`, `frankfurt_000001_023769`, `frankfurt_000001_004736`, `frankfurt_000001_068208`, `munster_000139_000019`, `lindau_000050_000019`, `lindau_000008_000019`, `frankfurt_000000_008206`
* Cityscapes thin: `frankfurt_000001_011162`, `lindau_000053_000019`, `frankfurt_000001_007973`, `frankfurt_000000_012868`, `frankfurt_000001_070099`, `frankfurt_000000_005898`, `frankfurt_000001_038245`, `frankfurt_000001_078803`, `frankfurt_000001_043564`, `frankfurt_000001_038645`
* SYNTHIA overall: `0002716`, `0008124`, `0003591`, `0006154`, `0004545`, `0007637`, `0004729`, `0008073`, `0008929`, `0002993`
* SYNTHIA thin: `0000815`, `0006158`, `0006189`, `0007637`, `0006185`, `0000252`, `0008877`, `0005007`, `0007997`, `0006154`
* RUGD overall: `park-8_00141`, `park-8_00151`, `park-8_00146`, `park-8_00136`, `park-8_00126`, `park-8_00301`, `park-8_00156`, `park-8_00131`, `park-8_00121`, `park-8_00251`
* RUGD thin: `park-8_01161`, `park-8_00901`, `park-8_00916`, `park-8_00446`, `park-8_00946`, `park-8_00431`, `park-8_00441`, `park-8_00841`, `park-8_00426`, `park-8_00906`

## Files

| File | What it is |
|---|---|
| `PREVIEW_qualitative.png` | all rows assembled, reference only (the figure is built in PowerPoint) |
| `row1_cityscapes_overall__frankfurt_000001_066092/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row2_cityscapes_thin__lindau_000053_000019/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row3_synthia_overall__0002716/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row4_synthia_thin__0000815/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row5_rugd_overall__park-8_00141/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row6_rugd_thin__park-8_01161/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `legend_cityscapes_synthia.png` | colour legend |
| `legend_rugd.png` | colour legend |
| `candidates/candidates_<dataset>_<kind>.png` | top 10 alternatives with their numbers |
| `ranking_<dataset>.csv` | every val image: errors, thin IoU and both scores, sorted by overall score |

Columns of every row folder: 1 RGB, 2 GT (black = ignore), 3 STDC-Seg, 4 STDC-Seg + Ours, 5 HRNet-W48, 6 HRNet-W48 + Ours. Predictions are black where the GT is ignore.
