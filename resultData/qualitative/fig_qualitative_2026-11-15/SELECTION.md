# Figure A rows, fig_qualitative_2026-11-15

Generated 2026-11-15 10:37 by `gen_qual_rows.py`. All paths are relative to this folder. The earlier rows in `../fig_qualitative/` are untouched.

## How the rows were chosen

The method improves the overall segmentation, with a specific focus on thin classes, so every dataset gets one row of each:

* **overall**: the most pixels fixed (baseline wrong, Ours right) minus 2 x the pixels broken (baseline right, Ours wrong), as a share of the frame -- a broken pixel counts double, so rows change a lot where the baseline is wrong and little where it is right.
* **thin**: the most thin-class pixels fixed: thin pixels newly labelled with the right class minus new false thin pixels, as a share of the frame, so the gain is big enough to see in a full-frame panel. Only images where the per-image thin IoU (thin pixels labelled right / union of GT and predicted thin pixels) rises >= 5 points for STDC-Seg and >= 2 for HRNet-W48, and GT thin pixels cover >= 0.5% of the frame.
* Score = STDC-Seg gain + 0.5 x HRNet-W48 gain, the HRNet term capped at the STDC-Seg one (STDC-Seg is the main comparison; the HRNet term makes both column pairs show the effect).
* Clean images rank first. Clean: STDC-Seg breaks <= 0.3 x the pixels it fixes over the whole frame, and for both networks, inside the GT boundary band (<= 3 px from a GT class edge) Ours breaks <= 0.45 x the pixels it fixes, and no connected broken patch is larger than 0.2% of the frame -- so no area visibly gets worse. `extra/` shows fixed (green) / broken (red) per network.
* Every candidate, both kinds: fixed > 2 x broken AND thin IoU does not fall for both networks, and STDC-Seg's thin IoU goes up. So an "overall" row also improves thin classes, and a "thin" row also improves the whole frame.
* Rows of one dataset come from different cities / SYNTHIA frame ranges, or RUGD frames >= 50 apart.
* Thin classes: Cityscapes pole, tlight, tsign, rider, moto, bicycle; SYNTHIA pole, tlight, tsign, rider, moto, bicycle; RUGD pole, fence. (The dataset's thin classes that also gained in the per-class table, bnd r1 >= +1.0 and full >= 0.)

## Rows

| Row | Dataset | Kind | Image | STDC-Seg error | STDC-Seg thin IoU | HRNet-W48 error | HRNet-W48 thin IoU | Fixed / broken (STDC-Seg, HRNet-W48) | Clean | Chosen | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cityscapes | overall | `frankfurt_000001_066092` | 14.6% -> 7.5% | 52.4 -> 60.6 | 15.9% -> 13.1% | 72.4 -> 76.0 | 9.5% / 2.4%, 5.2% / 2.4% | no | picked by hand | `row1_cityscapes_overall__frankfurt_000001_066092/` |
| 2 | Cityscapes | thin | `lindau_000053_000019` | 3.3% -> 2.3% | 69.6 -> 83.3 | 3.5% -> 2.9% | 69.9 -> 73.8 | 1.8% / 0.8%, 1.1% / 0.5% | no | picked by hand | `row2_cityscapes_thin__lindau_000053_000019/` |
| 3 | SYNTHIA | overall | `0006020` | 5.4% -> 3.4% | 63.3 -> 71.8 | 1.2% -> 0.9% | 87.8 -> 89.7 | 2.5% / 0.5%, 0.5% / 0.2% | yes | rank 1 of 248 | `row3_synthia_overall__0006020/` |
| 4 | SYNTHIA | thin | `0004734` | 4.5% -> 2.7% | 49.8 -> 60.6 | 1.2% -> 0.9% | 76.3 -> 80.9 | 2.2% / 0.4%, 0.5% / 0.2% | yes | rank 1 of 210 | `row4_synthia_thin__0004734/` |
| 5 | RUGD | overall | `park-8_00141` | 31.5% -> 4.7% | 9.6 -> 26.9 | 4.3% -> 2.8% | 21.0 -> 38.0 | 27.7% / 0.9%, 2.0% / 0.5% | no | picked by hand | `row5_rugd_overall__park-8_00141/` |
| 6 | RUGD | thin | `park-8_01161` | 15.0% -> 11.1% | 11.5 -> 39.2 | 11.2% -> 10.3% | 11.1 -> 17.3 | 5.8% / 2.0%, 2.1% / 1.2% | no | picked by hand | `row6_rugd_thin__park-8_01161/` |

Error: lower is better. Thin IoU: higher is better. Baseline -> Ours. Fixed / broken: share of the frame Ours fixed / broke.

## Alternatives (best first; swap one in with `--pick <dataset>:<kind>:<image>`)

* Cityscapes overall: `frankfurt_000001_066092`, `frankfurt_000001_004736`, `munster_000139_000019`, `lindau_000050_000019`, `frankfurt_000001_070099`, `lindau_000053_000019`, `frankfurt_000001_025921`, `frankfurt_000001_049698`, `munster_000060_000019`, `frankfurt_000001_058914`
* Cityscapes thin: `lindau_000053_000019`, `frankfurt_000001_070099`, `frankfurt_000001_025921`, `frankfurt_000001_073464`, `frankfurt_000001_004736`, `frankfurt_000001_058914`, `frankfurt_000001_066092`, `munster_000060_000019`, `munster_000033_000019`
* SYNTHIA overall: `0006020`, `0004336`, `0004734`, `0003755`, `0006082`, `0004642`, `0004600`, `0004404`, `0004039`, `0004614`
* SYNTHIA thin: `0004734`, `0004039`, `0003925`, `0004336`, `0006082`, `0003755`, `0000005`, `0004670`, `0005680`, `0004404`
* RUGD overall: `park-8_00306`, `park-8_00311`, `park-8_00141`, `park-8_00151`, `park-8_00146`, `park-8_00136`, `park-8_00126`, `park-8_00156`, `park-8_00131`, `park-8_00121`
* RUGD thin: `park-8_00311`, `park-8_00306`, `park-8_00446`, `park-8_00431`, `park-8_00841`, `park-8_00406`, `park-8_00411`, `park-8_00091`, `park-8_00056`, `park-8_00126`

## Files

| File | What it is |
|---|---|
| `PREVIEW_qualitative.png` | all rows assembled, reference only (the figure is built in PowerPoint) |
| `row1_cityscapes_overall__frankfurt_000001_066092/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row2_cityscapes_thin__lindau_000053_000019/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row3_synthia_overall__0006020/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row4_synthia_thin__0004734/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row5_rugd_overall__park-8_00141/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row6_rugd_thin__park-8_01161/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `legend_cityscapes_synthia.png` | colour legend |
| `legend_rugd.png` | colour legend |
| `candidates/candidates_<dataset>_<kind>.png` | top 10 alternatives with their numbers |
| `ranking_<dataset>.csv` | every val image: errors, thin IoU and both scores, sorted by overall score |

Columns of every row folder: 1 RGB, 2 GT (black = ignore), 3 STDC-Seg, 4 STDC-Seg + Ours, 5 HRNet-W48, 6 HRNet-W48 + Ours. Predictions are black where the GT is ignore.
