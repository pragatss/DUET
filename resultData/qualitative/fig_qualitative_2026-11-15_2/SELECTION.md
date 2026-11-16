# Figure A rows, fig_qualitative_2026-11-15_2

Generated 2026-11-15 18:09 by `gen_qual_rows.py`. All paths are relative to this folder. The earlier rows in `../fig_qualitative/` are untouched.

## How the rows were chosen

The method improves the overall segmentation, with a specific focus on thin classes, so every dataset gets one row of each:

* **overall**: the most visible pixels fixed (baseline wrong, Ours right) minus 2 x the visible pixels broken (baseline right, Ours wrong), as a share of the frame -- a broken pixel counts double, so rows change a lot where the baseline is wrong and little where it is right. Visible = inside an area big enough to see in a figure panel (survives a morphological opening with a kernel of radius image width / 800 px; 1-2 px slivers along edges drop out).
* **thin**: the most thin-class pixels fixed: thin pixels newly labelled with the right class minus new false thin pixels, as a share of the frame, so the gain is big enough to see in a full-frame panel. Only images where the per-image thin IoU (thin pixels labelled right / union of GT and predicted thin pixels) rises >= 5 points for STDC-Seg and >= 2 for HRNet-W48, and GT thin pixels cover >= 0.5% of the frame.
* Score = STDC-Seg gain + 0.5 x HRNet-W48 gain, the HRNet term capped at the STDC-Seg one (STDC-Seg is the main comparison; the HRNet term makes both column pairs show the effect).
* Clean images rank first. Clean: visible broken <= 0.20 x visible fixed for STDC-Seg and <= 0.35 x for HRNet-W48 -- Ours fixes wrong areas without visibly hurting right ones. `extra/` shows fixed (green) / broken (red) per network.
* Every candidate, both kinds: the error drops, visible fixed > 2 x visible broken AND thin IoU does not fall for both networks, and STDC-Seg's thin IoU goes up. So an "overall" row also improves thin classes, and a "thin" row also improves the whole frame.
* Rows picked by eye (`--pick`) were chosen from the top clean candidates after checking their `extra/` maps; the Chosen column gives their rank. Without `--pick`, rows of one dataset come from different cities / SYNTHIA frame ranges, or RUGD frames >= 50 apart.
* Thin classes: Cityscapes pole, tlight, tsign, rider, moto, bicycle; SYNTHIA pole, tlight, tsign, rider, moto, bicycle; RUGD pole, fence. (The dataset's thin classes that also gained in the per-class table, bnd r1 >= +1.0 and full >= 0.)

## Rows

| Row | Dataset | Kind | Image | STDC-Seg error | STDC-Seg thin IoU | HRNet-W48 error | HRNet-W48 thin IoU | Fixed / broken (STDC-Seg, HRNet-W48) | Clean | Chosen | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cityscapes | overall | `frankfurt_000001_004736` | 7.7% -> 4.9% | 43.6 -> 54.8 | 4.5% -> 3.6% | 62.3 -> 65.3 | 3.5% / 0.8%, 1.5% / 0.6% | yes | rank 2 of 56, picked by eye | `row1_cityscapes_overall__frankfurt_000001_004736/` |
| 2 | Cityscapes | thin | `frankfurt_000001_070099` | 4.5% -> 3.4% | 57.9 -> 69.4 | 3.1% -> 2.6% | 72.8 -> 78.1 | 1.8% / 0.7%, 0.9% / 0.4% | yes | rank 1 of 31, picked by eye | `row2_cityscapes_thin__frankfurt_000001_070099/` |
| 3 | SYNTHIA | overall | `0008036` | 5.4% -> 3.4% | 45.3 -> 60.3 | 2.4% -> 2.0% | 79.4 -> 83.0 | 2.6% / 0.6%, 0.9% / 0.4% | yes | rank 3 of 370, picked by eye | `row3_synthia_overall__0008036/` |
| 4 | SYNTHIA | thin | `0008977` | 7.4% -> 5.4% | 40.5 -> 51.4 | 3.1% -> 2.4% | 69.9 -> 77.5 | 3.0% / 1.0%, 1.2% / 0.5% | yes | rank 8 of 313, picked by eye | `row4_synthia_thin__0008977/` |
| 5 | RUGD | overall | `park-8_00126` | 15.1% -> 3.7% | 13.6 -> 31.9 | 4.5% -> 2.3% | 27.6 -> 52.0 | 12.2% / 0.8%, 2.7% / 0.4% | yes | rank 5 of 53, picked by eye | `row5_rugd_overall__park-8_00126/` |
| 6 | RUGD | thin | `park-8_00446` | 5.4% -> 4.0% | 19.6 -> 69.5 | 3.3% -> 2.5% | 37.8 -> 65.6 | 2.0% / 0.6%, 1.4% / 0.5% | yes | rank 1 of 38, picked by eye | `row6_rugd_thin__park-8_00446/` |

Error: lower is better. Thin IoU: higher is better. Baseline -> Ours. Fixed / broken: share of the frame Ours fixed / broke.

## Alternatives (best first; swap one in with `--pick <dataset>:<kind>:<image>`)

* Cityscapes overall: `munster_000139_000019`, `frankfurt_000001_004736`, `munster_000164_000019`, `frankfurt_000001_070099`, `munster_000015_000019`, `munster_000060_000019`, `frankfurt_000001_083029`, `frankfurt_000000_009969`, `frankfurt_000001_073464`, `frankfurt_000001_032711`
* Cityscapes thin: `frankfurt_000001_070099`, `frankfurt_000001_083029`, `frankfurt_000000_009969`, `frankfurt_000001_073464`, `frankfurt_000001_004736`, `munster_000164_000019`, `munster_000060_000019`, `frankfurt_000001_032711`, `lindau_000053_000019`, `frankfurt_000000_012868`
* SYNTHIA overall: `0001164`, `0002717`, `0008036`, `0005007`, `0002742`, `0008261`, `0001219`, `0003724`, `0006082`, `0008977`
* SYNTHIA thin: `0007637`, `0000252`, `0005007`, `0007647`, `0007631`, `0009126`, `0008261`, `0008977`, `0008264`, `0001717`
* RUGD overall: `park-8_00141`, `park-8_00151`, `park-8_00146`, `park-8_00136`, `park-8_00126`, `park-8_00156`, `park-8_00131`, `park-8_00121`, `park-8_00171`, `park-8_00251`
* RUGD thin: `park-8_00446`, `park-8_00311`, `park-8_00746`, `park-8_00126`, `park-8_00306`, `park-8_00096`, `park-8_00161`, `park-8_00176`, `park-8_00141`, `park-8_00121`

## Files

| File | What it is |
|---|---|
| `PREVIEW_qualitative.png` | all rows assembled, reference only (the figure is built in PowerPoint) |
| `row1_cityscapes_overall__frankfurt_000001_004736/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row2_cityscapes_thin__frankfurt_000001_070099/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row3_synthia_overall__0008036/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row4_synthia_thin__0008977/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row5_rugd_overall__park-8_00126/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row6_rugd_thin__park-8_00446/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `legend_cityscapes_synthia.png` | colour legend |
| `legend_rugd.png` | colour legend |
| `candidates/candidates_<dataset>_<kind>.png` | top 10 alternatives with their numbers |
| `ranking_<dataset>.csv` | every val image: errors, thin IoU and both scores, sorted by overall score |

Columns of every row folder: 1 RGB, 2 GT (black = ignore), 3 STDC-Seg, 4 STDC-Seg + Ours, 5 HRNet-W48, 6 HRNet-W48 + Ours. Predictions are black where the GT is ignore.
