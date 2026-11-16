# Figure A rows, fig_qualitative_2026-11-16

Generated 2026-11-16 09:42 by `gen_qual_rows.py`. All paths are relative to this folder. The earlier rows in `../fig_qualitative/` are untouched.

## How the rows were chosen

The method improves the overall segmentation, with a specific focus on thin classes, so every dataset gets one row of each:

* **Class-weighted IoU gain** (per image, per network): per-class IoU, Ours minus baseline, for every class whose GT covers >= 0.1% of the frame, averaged with weight sqrt(GT class size). Classes absent from the GT are left out (their IoU only jumps between 0 and 1); their false pixels still lower the IoU of the GT class they cover. A class 100x larger counts 10x more, not 100x, so a gain in sky or grass cannot hide a loss in a pole or sign, and a speck cannot outweigh the road. Lost IoU points count 2 x: gain = (sum w max(d, 0) - 2 sum w max(-d, 0)) / sum w.
* **overall**: the largest class-weighted gain over all counted classes.
* **thin**: the largest class-weighted gain over the thin classes; GT thin pixels >= 0.5% of the frame and thin gain >= 5 points (STDC-Seg) / >= 2 (HRNet-W48).
* Every candidate, both kinds and both networks: the error drops, **no counted class loses more than 3 IoU points**, the class-weighted gain is positive and the thin gain is not negative.
* Clean images rank first: Ours also does not visibly hurt an area that was right (a class can gain in one place and lose in another, which per-class IoU cannot see). Visible broken pixels (baseline right, Ours wrong, in an area big enough to see -- survives an opening of radius image width / 800 px) <= 0.20 x the visible fixed pixels for STDC-Seg and <= 0.35 x for HRNet-W48. `extra/` shows fixed (green) / broken (red) per network.
* Score = STDC-Seg gain + 0.5 x HRNet-W48 gain, the HRNet term capped at the STDC-Seg one (STDC-Seg is the main comparison).
* Rows of one dataset come from different cities / SYNTHIA frame ranges, or RUGD frames >= 50 apart, unless only a non-clean image would satisfy that (clean comes first).
* Thin classes: Cityscapes pole, tlight, tsign, rider, moto, bicycle; SYNTHIA pole, tlight, tsign, rider, moto, bicycle; RUGD pole, fence. (The dataset's thin classes that also gained in the per-class table, bnd r1 >= +1.0 and full >= 0.)

## Rows

| Row | Dataset | Kind | Image | Class gain (STDC-Seg / HRNet-W48) | Thin gain | Worst class | STDC-Seg error | HRNet-W48 error | Clean | Chosen | Folder |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Cityscapes | overall | `frankfurt_000001_004736` | +7.0 / +2.3 | +9.6 / +2.8 | -1.7 / -0.8 | 7.7% -> 4.9% | 4.5% -> 3.6% | yes | rank 1 of 96 | `row1_cityscapes_overall__frankfurt_000001_004736/` |
| 2 | Cityscapes | thin | `frankfurt_000001_070099` | +3.1 / +1.3 | +11.5 / +5.3 | -0.7 / -1.5 | 4.5% -> 3.4% | 3.1% -> 2.6% | yes | rank 1 of 46 | `row2_cityscapes_thin__frankfurt_000001_070099/` |
| 3 | SYNTHIA | overall | `0007637` | +5.3 / +1.3 | +11.8 / +2.6 | +0.0 / -3.0 | 9.4% -> 7.0% | 3.7% -> 3.1% | yes | rank 1 of 747 | `row3_synthia_overall__0007637/` |
| 4 | SYNTHIA | thin | `0005462` | +4.1 / +1.5 | +18.8 / +9.9 | +0.0 / +0.0 | 5.1% -> 3.8% | 1.4% -> 1.1% | yes | rank 1 of 636 | `row4_synthia_thin__0005462/` |
| 5 | RUGD | overall | `park-8_00141` | +21.4 / +4.2 | +18.0 / +16.6 | -1.5 / +0.0 | 31.5% -> 4.7% | 4.3% -> 2.8% | yes | rank 1 of 136 | `row5_rugd_overall__park-8_00141/` |
| 6 | RUGD | thin | `park-8_00446` | +4.8 / +4.3 | +51.0 / +27.4 | -0.3 / +0.0 | 5.4% -> 4.0% | 3.3% -> 2.5% | yes | rank 1 of 25 | `row6_rugd_thin__park-8_00446/` |

Gains and worst class in IoU points, Ours minus baseline (worst class: largest per-class loss, 0 if none lost). Error: share of labelled pixels wrong, baseline -> Ours.

### Per-class IoU change, every counted class (Ours minus baseline, IoU points)

* row 1 `frankfurt_000001_004736`
  * STDC-Seg: fence +43.1, wall +23.4, pole +14.2, tsign +9.5, building +8.2, tlight +4.7, person +3.7, sidewalk +3.3, veg +1.5, sky +1.1, road +0.5, car +0.1, bicycle -0.5, terrain -1.7
  * HRNet-W48: fence +7.9, person +6.6, wall +5.1, tsign +4.3, tlight +3.8, terrain +3.4, building +2.7, pole +2.7, sidewalk +2.7, veg +2.0, sky +0.4, road +0.0, bicycle +0.0, car -0.8
* row 2 `frankfurt_000001_070099`
  * STDC-Seg: tsign +12.1, pole +11.7, tlight +10.2, sidewalk +9.0, person +5.2, building +3.0, veg +1.6, car +0.7, road +0.5, sky -0.7
  * HRNet-W48: tsign +7.2, pole +5.5, tlight +2.9, veg +1.5, sidewalk +1.5, sky +1.4, building +1.2, car +0.2, road +0.1, person -1.5
* row 3 `0007637`
  * STDC-Seg: tsign +17.6, tlight +16.3, car +15.1, pole +10.1, person +9.5, rider +9.4, fence +9.0, bicycle +8.5, road +4.0, sidewalk +3.9, veg +3.2, bus +2.1, sky +1.0, building +0.9
  * HRNet-W48: bicycle +7.6, person +5.5, rider +4.6, tlight +4.3, fence +4.2, pole +2.9, veg +1.5, sidewalk +1.4, bus +0.9, road +0.6, car +0.5, building +0.4, sky +0.4, tsign -3.0
* row 4 `0005462`
  * STDC-Seg: car +31.8, pole +18.8, person +14.0, sidewalk +4.3, sky +3.5, road +2.2, veg +1.9, building +0.7
  * HRNet-W48: pole +9.9, car +7.4, person +6.3, sidewalk +1.5, veg +1.1, road +0.9, sky +0.2, building +0.2
* row 5 `park-8_00141`
  * STDC-Seg: asphalt +55.3, fence +23.7, grass +13.4, pole +12.1, concrete +10.4, sky +5.6, tree +1.5, mulch +0.0, vehicle -1.5
  * HRNet-W48: fence +28.2, grass +15.5, mulch +5.2, vehicle +4.4, pole +4.3, sky +2.3, asphalt +2.2, tree +1.3, concrete +0.0
* row 6 `park-8_00446`
  * STDC-Seg: fence +51.0, sky +7.3, mulch +2.6, tree +2.0, asphalt +0.2, vehicle +0.0, grass -0.3
  * HRNet-W48: asphalt +31.9, fence +27.4, vehicle +11.9, sky +4.3, mulch +2.8, tree +1.0, grass +0.6

## Alternatives (best first; swap one in with `--pick <dataset>:<kind>:<image>`)

* Cityscapes overall: `frankfurt_000001_004736`, `munster_000139_000019`, `frankfurt_000001_083029`, `frankfurt_000001_070099`, `frankfurt_000001_073464`, `frankfurt_000001_023769`, `frankfurt_000001_052120`, `frankfurt_000001_040575`, `frankfurt_000001_058914`, `frankfurt_000001_083199`
* Cityscapes thin: `frankfurt_000001_070099`, `frankfurt_000001_004736`, `frankfurt_000001_073464`, `frankfurt_000001_083029`, `frankfurt_000001_033655`, `frankfurt_000001_052120`, `munster_000093_000019`, `frankfurt_000000_000576`, `frankfurt_000001_049078`, `munster_000091_000019`
* SYNTHIA overall: `0007637`, `0007631`, `0004336`, `0004545`, `0002717`, `0008914`, `0008977`, `0007162`, `0004614`, `0000252`
* SYNTHIA thin: `0005462`, `0003651`, `0005494`, `0008036`, `0002717`, `0000895`, `0008914`, `0008386`, `0004851`, `0009208`
* RUGD overall: `park-8_00141`, `park-8_00151`, `park-8_00121`, `trail-5_00931`, `park-8_00746`, `park-8_00446`, `park-8_01716`, `park-8_00096`, `park-8_00176`, `trail-5_01591`
* RUGD thin: `park-8_00446`, `park-8_00746`, `park-8_00121`, `park-8_00141`, `park-8_00151`, `park-8_00096`, `park-8_00176`, `park-8_00411`, `park-8_00731`, `park-8_00071`

## Files

| File | What it is |
|---|---|
| `PREVIEW_qualitative.png` | all rows assembled, reference only (the figure is built in PowerPoint) |
| `row1_cityscapes_overall__frankfurt_000001_004736/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row2_cityscapes_thin__frankfurt_000001_070099/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x1 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row3_synthia_overall__0007637/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row4_synthia_thin__0005462/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x2 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row5_rugd_overall__park-8_00141/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `row6_rugd_thin__park-8_00446/` | row: 1_rgb.png, 2_gt.png, 3_stdc.png, 4_stdc_ours.png, 5_hrnet.png, 6_hrnet_ours.png (x3 enlargement, nearest for label maps); `extra/` = pixels Ours fixed (green) / broke (red), per network, for choosing only |
| `legend_cityscapes_synthia.png` | colour legend |
| `legend_rugd.png` | colour legend |
| `candidates/candidates_<dataset>_<kind>.png` | top 10 alternatives with their numbers |
| `ranking_<dataset>.csv` | every val image: errors, fixed / broken pixels, thin IoU, class-weighted gains, worst class drop and both scores, sorted by overall score |

Columns of every row folder: 1 RGB, 2 GT (black = ignore), 3 STDC-Seg, 4 STDC-Seg + Ours, 5 HRNet-W48, 6 HRNet-W48 + Ours. Predictions are black where the GT is ignore.
