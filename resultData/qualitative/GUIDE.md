# Qualitative figures -- what every file is and how to assemble them

Generated 2026-09-30 19:57 by `gen_qualitative.py`. All paths are relative to `resultData/qualitative/`.
Models: baseline and "Ours" (= HI: RSR + BPM) only. STDC-Seg: I0 / HI1 (Cityscapes), Synthia / Synthia-HI1, RUGD / RUGD-HI1. HRNet-W48: baseline / HI1 (Cityscapes only -- the only dataset HRNet was trained on).

## Folder map

```
fig_qualitative/                  Figure A (cf. ref_images/qualitative/QualitativeData.png)
  PREVIEW_qualitative.png         auto-assembled preview in the recommended two-block layout (reference only)
  legend_cityscapes_synthia.png   colour legend for Cityscapes + SYNTHIA rows (same classes)
  legend_rugd.png                 colour legend for RUGD rows
  rowN_<dataset>__<image>/        one folder per figure row
    1_rgb.png 2_gt.png 3_stdc.png 4_stdc_ours.png [5_hrnet.png 6_hrnet_ours.png]   (nothing else;
    --extras adds boxed/, zoom/ and green-fixed/red-broken diagnostics if you ever want them)
  candidates/candidates_<dataset>.png   top-10 alternative images per dataset, with scores
fig_framework/                    Figure B (cf. ref_images/qualitative/duet_framework.png)
  INDEX_candidates.png            one row per candidate image -- choose here
  candNN__<image>/                everything for one candidate (full/, zoom1/, zoom2/; see Figure B)
manifest.json                     machine-readable version of this guide
```

## Figure A -- qualitative comparison

### Rows (chosen automatically, strategy = `top`)

| Row | Dataset | Image | Folder | Rank / pool | Boundary-band acc. gain (points) |
|---|---|---|---|---|---|
| 1 | Cityscapes | lindau_000021_000019 | `fig_qualitative/row1_cityscapes__lindau_000021_000019/` | 1 / 500 | STDC +3.6, HRNet +1.3 |
| 2 | Cityscapes | munster_000164_000019 | `fig_qualitative/row2_cityscapes__munster_000164_000019/` | 3 / 500 | STDC +4.6, HRNet +2.4 |
| 3 | SYNTHIA | 0003112 | `fig_qualitative/row3_synthia__0003112/` | 1 / 940 | STDC +6.5 |
| 4 | SYNTHIA | 0007665 | `fig_qualitative/row4_synthia__0007665/` | 2 / 940 | STDC +7.1 |
| 5 | RUGD | park-8_00391 | `fig_qualitative/row5_rugd__park-8_00391/` | 1 / 733 | STDC +5.1 |
| 6 | RUGD | trail-5_00606 | `fig_qualitative/row6_rugd__trail-5_00606/` | 57 / 733 | STDC +6.4 |

"Boundary-band acc. gain" = change in the fraction of correctly labelled pixels within 1 px of a GT boundary in that image (Ours minus baseline). Rank 1 = the largest gain in the val set. For Cityscapes the score is the smaller of the STDC and HRNet gains, so both model pairs show a visible difference. One image per city (Cityscapes), video sequence (RUGD) or frame range (SYNTHIA).

### Columns (same file names in every row folder)

| # | File | Column title in the figure | What it is |
|---|---|---|---|
| 1 | 1_rgb.png | RGB | input image |
| 2 | 2_gt.png | GT | ground truth; black = ignore / unlabeled |
| 3 | 3_stdc.png | STDC-Seg | baseline prediction |
| 4 | 4_stdc_ours.png | STDC-Seg + Ours | RSR + BPM prediction |
| 5 | 5_hrnet.png | HRNet-W48 | baseline prediction (Cityscapes rows only) |
| 6 | 6_hrnet_ours.png | HRNet-W48 + Ours | RSR + BPM prediction (Cityscapes rows only) |

Predictions are masked to black wherever the GT is ignore, so they read like the GT. Colours: Cityscapes palette for Cityscapes and SYNTHIA (SYNTHIA uses Cityscapes classes), RUGD's own palette for RUGD.

### How to assemble it in PowerPoint

1. Open `fig_qualitative/PREVIEW_qualitative.png` to see the target layout.
2. HRNet exists for Cityscapes only, so SYNTHIA/RUGD rows have no HRNet panels. Recommended layout (the preview): **two blocks**. Top block = the Cityscapes rows x 6 columns (RGB, GT, STDC-Seg, STDC-Seg + Ours, HRNet-W48, HRNet-W48 + Ours). Dashed separator (like the reference figure). Bottom block = the SYNTHIA and RUGD rows x 4 columns (RGB, GT, STDC-Seg, STDC-Seg + Ours) with their own column titles; the 4 panels can be widened to span the full figure width. No empty cells, and the figure reads as "two hosts on Cityscapes" + "one host across domains".
   - Alternative A: split into two figures (Cityscapes both hosts; cross-dataset STDC-Seg only).
   - Alternative B: one 6-column grid with the HRNet cells on SYNTHIA/RUGD rows marked "--" and a caption note. Simplest, but empty cells look like missing results.
3. Insert each row's images left to right in file order 1..6 (1..4 for SYNTHIA/RUGD).4. Aspect ratios differ: Cityscapes 2:1, SYNTHIA 1280x760 (1.68:1), RUGD 688x550 (1.25:1). Either crop every image in a row identically (Picture Format -> Crop, then copy the crop to the other panels of the row) or let rows have different heights. Always crop all panels of a row the same way.
5. Put a row label on the left (Cityscapes / SYNTHIA / RUGD), column titles on top (table above), and the legend strip(s) at the bottom: `legend_cityscapes_synthia.png` and `legend_rugd.png`.
6. Other image choices: look at `candidates/candidates_<dataset>.png`, then re-run with `--pick <dataset>:<image>`.

Caption suggestion: "Qualitative results on Cityscapes (top: STDC-Seg and HRNet-W48, each without and with Ours) and on SYNTHIA and RUGD (bottom: STDC-Seg; HRNet-W48 was trained on Cityscapes only)." If the rows are best-case picks (strategy `top`), say they were "selected for visible differences"; `--strategy median` gives typical images instead.

## Figure B -- framework diagram (STDC-Seg + Ours = HI1, Cityscapes val)

8 candidate images, each rendered completely. Pick the one you like and fill the diagram from that folder. Start with `fig_framework/INDEX_candidates.png` (one row per candidate), then open `candNN__<image>/zoom1/SHEET_zoom1.png` for every panel with its diagram label.

| Candidate folder | Image | Thin-band acc. vs baseline | RSR changed (stride-4 cells) | OHEM picks in boundary band: BPM vs stock | Band share of pixels |
|---|---|---|---|---|---|
| `fig_framework/cand01__munster_000016_000019/` | munster_000016_000019 | +36.4 pts | 1995 | 52% vs 34% | 4.5% |
| `fig_framework/cand02__frankfurt_000001_033655/` | frankfurt_000001_033655 | +33.3 pts | 2569 | 65% vs 43% | 5.5% |
| `fig_framework/cand03__frankfurt_000000_004617/` | frankfurt_000000_004617 | +31.2 pts | 4396 | 46% vs 34% | 7.0% |
| `fig_framework/cand04__lindau_000021_000019/` | lindau_000021_000019 | +32.7 pts | 1380 | 39% vs 23% | 3.0% |
| `fig_framework/cand05__frankfurt_000001_064651/` | frankfurt_000001_064651 | +29.7 pts | 2702 | 77% vs 50% | 6.7% |
| `fig_framework/cand06__munster_000036_000019/` | munster_000036_000019 | +26.9 pts | 2430 | 45% vs 31% | 6.1% |
| `fig_framework/cand07__munster_000035_000019/` | munster_000035_000019 | +26.4 pts | 2820 | 57% vs 38% | 5.5% |
| `fig_framework/cand08__lindau_000057_000019/` | lindau_000057_000019 | +25.1 pts | 4226 | 50% vs 38% | 7.2% |

Candidates = Cityscapes val images where Ours most improves thin structures near boundaries (per-image thin-band accuracy, HI1 vs the I0 baseline), at most 3 per city.

### Inside each candidate folder

```
candNN__<image>/
  where_the_zooms_are.png   the image with the zoom1 / zoom2 squares drawn in
  full/     F01..F10 + extras/X1..X8 at full GT resolution (2048x1024) + SHEET_full.png
  zoom1/    the same panels cropped to square window 1, 1536x1536   <- use these for the diagram boxes
  zoom2/    a second, non-overlapping window (alternative crop)
  meta.json numbers for the caption
```

**Accuracy and resolution.** Every panel is a real tensor of the trained HI1 model on that image (the model sees 768x1536, i.e. scale 0.75, exactly as in evaluation). *Z* is a 96x192 grid (stride 8) and *Z'* a 192x384 grid (stride 4). In the zooms, each window is 512 GT px = 384 input px = 48 stride-8 cells = 96 stride-4 cells, placed on a lattice aligned to the grid cells. All enlargements are exact integers (GT x3, stride-4 x16, stride-8 x32), so every block you see is one real grid cell, with no resampling artifacts. Label maps and grids use nearest-neighbour enlargement (crisp edges); RGB and heatmaps use Lanczos. Colour scales (loss, weight, Delta, PCA features) are computed on the whole image, so the full frame and both zooms share one scale. Each candidate's prediction was checked against the evaluation pipeline's cached prediction for the same image (agreement in `meta.json`).

**Tip:** when you scale these PNGs down in PowerPoint, set the picture's resampling to keep edges sharp: File -> Options -> Advanced -> Image Size and Quality -> "Do not compress images in file", with default resolution "High fidelity".

### Main slots (one per hatched box in duet_framework.png)

| Diagram box (label in the figure) | File (in full/ or zoom1/) | What it shows | How it was made |
|---|---|---|---|
| Input image *I* | F01_input_image.png | the val image | RGB as loaded |
| context feat. | F02_context_feat.png | context-path features fed to the decoder (FFM) | feat_cp8 (128 ch, stride 8; built from stride-16/32 context, so it looks coarse) -> first 3 PCA components as RGB |
| *Z* (stride rho) | F03_Z_stride8_logits.png | host logits before RSR, as a label map | argmax of conv_out output, stride 8 (rho = 8 for STDC-Seg) |
| *F* (stride rho/2) | F04_F_stride4_features.png | high-res detail features RSR consumes | feat_res4 (64 ch, stride 4) -> first 3 PCA components as RGB |
| *Z'* (stride rho/2) | F05_Zprime_stride4_refined.png | logits after RSR, as a label map | argmax of Z' = Z~' + gamma*Delta (verified equal to the module's output), stride 4 |
| Prediction *Y^* | F06_prediction.png | final full-resolution prediction | Z' upsampled x4 to the input then to 1024x2048, argmax; black where the GT category is not one of the 19 evaluated classes (same black as F07; unmasked version = extras/X9) |
| Ground truth *Y* | F07_ground_truth.png | GT labels | black = ignore |
| *w* (Eq. 9) -- "weight map" | F08_weight_map_w.png | BPM boundary weights | yellow = w_bnd (3) within r=3 px of a GT boundary, blue = 1, black = ignore |
| *l_i* -- "loss map" | F09_loss_map.png | per-pixel cross-entropy | log(1+CE), magma colour map |
| bottom-left box ("Output image *I*") | F10_selected_pixels_BPM.png | pixels BPM's OHEM keeps (S_w) | red = kept and inside the boundary band, yellow = kept elsewhere, on a darkened RGB; OHEM rule of loss.py with n_min = 1/16 of the pixels, as in train.py |

**The bottom-left box is mislabeled in the current diagram.** It reads "Output image *I*" with an "input image" placeholder, but BPM outputs a loss, not an image. Relabel it "OHEM-selected pixels *S_w*" and fill it with F10. That makes the BPM row end in a picture of what BPM does. Stronger option: put `extras/X8_selected_pixels_stock_OHEM.png` next to it as "stock OHEM" for contrast. Otherwise delete the box and let the arrow end at *L_BPM*.

Caption numbers are per candidate in the table above and in `meta.json`, e.g. "the boundary band is 4.5% of the pixels; stock OHEM spends 34% of its selection there, BPM 52%" (candidate 1).
Illustration caveat: the loss map and OHEM selection are computed on this val image with the trained model (per-image selection); during training they are computed on 512x1024 crops in batches of 16.

### Optional extras (`extras/`, tensors the diagram names but has no box for)

| Where it would go | File | What it shows |
|---|---|---|
| *Z~'* (after "bilinear x2") | X1_Ztilde_bilinear_x2.png | Z upsampled x2 before correction: compare with F05 to see what RSR adds |
| *P* (after ConvBNReLU) | X2_P_projected_features.png | projected F inside RSR (PCA RGB) |
| *Delta* (before x gamma) | X3_Delta_correction_magnitude.png | |gamma*Delta| per stride-4 cell: where the correction acts (inferno) |
| next to *Z'*, or as an inset | X4_pixels_changed_by_RSR.png | cyan = stride-4 cells whose label differs between Z~' and Z' (what RSR changes) |
| boundary set *B* (Eq. 8) | X5_boundary_set_B.png | GT boundary pixels (full/: drawn 3 px wide so it survives downscaling; zooms: true width x3) |
| dilate to *B_r* | X6_boundary_band_Br.png | band of radius r (white) |
| after the (x) node, *w_i l_i* | X7_weighted_loss_w_times_l.png | weighted loss map (magma) |
| contrast panel for F10 | X8_selected_pixels_stock_OHEM.png | what stock OHEM (w = 1) would keep |
| instead of F06, if you want it | X9_prediction_unmasked.png | the raw prediction everywhere, incl. regions whose category is not evaluated |

**Black in F06/F07.** Black marks pixels whose Cityscapes category is not one of the 19 evaluated classes: parking, ground, static and dynamic objects, the ego vehicle, image borders. They are real, annotated image content, but excluded from training and from every metric (standard Cityscapes protocol), and the model cannot output those categories. The prediction is masked with exactly the GT's black region, so it hides nothing that is scored. Say so once in the legend or caption, e.g. "black: not among the 19 evaluated classes" (or an "n/a" legend entry).

### How to fill the diagram in PowerPoint

1. Choose a candidate (INDEX_candidates.png), then a crop (zoom1 or zoom2; see where_the_zooms_are.png).
2. Open `ref_images/qualitative/duet_framework.pptx`. For each hatched placeholder: right-click -> Change Picture, using the same-named file from that candidate's `zoom1/` (or `zoom2/`). Use the same candidate and the same crop for every box, so the panels line up as one example.
3. Keep the label text above each box. The images replace only the hatching. For the input-image box (a 1:1 box too), use zoom1/F01; if you want the whole scene there instead, use full/F01 and crop it in PowerPoint.
4. Optional: add `extras/X4_pixels_changed_by_RSR.png` as a small inset beside *Z'*, and swap the bottom-left box for F10 + X8 as described above.
5. Label maps (Z, Z', prediction, GT) use the Cityscapes palette; the legend is `fig_qualitative/legend_cityscapes_synthia.png` if the figure needs one.

## Regenerating

```
/home/husky/anaconda3/envs/stdcseg/bin/python gen_qualitative.py                 # everything
... --pick cityscapes:<image> --pick rugd:<image>                                   # choose rows
... --framework_image <img> [<img> ...] --framework_n 8                         # framework candidates
... --strategy median                                                             # typical rows
... --render_only                                                                 # no GPU, re-render
```
