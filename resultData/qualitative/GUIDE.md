# Qualitative figures -- what every file is and how to assemble them

Generated 2026-09-29 19:27 by `gen_qualitative.py`. All paths are relative to `resultData/qualitative/`.
Models: baseline and "Ours" (= HI: RSR + BPM) only. STDC-Seg: I0 / HI1 (Cityscapes), Synthia / Synthia-HI1, RUGD / RUGD-HI1. HRNet-W48: baseline / HI1 (Cityscapes only -- the only dataset HRNet was trained on).

## Folder map

```
fig_qualitative/                  Figure A (cf. ref_images/qualitative/QualitativeData.png)
  PREVIEW_qualitative.png         auto-assembled preview of the whole figure (reference only)
  PREVIEW_qualitative_clean.png   same without the dashed boxes
  legend_cityscapes_synthia.png   colour legend for Cityscapes + SYNTHIA rows (same classes)
  legend_rugd.png                 colour legend for RUGD rows
  rowN_<dataset>__<image>/        one folder per figure row
    1_rgb.png 2_gt.png 3_stdc.png 4_stdc_ours.png [5_hrnet.png 6_hrnet_ours.png]
    boxed/   same panels with a dashed box around the region where Ours helps most
    zoom/    the boxed region cropped (512 px wide) -- for insets
    extra/   diagnostics: green = pixel fixed by Ours, red = pixel broken by Ours (not for the paper grid)
  candidates/candidates_<dataset>.png   top-10 alternative images per dataset, with scores
fig_framework/                    Figure B (cf. ref_images/qualitative/duet_framework.png)
  F01..F10_*.png                  one image per hatched box in the diagram (table below)
  zoom/                           the same images cropped to one square region (recommended for the boxes)
  extras/ (+ extras/zoom/)        optional panels for tensors the diagram names but has no box for
  framework_meta.json             numbers for the caption (gamma, OHEM statistics, sizes)
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
2. Layout choice for the missing HRNet cells on SYNTHIA/RUGD rows (HRNet exists for Cityscapes only):
   - **Recommended:** two blocks. Top block: the 2 Cityscapes rows x 6 columns. Bottom block: the 4 SYNTHIA/RUGD rows x 4 columns (RGB, GT, STDC-Seg, STDC-Seg + Ours), separated by a dashed line like the reference figure. No empty cells.
   - Alternative: one 6-column grid with the HRNet cells on SYNTHIA/RUGD rows left blank or marked "--".
3. Insert the row images left to right in file order 1..6. Use either the plain files or the `boxed/` versions (the box is the same in every column of a row, placed where Ours fixes the most pixels). If you prefer your own boxes, use the plain files and draw a dashed rectangle in PowerPoint.
4. Aspect ratios differ: Cityscapes 2:1, SYNTHIA 1280x760 (1.68:1), RUGD 688x550 (1.25:1). Either crop every image in a row identically (Picture Format -> Crop, then copy the crop to the other panels of the row) or let rows have different heights. Always crop all panels of a row the same way.
5. Put a row label on the left (Cityscapes / SYNTHIA / RUGD), column titles on top (table above), and the legend strip(s) at the bottom: `legend_cityscapes_synthia.png` and `legend_rugd.png`.
6. Optional insets: the `zoom/` crops can go below or beside a row to magnify the boxed region.
7. Other image choices: look at `candidates/candidates_<dataset>.png`, then re-run with `--pick <dataset>:<image>`.

Caption suggestion: "Qualitative results on Cityscapes, SYNTHIA and RUGD val. Dashed boxes mark regions where Ours corrects boundaries and thin structures. HRNet-W48 was trained on Cityscapes only." If the rows are best-case picks (strategy `top`), say they were "selected for visible differences"; `--strategy median` gives typical images instead.

## Figure B -- framework diagram (STDC-Seg + Ours, image `lindau_000021_000019`)

Every image is a real intermediate tensor of the HI1 model on this Cityscapes val image. The model sees the image at 768x1536 (scale 0.75). Feature maps are drawn at their true resolution (nearest-neighbour upsampling), so the blockiness of Z vs Z' IS the point of RSR. For the boxes in the diagram, use the `zoom/` versions: the full frames are 2:1, the boxes are square, and the zoom region (see `zoom/where_the_zoom_is.png`) is where RSR changed the most thin-structure pixels.

### Main slots (one per hatched box in duet_framework.png)

| Diagram box (label in the figure) | File | What it shows | How it was made |
|---|---|---|---|
| Input image *I* | F01_input_image.png | the val image | RGB as loaded |
| context feat. | F02_context_feat.png | context-path features fed to the decoder (FFM) | feat_cp8 (128 ch, stride 8) -> first 3 PCA components as RGB |
| *Z* (stride rho) | F03_Z_stride8_logits.png | host logits before RSR, as a label map | argmax of conv_out output, stride 8 (96x192 grid) |
| *F* (stride rho/2) | F04_F_stride4_features.png | high-res detail features RSR consumes | feat_res4 (64 ch, stride 4) -> PCA RGB |
| *Z'* (stride rho/2) | F05_Zprime_stride4_refined.png | logits after RSR, as a label map | argmax of Z' = Z~' + gamma*Delta, stride 4 (192x384 grid) |
| Prediction *Y^* | F06_prediction.png | final full-resolution prediction | Z' upsampled to 1024x2048, argmax (not masked) |
| Ground truth *Y* | F07_ground_truth.png | GT labels | black = ignore |
| *w* (Eq. 9) -- "weight map" | F08_weight_map_w.png | BPM boundary weights | yellow = w_bnd (3) within r=3 px of a GT boundary, blue = 1, black = ignore |
| *l_i* -- "loss map" | F09_loss_map.png | per-pixel cross-entropy | log(1+CE), magma colour map |
| bottom-left box ("Output image *I*") | F10_selected_pixels_BPM.png | pixels BPM's OHEM keeps (S_w) | red = kept & inside boundary band, yellow = kept elsewhere, on a darkened RGB; n_min = 131072 (1/16 of pixels, as in train.py) |

**The bottom-left box is mislabeled in the current diagram.** It reads "Output image *I*" with an "input image" placeholder, but BPM outputs a loss, not an image. Relabel it "OHEM-selected pixels *S_w*" and fill it with F10. That makes the BPM row end in a picture of what BPM does. Stronger option: put `extras/X8_selected_pixels_stock_OHEM.png` next to it as "stock OHEM" for contrast. Otherwise delete the box and let the arrow end at *L_BPM*.

Numbers for the caption (from `framework_meta.json`): on this image, 3% of valid pixels lie in the r=3 boundary band. Stock OHEM keeps 23% of its selected pixels in that band; BPM keeps 39%. RSR changed 1380 stride-4 pixels; gamma (res_scale) = +2.34.

### Optional extras (`extras/`, tensors the diagram names but has no box for)

| Where it would go | File | What it shows |
|---|---|---|
| *Z~'* (after "bilinear x2") | X1_Ztilde_bilinear_x2.png | Z upsampled x2 before correction: compare with F05 to see what RSR adds |
| *P* (after ConvBNReLU) | X2_P_projected_features.png | projected F inside RSR (PCA RGB) |
| *Delta* (before x gamma) | X3_Delta_correction_magnitude.png | |gamma*Delta| per pixel: where the correction acts (inferno) |
| next to *Z'*, or as an inset | X4_pixels_changed_by_RSR.png | cyan = pixels whose label differs between Z~' and Z' (what RSR fixes) |
| boundary set *B* (Eq. 8) | X5_boundary_set_B.png | GT boundary pixels (drawn 3 px wide for visibility) |
| dilate to *B_r* | X6_boundary_band_Br.png | band of radius r (white) |
| after the (x) node, *w_i l_i* | X7_weighted_loss_w_times_l.png | weighted loss map (magma) |
| contrast panel for F10 | X8_selected_pixels_stock_OHEM.png | what stock OHEM (w = 1) would keep |

### How to fill the diagram in PowerPoint

1. Open `ref_images/qualitative/duet_framework.pptx`.
2. For each hatched placeholder: right-click -> Change Picture (or insert the file and send the placeholder backward), using the file in the table above. Use the `zoom/` version for square boxes.
3. Keep the label text above each box (e.g. "*Z* (stride rho)"). The images replace only the hatching.
4. Optional: add a small inset of `extras/zoom/X4_pixels_changed_by_RSR.png` beside *Z'*, and swap the bottom-left box for F10 + X8 as described above.
5. Label maps (Z, Z', prediction, GT) use the Cityscapes palette; the legend is `fig_qualitative/legend_cityscapes_synthia.png` if the figure needs one.

## Regenerating

```
/home/husky/anaconda3/envs/stdcseg/bin/python gen_qualitative.py                 # everything
... --pick cityscapes:<image> --pick rugd:<image>                                   # choose rows
... --framework_image <cityscapes image>                                          # framework image
... --strategy median                                                             # typical rows
... --render_only                                                                 # no GPU, re-render
```
