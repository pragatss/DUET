# resultData — raw numbers for the paper tables and figures

Every file here is produced by a script in the repo root. Do not edit these files by hand;
re-run the generator instead. Unless a column says otherwise, IoU values are in **%**
and deltas are in **points**.

| folder | generator | paper artifact | reference layout |
|---|---|---|---|
| `perclass/` | `gen_perclass.py` | per-class IoU tables per dataset (Cityscapes: STDC-Seg and HRNet; SYNTHIA, RUGD: STDC-Seg; each ± Ours) and a cross-dataset summary | `ref_images/quantative/PerClass.png` |
| `ablation/` | `gen_ablation.py` | ablation table (STDC-Seg only, vs full model HI1) | `ref_images/quantative/Ablation.png` |
| `runtime/` | `gen_runtime_memory.py` | training/inference throughput + GPU memory | `ref_images/quantative/RuntimeAndMemory.png` |
| `qualitative/` | `gen_qualitative.py` | qualitative grid panels + framework-diagram panels; see `qualitative/GUIDE.md` | `ref_images/qualitative/QualitativeData.png`, `duet_framework.png` |
| `figure_assets/` | `tools/make_figure_assets.py` | framework-figure tiles, one folder per candidate image (`candNN__<image>/`): img_{input,Z,Zprime,weight,loss,pred,gt}.{png,pdf} at 1632x1440 (full-res crop x3), ref_crop_overview.png, assets_meta.json (crop, k, res_scale, checks, metrics); INDEX.png compares candidates | `duet_framework.png` |
| `cache/eval/` | `paper/eval_worker.py` | one JSON per evaluated checkpoint | — |
| `digest/` | `gen_digest.py` | `eval_digest.md`, `perclass_digest.md`: compact, self-describing summaries for building the paper tables elsewhere | — |

All three generators run under any python. They start the GPU work themselves, one
subprocess per host, each in that host's conda env (`paper/common.py` `ENV_PY`; override
with the `STDC_PY` / `HRNET_PY` env vars).

## Metric definitions (identical for every host)
- **full**: full-image IoU over valid (non-ignore) pixels. mIoU = mean over classes with union > 0.
- **bnd_rN**: the same IoU restricted to pixels within N px (max-pool dilation) of a GT class
  boundary, computed on the full-resolution GT. This is the primary metric.
- **thin_rN**: mean IoU of the dataset's thin classes inside the r=N band. Cityscapes and SYNTHIA: pole,
  traffic light, traffic sign, rider, motorcycle, bicycle. RUGD: pole, sign, fence, bicycle, log.
  Same rule everywhere: narrow or small structures, typically thinner than one stride-8 cell.
- Each dataset is scored in its own classes (no mapping to Cityscapes). The boundary band is
  wherever the GT label changes, so the metric works for any label set.
- **RUGD support filter**: RUGD val is two videos, and several classes are nearly absent. Every
  RUGD mean (full, boundary, thin) uses only classes with >= 0.1% of val GT pixels, the same as
  `eval_rugd.py`. `mIoU (all classes)` is the unfiltered mean over present classes, which is
  what train.py logs. Classes absent from a val set (SYNTHIA: terrain, truck, train) are blank
  and excluded.
- The math is a copy of `evaluation.py` (`paper/common.py` `SegMeter`). The new evaluator reproduces the HI1 row
  of `ablation_results.txt` to 4 decimals.
- Each host is evaluated with its own native protocol (`paper/common.py` `PROTOCOL`):
  STDC-Seg input resized to scale 0.75; HRNet at full resolution. Compare only within a host.
- SYNTHIA and RUGD at scale 0.75 (570x960, 412x516) are not 32-divisible, and non-32-divisible
  inputs caused boundary artifacts before. Each checkpoint is therefore also scored at a nearby
  32-divisible size (576x960, 416x512), in `<dataset>/scale_check.csv`. The Ours-minus-baseline
  deltas should agree between the two.

## cache/eval/
`<host>__<key>[__s32].json` (`__s32` = 32-divisible input): summary metrics, per-class IoU / GT pixel counts / union for every
region, and a fingerprint (checkpoint path, mtime, size, protocol). A cache entry is used
only when its fingerprint still matches the checkpoint on disk. If a checkpoint is retrained,
its entry is re-evaluated automatically. `--force` re-evaluates everything.

## perclass/
- `datasets_summary.csv` / `.md`: every dataset x method in one table: full mIoU, full mIoU
  (all classes), bnd r=1 / r=3, thin r=1 / r=3, and deltas vs the host baseline. Start here.
- `<dataset>/` (cityscapes, synthia, rugd):
  - `perclass_{full,bnd_r1,bnd_r3}.csv`: one table per region. The first rows are `support`
    (% of val GT pixels per class) and `mask` (1 = class enters the means). `Row type`:
    `main` = the paper rows; `replicate` = single Cityscapes baseline seeds behind the mean;
    `arm` = a single-arm run (e.g. SYNTHIA BPM-only). The last column is the mIoU delta vs
    that host's baseline row.
  - `perclass_delta.csv`: Ours minus baseline per class, per host and region.
  - `perclass_long.csv`: tidy format (iou as a fraction, used_in_mean, gt_pixels), for plots.
  - `perclass_summary.json`: everything at full precision, plus the thin and used class lists.
  - `scale_check.csv` (synthia, rugd): scale 0.75 vs 32-divisible input.

## ablation/
- `ablation.csv`: for each metric, the absolute value, `d` (variant minus full model), and `d/noise`
  (|d| divided by the seed-noise floor = that metric's range across the baseline replicates).
  `Status=pending` rows have no checkpoint yet.
- `ablation_sensitivity.csv`: the bnd_weight {1,2,3,5,8} and bnd_radius {1,2,3,5} sweeps (w=3/r=3 is HI1; w=1 is H1).
- `ablation_thin_r1.csv`: per-thin-class IoU at r=1, with deltas vs the full model.

## runtime/
- `runtime_memory.csv`: one row per method.
  - Train: one full training iteration (forward, every loss, backward, SGD) at that host's
    training batch (STDC 16x3x512x1024, HRNet 3x3x512x1024), with real augmented batches.
    Data loading is excluded.
  - Infer: batch 1, a 1024x2048 image in, a label map out (including the host's resize and final upsample).
  - Memory: peak `torch.cuda.max_memory_allocated` (tensors) and peak reserved (allocator
    cache), measured in an isolated pass with cudnn.benchmark off. nvidia-smi would read about
    0.3-0.5 GB higher because of the CUDA context.
  - Throughput is pooled over 30 fine-grained interleaved rounds after a 90 s heat soak. The T4
    throttles to 825-1545 MHz at 83-84 °C, so absolute numbers reflect a throttled T4.
  - `d ... (% paired)`: median over rounds of (config / host baseline) within the same round.
    Use this for the overhead claim, not the difference of the absolute columns.
  - `As-trained it/s`: median from the real training log, which includes data loading and periodic validation.
  - `GMACs` = Conv2d+Linear multiply-accumulates, the same counter for both hosts. `thop GFLOPs`
    matches pareto_flops.csv but is STDC-only (thop isn't installed in the hrnet env). The two
    counters differ, so never mix them in one table.
- `raw/<host>.json`: every per-round sample, the GPU temperature/clock/power per round, and the memory pass.

## qualitative/
See `qualitative/GUIDE.md`, which is generated with the actual image names: every file, which box of each
figure it goes in, and how to assemble both figures in PowerPoint. `cache/` holds the per-image predictions
(gitignored, regenerable).
