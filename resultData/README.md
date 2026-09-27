# resultData — raw numbers for the paper tables and figures

Every file here is produced by a script in the repo root. Do not edit these files by hand;
re-run the generator instead. Unless a column says otherwise, IoU values are in **%**
and deltas are in **points**.

| folder | generator | paper artifact | reference layout |
|---|---|---|---|
| `perclass/` | `gen_perclass.py` | per-class IoU table (STDC-Seg and HRNet, each ± Ours) | `ref_images/quantative/PerClass.png` |
| `ablation/` | `gen_ablation.py` | ablation table (STDC-Seg only, vs full model HI1) | `ref_images/quantative/Ablation.png` |
| `runtime/` | `gen_runtime_memory.py` | training/inference throughput + GPU memory | `ref_images/quantative/RuntimeAndMemory.png` |
| `qualitative/` | (planned) `gen_qualitative.py` | prediction grid | `ref_images/qualitative/QualitativeData.png` |
| `cache/eval/` | `paper/eval_worker.py` | one JSON per evaluated checkpoint | — |

All three generators run under any python. They start the GPU work themselves, one
subprocess per host, each in that host's conda env (`paper/common.py` `ENV_PY`; override
with the `STDC_PY` / `HRNET_PY` env vars).

## Metric definitions (identical for every host)
- **full**: full-image IoU over valid (non-ignore) pixels. mIoU = mean over classes with union > 0.
- **bnd_rN**: the same IoU restricted to pixels within N px (max-pool dilation) of a GT class
  boundary, computed on the full-resolution 1024x2048 GT. This is the primary metric.
- **thin_rN**: mean IoU of pole, traffic light, traffic sign, rider, motorcycle, bicycle inside the r=N band.
- The math is a copy of `evaluation.py` (`paper/common.py` `SegMeter`). The new evaluator reproduces the HI1 row
  of `ablation_results.txt` to 4 decimals.
- Each host is evaluated with its own native protocol (`paper/common.py` `PROTOCOL`):
  STDC-Seg input resized to scale 0.75; HRNet at full resolution. Compare only within a host.

## cache/eval/
`<host>__<key>.json`: summary metrics, per-class IoU / GT pixel counts / union for every
region, and a fingerprint (checkpoint path, mtime, size, protocol). A cache entry is used
only when its fingerprint still matches the checkpoint on disk. If a checkpoint is retrained,
its entry is re-evaluated automatically. `--force` re-evaluates everything.

## perclass/
- `perclass_{full,bnd_r1,bnd_r3}.csv`: one table per region. `Row type` = `main` (the
  paper rows) or `replicate` (single STDC baseline seeds that make up the mean). The last
  column is the mIoU delta vs that host's baseline row.
- `perclass_delta.csv`: Ours minus baseline per class, per host and region.
- `perclass_long.csv`: tidy format (host, method, region, class, iou fraction, gt_pixels), for plots.
- `perclass_summary.json`: everything at full precision.

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

## qualitative/ (planned)
Grid: rows = 2 Cityscapes, 2 SYNTHIA, 2 RUGD; columns = RGB, GT, STDC-Seg, STDC-Seg+Ours,
HRNet, HRNet+Ours (a transformer host is optional or not yet available). HRNet was trained
on Cityscapes only, so the SYNTHIA and RUGD rows have no HRNet predictions unless HRNet is
trained on those datasets.
