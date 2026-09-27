# Project context: boundary segmentation for real-time semantic segmentation

## What this project is
This repo supports an IEEE RA-L paper (working title "DUET: Coupling Boundary
Supervision with Decision Resolution for Real-Time Semantic Segmentation").
The goal is to improve BOUNDARY quality of real-time semantic segmentation
(thin structures, class edges) without hurting overall accuracy, using two
small, plug-and-play additions that fix two separate bottlenecks.

## The two contributions (always treat them as a pair)
Paper names in parentheses; code and run names use "Arm H" / "Arm I".

1. Arm H = BoundaryRefine Head, BRH (paper: RSR, Residual Sub-grid Refinement)
   - Problem it fixes (RESOLUTION): STDC-Seg predicts logits at stride 8 and
     upsamples 8x in one jump. Anything thinner than one stride-8 cell (e.g. a
     2-px pole) cannot be drawn. feat_res4 (stride 4, 64ch) is computed every
     forward pass but never reaches the seg logits.
   - What it does: after `feat_out = self.conv_out(feat_fuse)`, upsample the
     stride-8 logits to stride 4, project feat_res4 with ConvBNReLU, concat +
     fuse, 1x1 conv -> delta, residual add:
     output = upsampled + res_scale * correction
   - res_scale is a zero-initialized nn.Parameter, so at init the model is
     bit-identical to baseline. ~0.086M params. No new loss.
   - Flag: --use_brh (must match between train and eval).

2. Arm I = BoundaryOhemCELoss (paper: BPM, Boundary-Priority Mining)
   - Problem it fixes (OBJECTIVE): OhemCELoss weights all pixels equally, so
     boundary pixels are a tiny minority and interiors dominate the gradient.
     The existing detail-head BCE/dice losses only teach WHERE edges are; they
     never ask the seg head to CLASSIFY boundary pixels correctly.
   - What it does: builds a boundary weight map from GT labels (neighbor
     comparison + max_pool2d dilation), multiplies per-pixel CE by the weight
     BEFORE OHEM sort-and-select, so boundary pixels are both weighted more
     and more likely to survive hard-example mining.
   - Loss-only: zero params, zero inference cost, no eval flag needed.
   - At w_bnd=1.0 it is algebraically identical to stock OhemCELoss (this is
     the I0 control). Applied only to criteria_p (main head); criteria_16/32
     stay stock.
   - Flags: --bnd_weight, --bnd_radius.

Framing: Arm I changes what the network is rewarded for; Arm H changes what
the network can physically draw. Orthogonal levers -> they compose.

## Host architectures
- STDC-Seg (primary host): this repo, fork of MichaelFan01/STDC-Seg.
  Branch RUGD-ARMIANDH is the most up-to-date H+I code.
  Key locations (verify, line numbers may have drifted):
    models/model_stages.py : BoundaryRefine (~L41), res_scale (~L50),
                             get_params() appends res_scale (~L74)
    loss/loss.py           : boundary_weight_map() (~L30),
                             BoundaryOhemCELoss (~L52)
    train.py               : H/I flags (~L139-142), criteria_p (~L231-232)
    evaluation.py          : IS the boundary scorer (there is no separate
                             boundary_eval.py). Per-class IoU at r=1 and r=3,
                             thin-class reporting, editable RUNS list at the
                             bottom, prints res_scale, warns on key mismatch.
  STDC-Seg's own "detail head" / boundary head (1-channel edge prediction
  with BCE + dice, Laplacian-derived targets) is part of the stock model and
  is separate from Arm H.

- HRNet (second host, generalizability): github.com/HRNet/HRNet-Semantic-
  Segmentation, master branch, **HRNetV2-W48** (NOT w18-small-v2 -- verified
  from the yamls/pretrained path), Cityscapes only, 120 epochs, batch 3,
  crop 512x1024, OHEM on in both arms. HRNet keeps
  high-resolution branches in parallel through the whole network and fuses
  multi-resolution features repeatedly, so it is a structurally different
  host from STDC-Seg's single-path encoder + detail branch. Scope: baseline x1
  and HI1 x1 only. Files touched: lib/config/default.py,
  lib/core/criterion.py, lib/models/seg_hrnet.py, tools/train.py,
  lib/utils/utils.py, two new YAMLs, a verification script.
  On HRNet, BRH refines stride-4 logits to stride 2 using the stem's stride-2
  64ch activation (feat_half).
  Checkout: ../../hrnet/HRNet-Semantic-Segmentation (relative to this repo;
  NOT tracked in git -- we only read from it, all new code lives here).
    configs : experiments/cityscapes/arm_{baseline,HI1}_w48.yaml
    ckpts   : output/hrnet_{baseline,HI1}/cityscapes/arm_{baseline,HI1}_w48/best.pth
              (FullModel state dict: 'model.*' + 'loss.*'; best val mIoU,
              training log: baseline 0.7562, HI1 0.7722; HI1 res_scale +1.40)
    env     : /home/husky/anaconda3/envs/hrnet (torch 1.1.0, no thop)

- A third, transformer-based host is required but NOT yet chosen.

## Datasets
Cityscapes (main results + all ablations), SYNTHIA, RUGD (off-road).
A small local sample for reference lives at
/Users/pragatwagle/Desktop/research/data (NOT a full dataset; use it only to
understand formats, label IDs, and loaders).

## Metrics — read this before interpreting any number
- PRIMARY: boundary-region IoU at radius r=1 and r=3, plus the same on thin
  classes (see the thin-class list in evaluation.py).
- Full-image mIoU is a REGRESSION GUARD ONLY. Its run-to-run spread (0.0123)
  exceeds every arm's effect, so it can never support a claim.
- Always compare to the 3-replicate baseline MEAN, and express effects as a
  multiple of the noise floor:
    noise floor: full 0.0123 | bnd r1 0.0032 | bnd r3 0.0048 |
                 thin r1 0.0041 | thin r3 0.0055
- Protocol: Cityscapes val, scale 0.75, model_maxmIOU75.pth, 60k iters,
  nproc=1. Use 32-divisible input scales only (others create boundary-metric
  artifacts).
- Cost: report params and GFLOPs. Wall-clock FPS on the T4 is unreliable
  (thermal throttling), so it is secondary.

## Current results (Cityscapes, bnd r=1 / thin r=1)
  Baseline (n=3 mean) 0.3580 / 0.3316
  I1  (w_bnd=3)       0.3778 / 0.3543
  H1  (use_brh)       0.3757 / 0.3540
  HI1 (brh + w=3)     0.3977 / 0.3862   <- ~12x noise, >= sum of parts
Treatment arms are n=1 so far; HI1 replicates are pending.

HRNet-W48, Cityscapes (full-res eval, n=1 each; gen_perclass.py 2026-09-27):
  full / bnd r1 / bnd r3 / thin r1 / thin r3
  baseline  0.7562 / 0.4088 / 0.5089 / 0.4170 / 0.5160
  HI1       0.7722 / 0.4324 / 0.5375 / 0.4528 / 0.5581
  -> +2.4 bnd r1, +3.6 thin r1. Only per-class loss is Train (-2.5 bnd r1),
     a class with ~40% IoU and tiny val support; truck is ~flat.

Cost (gen_runtime_memory.py 2026-09-27, T4, paired deltas vs host baseline):
  STDC  +Ours: +0.086M params, +9.5% GMACs, train -8.5% it/s (+0.58 GB),
               infer -5 to -9% img/s (RSR-only and Ours share an arch but
               read -9.2 / -4.9, so about +/-3% timing noise remains), infer mem unchanged.
               BPM only: -1% train/infer, which is noise (arch identical).
  HRNet +Ours: +0.086M params, +6.0% GMACs, train -4.8% (+0.53 GB),
               infer -6.6%, infer mem unchanged.

## Ablation ladder (Cityscapes / STDC-Seg only; each compared to HI1)
Module (Arm H) removals:
  M1 no HR feats (done) | M2 stride 8 (done) | M3 no_residual | M4 no_logit
  | M5 no_gate
Loss (Arm I) removals:
  L1 unweighted_rank (done) | L2 no OHEM (done) | L2b no OHEM, w=1
  | L3 no detail loss | L4 no aux
Sensitivity sweeps: --bnd_weight {2,5,8}, --bnd_radius {1,2}
Also needed: runtime/memory table (baseline vs H1 vs HI1).

## Paper data pipeline (numbers for tables/figures) -- keep this current
Only this repo (STDC-Seg) is tracked in git, so ALL new code goes here,
including code that evaluates HRNet (it imports HRNet's lib/ by path and reads
its checkpoints in place). Outputs go to resultData/ (see resultData/README.md
for every file and column). The user builds the final tables from those
files, so keep outputs as plain CSV/JSON/MD.

Reference layouts (what the outputs must be able to fill):
  ref_images/quantative/PerClass.png          -> gen_perclass.py
  ref_images/quantative/Ablation.png          -> gen_ablation.py
  ref_images/quantative/RuntimeAndMemory.png  -> gen_runtime_memory.py
  ref_images/qualitative/QualitativeData.png  -> gen_qualitative.py (TODO)

Checkpoints: models_i_care_about.txt is the source of truth (cutoff: I0 and
newer). The registry in paper/common.py (STDC_RUNS, HRNET_RUNS) holds the
paper keys. Add new runs there, not in the scripts.

Layout:
  paper/common.py          paths, run registry, PROTOCOL, SegMeter (a copy of
                           evaluation.py's boundary math), eval cache, CSV/JSON helpers
  paper/host_stdc.py       STDC build/eval/train-step (reuses evaluation.load_net)
  paper/host_hrnet.py      HRNet build/eval/train-step (sys.path -> HRNet lib)
  paper/eval_worker.py     scores checkpoints -> resultData/cache/eval/<host>__<key>.json
  paper/runtime_worker.py  throughput/memory/params/GMACs -> resultData/runtime/raw/
  gen_perclass.py          Cityscapes per-class IoU, full + bnd r1/r3,
                           STDC (n=3 baseline mean) and HRNet, each +/- Ours
  gen_ablation.py          STDC-only ablations vs HI1, all 5 metrics, with d and d/noise
  gen_runtime_memory.py    train/infer throughput + peak memory, params, GMACs,
                           plus as-trained it/s parsed from the training logs
Every gen_*.py runs under any python and launches GPU work itself, one
subprocess per host in that host's env (stdc -> envs/stdcseg, hrnet ->
envs/hrnet). STDC and HRNet cannot share a process: both have a top-level
`models` package.

Decisions:
- Per-class STDC baseline = mean of I0/Baseline2/Baseline (--stdc_baseline I0
  for a single run). HRNet rows are n=1.
- Each host is scored with its native protocol (STDC scale 0.75; HRNet full
  res). Boundary bands always use the full-res GT. Compare only within a host.
- Ablations: primary = bnd r1/r3 + thin, full mIoU = guard. Full-mIoU
  ablation deltas (<=0.7 pt) are all below seed noise (1.2 pt), and M1 noHR
  even beats HI1 on full mIoU. A full-mIoU-only table would read as "nothing
  matters".
- Runtime: params/GMACs are the primary cost (exact). it/s and img/s come
  from interleaved rounds on an unlocked T4, so trust relative values more
  than absolute ones. Peak memory is measured with cudnn.benchmark OFF:
  with it ON, autotuning scratch space made HRNet HI1 inference read 11.6 GB
  vs 5.9 GB for the baseline (a measurement artifact, not a real cost).
  Timing: inference nets resident, training configs rebuilt per block (4
  STDC train configs at batch 16 don't fit together and OOM the T4), 90 s
  heat soak, 30 rounds of short random-order blocks. Overhead = median paired per-round ratio. The first version
  (coarse 5-round blocks) read STDC "BPM only" as 8.6% slower at inference,
  even though it is architecturally identical to the baseline: clock
  wandered 825-1545 MHz between blocks. GMACs = conv+linear hook counter
  for both hosts. thop (pareto_flops.csv) counts ~1.5x higher and exists for STDC only.
- checkpoints/train_STDC2-Seg-Baseline is in torch>=1.6 zip format, which
  torch 1.1 can't load. ensure_evaluated() routes zip checkpoints to
  envs/stdcseg18 (torch 2.4) automatically. It reproduces RESULTS.md exactly.
- The I0 cutoff date "2026-08-09 21:05" in models_i_care_about.txt is the
  name of an EMPTY log. I0's actual 60k-iter log is BiSeNet-2026-08-07-18-59-32.log.

Qualitative (TODO, same pattern -> resultData/qualitative/): 6 rows = 2
Cityscapes, 2 SYNTHIA, 2 RUGD; columns = RGB, GT, STDC-Seg, STDC-Seg+Ours,
HRNet, HRNet+Ours (a transformer host is optional or unavailable). Gap: HRNet
has only Cityscapes checkpoints, so the SYNTHIA and RUGD rows have no HRNet
predictions. Plan: a gen_qualitative.py that saves per-image
color-coded predictions + GT + RGB (Cityscapes palette, as in the reference) and
ranks candidate images by per-image boundary-IoU gain, so the user can pick
the rows.

## Out of scope — do not modify or revive
SBG (Arms A-G, branch ARM-D-Option1) and CtxGCN (branch GCN) are archived
negative results. SynthiaBaseline is a separate dataset track.

## Known traps (these have each cost real time before)
- argparse `type=bool` is broken; use the repo's `str2bool`.
- nn.Parameter (res_scale) is silently dropped by the get_params() pattern
  unless appended explicitly to the no-weight-decay list.
- Train and eval flags must match exactly, or eval silently builds the wrong
  model.
- train.py uses find_unused_parameters=True (why dead conv_out_sp8 doesn't
  crash).
- train.py sets no seed; replication = different --respath, no code change.
- HRNet: (1) its OHEM ranks by GT-class probability, not loss, so Arm I needs
  a p^w ranking-key transform, not just loss weighting; (2) init_weights()
  overwrites every nn.Conv2d including BRH's and must be guarded;
  (3) zero-initializing BOTH the delta conv and res_scale kills the module
  permanently; (4) adjust_learning_rate() only updates param_groups[0];
  (5) the stock w18-small-v2 YAML has USE_OHEM: false, so the baseline must
  enable OHEM explicitly or the comparison changes two variables.

## How to work with me
- Explain plainly first, then the mechanism, then numbers. Don't lead with
  code.
- Flag failure modes honestly and push back when code or data doesn't support
  a claim.
- For every change: say what to test and how we'll know it worked.
- Never change training/eval code without telling me which runs it affects
  and whether existing checkpoints stay comparable.