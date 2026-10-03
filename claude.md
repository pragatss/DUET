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
  Branch 10-3-Cleanup (2026-10-03) is the cleaned, current tree; it contains
  RUGD-ARMIANDH (the earlier most up-to-date H+I code).
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
  Segmentation, HRNet-OCR branch (base commit 0bbb288; NOT master), **HRNetV2-W48** (NOT w18-small-v2 -- verified
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

SYNTHIA / RUGD, STDC only, n=1 (gen_perclass.py; resultData/perclass/datasets_summary.*):
  full / bnd r1 / thin r1 at scale 0.75 (the training-log protocol)
  SYNTHIA base 72.56 / 37.41 / 31.40   HI1 77.19 / 43.67 / 38.56  (+4.6/+6.3/+7.2)
          "I1" 72.90 / 38.17 / 31.81   (BPM alone ~ +0.3..0.8: all of the SYNTHIA gain needs RSR)
  RUGD    base 53.14 / 25.22 / 17.16   HI1 54.62 / 26.65 / 19.23  (+1.5/+1.4/+2.1)
          (support-filtered, >= 0.1% of val px; all-class mIoU 41.56 -> 43.75)
  CHECKPOINT TRAPS (fixed in paper/common.py, don't undo):
  - Synthia/model_maxmIOU75.pth was OVERWRITTEN on 08-30 by an aborted restart
    (first val 0.2063). Use model_iter58000_..._0.7256.pth.
  - Synthia-HI1/ contains two 60k runs. The 08-30 one has no brh.* weights, so it
    is inferred to be the I1 command launched into the wrong respath (flags
    not logged). Registered as 'Synthia-I1' -> iter50000 (0.729).
    model_maxmIOU75.pth in that folder is the real HI1 (res_scale -3.22).
  32-divisible check: SYNTHIA at 576x960 instead of 570x960 gives +5.8 mIoU for
  EVERY model (base 78.38, HI1 82.16), so scale 0.75 is badly hit by the
  stride-32 artifact on SYNTHIA. Ours - baseline holds (+3.8 full, +7.4 bnd r1).
  RUGD 416x512: deltas hold (+1.2 full, +1.0 bnd r1, +2.1 thin). All-class mIoU
  is unstable (+5.9 there), which is why the support filter is used.
  RUGD thin set after the support filter = fence + log only: bicycle is absent,
  and pole (0.083%), sign, person are below 0.1%. Pole shows the largest RUGD gain
  (bnd r1 15.7 -> 27.6) but is excluded; don't lower the threshold after the fact.

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

Checkpoints: runs/CHECKPOINTS.txt is the source of truth (cutoff: I0 and
newer). The registry in paper/common.py (STDC_RUNS, HRNET_RUNS) holds the
paper keys. Add new runs there, not in the scripts.

Layout:
  paper/common.py          paths, run registry, PROTOCOL, SegMeter (a copy of
                           evaluation.py's boundary math), eval cache, CSV/JSON helpers
  paper/host_stdc.py       STDC build/eval/train-step (reuses evaluation.load_net)
  paper/host_hrnet.py      HRNet build/eval/train-step (sys.path -> HRNet lib)
  paper/eval_worker.py     scores checkpoints -> resultData/cache/eval/<host>__<key>.json
  paper/runtime_worker.py  throughput/memory/params/GMACs -> resultData/runtime/raw/
  gen_perclass.py          per-class IoU (full + bnd r1/r3) per dataset, plus
                           datasets_summary.* (all datasets, mIoU + boundary):
                           Cityscapes = STDC (n=3 baseline mean) + HRNet,
                           SYNTHIA/RUGD = STDC only; each +/- Ours. Datasets are
                           scored in their own classes (paper/common.py DATASETS).
  gen_ablation.py          STDC-only ablations vs HI1, all 5 metrics, with d and d/noise
  gen_runtime_memory.py    train/infer throughput + peak memory, params, GMACs,
                           plus as-trained it/s parsed from the training logs
  gen_qualitative.py       qualitative figure + framework-diagram panels, GUIDE.md
  tools/make_figure_assets.py  one image -> print-quality framework-figure tiles (input, Z, Z',
                           w, CE loss, masked prediction, GT; PNG + lossless PDF). Hooks on
                           net.conv_out / net.brh. Output = full-res crop x3, so Z cells are
                           exactly 32 px, Z' 16 px (crop on 32-px lattice). Fixed loss range
                           0.01..3. Candidates: resultData/figure_assets/candNN__<image>/ + INDEX.png
                           (same 8 images as fig_framework). Reuse a crop: --crop_from <meta>.
  eval_checkpoint.py       any checkpoint(s) on cityscapes/synthia/rugd by path, RSR auto-detected from
                           the weights; same code path as gen_perclass (verified identical numbers)
  scripts/train.sh <RUN>   portable training by registry name (flags == runs/*_commands.txt)
  prepare_synthia.py       raw SYNTHIA-RAND-CITYSCAPES -> data/SYNTHIA split via splits/synthia/*.txt
  splits/                  SYNTHIA (our own random split) and RUGD frame lists
  runs/                    original command logs + CHECKPOINTS.txt (was models_i_care_about.txt)
  envs/                    conda exports (stdcseg = torch 1.1 main env; stdcseg18 = torch 2.4,
                           zip checkpoints + ninja)
  gen_digest.py            resultData/digest/{eval,perclass}_digest.md: compact,
                           self-describing summaries (context header, key tables,
                           caveats) to hand to another Claude session that builds
                           the paper tables. Re-run after any new eval.
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
  for both hosts. thop counts ~1.5x higher and exists for STDC only.
- checkpoints/train_STDC2-Seg-Baseline is in torch>=1.6 zip format, which
  torch 1.1 can't load. ensure_evaluated() routes zip checkpoints to
  envs/stdcseg18 (torch 2.4) automatically. It reproduces runs/RESULTS_early.md exactly.
- The I0 cutoff date "2026-08-09 21:05" in runs/CHECKPOINTS.txt is the
  name of an EMPTY log. I0's actual 60k-iter log is BiSeNet-2026-08-07-18-59-32.log.

Qualitative (DONE: gen_qualitative.py -> resultData/qualitative/, read GUIDE.md there):
  Run with envs/stdcseg python (needs matplotlib/cv2). paper/qual_worker.py caches
  per-image predictions + stats per model (cache/ is gitignored, ~135 MB).
  Fig A (QualitativeData.png): rows 2 Cityscapes, 2 SYNTHIA, 2 RUGD; cols RGB, GT,
    STDC, STDC+Ours, HRNet, HRNet+Ours. Baseline/HI only (I0/HI1, Synthia/Synthia-HI1,
    RUGD/RUGD-HI1, HRNet baseline/HI1). HRNet columns exist for Cityscapes rows only;
    recommended layout = two blocks (Cityscapes x6 cols, SYNTHIA/RUGD x4 cols).
    Images auto-chosen by per-image boundary-band (r=1) accuracy gain (Cityscapes: min
    of STDC and HRNet gains), one per city/sequence; dashed box = window with most
    boundary/thin-class corrections. Override: --pick dataset:image; --strategy median.
  Fig A panels = RGB + GT + predictions only (--extras adds boxes/zooms/diagnostics).
  RUGD row choice (2026-10-02): the default score's thin term is fence+log, and a log row
    contradicted Table III (log flat in band, -2.2 full). --more_candidates rugd:16 [--more_focus
    fence,grass,pole,...] writes resultData/qualitative/more_candidates_rugd[_focus]/ (INDEX.png,
    CANDIDATES.md, candNN folders at x3 = 2064x1650) and leaves fig_qualitative/ untouched. Qualify:
    >=70% of band gain from classes with bnd r1 >= +1 AND full >= 0 in perclass_delta.csv, full acc not
    down. RUGD colours: purple 6600CC = fence, dark maroon 660000 = log.
  Fig B (duet_framework.png): 8 candidate Cityscapes images (top per-image thin-band gain
    HI1 vs I0, <= 3 per city) in fig_framework/candNN__<image>/{full,zoom1,zoom2}; choose via
    INDEX_candidates.png. F01..F10 map 1:1 to the diagram's hatched boxes, extras X1..X8.
    Zooms are 512 GT px squares on a 32-px lattice -> 1536x1536 with exact integer
    enlargement (GT x3, stride-4 x16, stride-8 x32; verified), 300 dpi. Every candidate's
    prediction matches the eval cache 100%. BPM puts 39-77% of OHEM picks in the boundary
    band vs 23-50% for stock OHEM (band = 3-7% of pixels). The diagram's bottom-left
    "Output image I" box is a mislabel; fill it with F10 (S_w), ideally next to X8.
    F06 prediction is MASKED like the GT (black = category not among the 19 evaluated
    classes: parking, ground, static/dynamic, ego vehicle, borders); unmasked = extras/X9.
    Internal tensors (Z, Z', ...) are never masked.

## Repo cleanup (2026-10-03, branch 10-3-Cleanup)
Removed (all still in git history): upstream TensorRT latency tools (latency/, model_stages_trt.py),
upstream scripts/*.sh and images/, the equal-cost Pareto analysis (pareto_*.py/csv/png, plot_pareto.py;
superseded by gen_runtime_memory.py), eval_ablations.py + ablation_results.txt (-> gen_ablation.py),
eval_rugd.py (-> eval_checkpoint.py --dataset rugd), tracked .pyc/.DS_Store, tracked data/ symlinks.
HRNet code is NOT in this repo for now (user decision: push STDC only); paper/host_hrnet.py still
reads ../../hrnet by path. README.md is the public entry point.

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