# DUET: Sharper Class Boundaries for Real-Time Semantic Segmentation

Code for the paper *DUET: Coupling Boundary Supervision with Decision Resolution for
Real-Time Semantic Segmentation* (working title, in preparation for IEEE RA-L), by Pragat Wagle.

Real-time segmentation networks get the interiors of objects right and the **edges** wrong:
thin structures (poles, signs, fences) and class boundaries are where most of their errors are.
This repository adds two small, plug-in components to [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg)
(CVPR 2021). Each one fixes a different cause of those errors:

| Component | Code name | What it fixes | How |
|---|---|---|---|
| **RSR**: Residual Sub-grid Refinement | Arm H, `--use_brh` | **Resolution.** STDC-Seg predicts at stride 8 and upsamples 8x, so it cannot draw anything thinner than one 8-px cell. | Upsamples the logits to stride 4, fuses them with the stride-4 detail features the backbone already computes, and adds a residual correction. The residual scale is zero at init, so training starts exactly at the baseline. +0.086M params. |
| **BPM**: Boundary-Priority Mining | Arm I, `--bnd_weight 3` | **Objective.** OHEM cross-entropy weights every pixel equally, so the few boundary pixels hardly influence training. | Up-weights pixels within r px of a ground-truth boundary *before* OHEM's hard-pixel selection, so boundary pixels both count more and survive mining. Loss only: zero parameters and zero inference cost. |

BPM changes what the network is rewarded for; RSR changes what it can physically output. The two
compose: together ("Ours" = RSR + BPM, run name **HI1**) they gain more on thin classes than the
sum of their separate gains.

## Results

Boundary quality is the primary metric: IoU inside the band of pixels within r px of a
ground-truth class boundary ("Bnd"), and the same over thin classes only ("Thin"). Full-image
mIoU is a regression guard. Numbers are val-set IoU (%), copied from
`resultData/perclass/datasets_summary.md`.

| Dataset | Method | Full mIoU | Bnd r=1 | Bnd r=3 | Thin r=1 | Thin r=3 |
|---|---|---|---|---|---|---|
| Cityscapes | STDC-Seg (mean of 3 runs) | 75.82 | 35.80 | 44.91 | 33.16 | 41.84 |
| Cityscapes | STDC-Seg + Ours | **77.25** | **39.77** | **49.82** | **38.62** | **48.39** |
| SYNTHIA | STDC-Seg | 72.56 | 37.41 | 47.64 | 31.40 | 40.45 |
| SYNTHIA | STDC-Seg + Ours | **77.19** | **43.67** | **55.31** | **38.56** | **48.93** |
| RUGD | STDC-Seg | 53.14 | 25.22 | 30.51 | 17.16 | 20.28 |
| RUGD | STDC-Seg + Ours | **54.62** | **26.65** | **33.13** | **19.23** | **23.21** |

STDC2 backbone, 60k iterations, evaluated at input scale 0.75. Run-to-run noise on Cityscapes
(range over 3 baseline seeds): full 1.23, bnd r=1 0.32, thin r=1 0.42 points. The Cityscapes
boundary gain is about 12x that noise. Ablations: `resultData/ablation/ablation.md`.
Cost: `resultData/runtime/runtime_memory.md`. The paper also applies the method to HRNetV2-W48
on Cityscapes; that code is not part of this repository yet.

---

## Setup

### 1. Environment

All results were produced with Python 3.6, PyTorch 1.1.0 and CUDA 10.0 on one NVIDIA T4
(Ubuntu 20.04, gcc 9.4). The full conda export is in `envs/stdcseg.yml`.

```bash
git clone https://github.com/pragatss/STDC-Seg.git
cd STDC-Seg
conda create -n stdcseg python=3.6 -y && conda activate stdcseg
conda install -y pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
python -c "import imageio; imageio.plugins.freeimage.download()"   # SYNTHIA's 16-bit label PNGs
```

* **InPlace-ABN is compiled the first time the model is imported** (`modules/`, via
  `torch.utils.cpp_extension`). This needs `ninja` (in requirements.txt), `nvcc` from a CUDA
  toolkit on `PATH`, and gcc. The build is cached in `/tmp/torch_extensions`. If the first
  import hangs or fails, check `which ninja nvcc`.
* **Optional second env** (`envs/stdcseg18.yml`, torch 2.4): only needed to load checkpoints
  saved in torch >= 1.6 zip format (`train_STDC2-Seg-Baseline` is one). The `gen_*.py` scripts
  switch to it automatically if you set `STDC18_PY=/path/to/that/env/bin/python`.
* `gen_*.py` scripts launch their GPU work in a subprocess. By default they use the python
  you run them with; override this with `STDC_PY=/path/to/python`.

### 2. ImageNet-pretrained backbone

Training starts from the STDC2 ImageNet weights released by the STDC-Seg authors:
`STDCNet1446_76.47.tar` (and `STDCNet813M_73.91.tar` for STDC1) from
[Google Drive](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1?usp=sharing)
or [BaiduYun](https://pan.baidu.com/s/1OdMsuQSSiK1EyNs6_KiFIw) (password `q7dt`). Put them in
`checkpoints/`.

### 3. Datasets

Everything lives under `data/` (git-ignored). Symlinks are fine.

```
data/
  leftImg8bit/{train,val}/<city>/*_leftImg8bit.png        Cityscapes
  gtFine/{train,val}/<city>/*_gtFine_labelIds.png
  SYNTHIA/RGB/{train,val,test}/<id>.png                    SYNTHIA-RAND-CITYSCAPES (prepare_synthia.py)
  SYNTHIA/GT/LABELS/{train,val,test}/<id>.png
  rugd/{train,val,test}/{img,ann,label}/<seq>_<frame>.png  RUGD (prepare_rugd.py writes label/)
```

| Dataset | Train | Val (all reported numbers) | Test (unused) | Classes | Resolution |
|---|---|---|---|---|---|
| [Cityscapes](https://www.cityscapes-dataset.com/) | 2975 | 500 | 1525 | 19 | 1024x2048 |
| [SYNTHIA-RAND-CITYSCAPES](https://synthia-dataset.net/downloads/) | 5593 | 940 | 2867 | 19 Cityscapes trainIds (16 occur in val) | 760x1280 |
| [RUGD](http://rugd.vision/) ([DatasetNinja release](https://datasetninja.com/rugd)) | 4779 | 733 | 1924 | 24 | 550x688 |

**Cityscapes.** Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`, then
link them:

```bash
ln -s /path/to/cityscapes/leftImg8bit data/leftImg8bit
ln -s /path/to/cityscapes/gtFine      data/gtFine
```

**SYNTHIA.** Download *SYNTHIA-RAND-CITYSCAPES (CVPR16)* (9,400 frames). It has no official
split. This project uses a fixed split at Cityscapes' proportions (frame ids interleave across splits), listed in
`splits/synthia/`. Lay it out with:

```bash
python prepare_synthia.py --raw /path/to/RAND_CITYSCAPES      # the folder with RGB/ and GT/LABELS/
```

Labels are converted from SYNTHIA ids to Cityscapes trainIds when they are loaded (`synthia.py`).
Terrain, truck and train never occur in val, so they drop out of every mean.

**RUGD.** This project uses the **DatasetNinja / Supervisely export** of RUGD, not the
colour-coded PNGs from rugd.vision, because `prepare_rugd.py` decodes Supervisely bitmap
annotations. Extract it so that `data/rugd/{train,val,test}/{img,ann}` exist, then decode the
labels once:

```bash
python prepare_rugd.py          # writes data/rugd/{train,val,test}/label/*.png (trainIds, 255 = void)
```

The split is by video sequence (val = `park-8`, `trail-5`). Frame lists are in `splits/rugd/`.
Because val is only two videos, several classes are almost absent. RUGD means therefore
average only the 14 classes with >= 0.1% of val pixels; the unfiltered number is printed
alongside.

---

## Training

Training uses STDC-Seg's `train.py` and launcher unchanged. Our two components are switched on
with flags: `--use_brh True` adds RSR, and `--bnd_weight 3.0` turns on BPM. Every paper run uses
the STDC2 backbone (`STDCNet1446`), 60k iterations, batch 16 on **one GPU**, and the stride-8
detail head. Keep `--respath` as `checkpoints/train_STDC2-Seg-<RUN>/`, because the evaluation
scripts look checkpoints up by that name (`paper/common.py`).

Note: Backbone STDCNet813 denotes STDC1, STDCNet1446 denotes STDC2. All results use STDC2.

* Train the STDC2-Seg baseline (Cityscapes):

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
--nproc_per_node=1 train.py \
--respath checkpoints/train_STDC2-Seg-I0/ \
--backbone STDCNet1446 \
--mode train \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet1446_76.47.tar \
--bnd_weight 1.0
```

* Train STDC2-Seg + Ours (RSR + BPM, Cityscapes):

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
--nproc_per_node=1 train.py \
--respath checkpoints/train_STDC2-Seg-HI1/ \
--backbone STDCNet1446 \
--mode train \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet1446_76.47.tar \
--use_brh True \
--bnd_weight 3.0
```

* Train on SYNTHIA or RUGD: run the same commands with `--dataset synthia` or `--dataset rugd`
  added. The paper's run names are `Synthia` / `Synthia-HI1` and `RUGD` / `RUGD-HI1`, e.g.:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
--nproc_per_node=1 train.py \
--respath checkpoints/train_STDC2-Seg-RUGD-HI1/ \
--backbone STDCNet1446 \
--mode train \
--dataset rugd \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet1446_76.47.tar \
--use_brh True \
--bnd_weight 3.0
```

* Single components and ablations change only the last lines of the "Ours" command:

| Run (`--respath checkpoints/train_STDC2-Seg-<RUN>/`) | Flags after `--pretrain_path ...` |
|---|---|
| `I0` (baseline; `Baseline`, `Baseline2` are seed replicates) | `--bnd_weight 1.0` (= stock OHEM) |
| `I1` (BPM only) | `--bnd_weight 3.0` |
| `H1` (RSR only) | `--use_brh True` |
| `HI1` (Ours) | `--use_brh True --bnd_weight 3.0` |
| `ABL-M1-noHR`, `ABL-M2-s8`, `ABL-M3-noRes`, `ABL-M4-noLogit`, `ABL-M5-noGate` | Ours + `--brh_variant no_hr` / `s8` / `no_residual` / `no_logit` / `no_gate` |
| `ABL-L1-unwRank`, `ABL-L2-noOHEM` | Ours + `--ohem_mode unweighted_rank` / `none` |
| `ABL-L2b-noOHEM-w1` | `--use_brh True --bnd_weight 1.0 --ohem_mode none` |
| `ABL-L3-noDetail` | Ours + `--use_detail_bce False --use_detail_dice False` |
| `ABL-L4-noAux` | Ours + `--use_aux False` |
| `SENS-w2`, `SENS-w5`, `SENS-w8` | `--use_brh True --bnd_weight 2` / `5` / `8` |
| `SENS-r1`, `SENS-r2`, `SENS-r5` | Ours + `--bnd_radius 1` / `2` / `5` |

The full flag reference:

| Flag | Default | Meaning |
|---|---|---|
| `--use_brh True` | False | add RSR. **Must match at evaluation** (`eval_checkpoint.py` detects it from the weights). |
| `--bnd_weight W` | 1.0 | BPM boundary weight (paper: 3). W = 1 is exactly stock OHEM. |
| `--bnd_radius R` | 3 | BPM band radius in px |
| `--brh_variant` | `full` | RSR ablations: `no_hr`, `no_logit`, `no_gate`, `no_residual`, `s8` |
| `--ohem_mode` | `weighted_rank` | BPM ablations: `unweighted_rank`, `none` |
| `--use_aux`, `--use_detail_bce`, `--use_detail_dice` | True | loss-term ablations |
| `--dataset` | `cityscapes` | `synthia`, `rugd` |

We will save the model's params in `model_maxmIOU50.pth` for input resolution 512x1024 and
`model_maxmIOU75.pth` for input resolution 768x1536 (on SYNTHIA/RUGD: scale 0.5 / 0.75 of the
image). Every paper number uses `model_maxmIOU75.pth`.

Notes:
* The paper used `--nproc_per_node=1` (one T4). Upstream STDC-Seg trains on 3 GPUs; the batch
  size is per GPU (`--n_img_per_gpu 16`), so N GPUs multiply the effective batch by N and the
  results are not directly comparable.
* `train.py` sets no seed. For a replicate, train again into a new `--respath`
  (e.g. `train_STDC2-Seg-HI1-rep2/`).
* `ninja` and `nvcc` must be on `PATH` (InPlace-ABN is compiled on first import; see Setup).
* Shortcut: `scripts/train.sh <RUN>` runs the commands above by run name
  (e.g. `scripts/train.sh HI1`, `scripts/train.sh RUGD-HI1`, `scripts/train.sh ABL-M3-noRes`)
  and prints the full command. The exact commands used for the paper, with their logs, are in
  `runs/`.

## Evaluation

### One or more checkpoints, any dataset

```bash
python eval_checkpoint.py --dataset cityscapes --ckpt checkpoints/train_STDC2-Seg-I0/pths/model_maxmIOU75.pth \
                                                      checkpoints/train_STDC2-Seg-HI1/pths/model_maxmIOU75.pth
python eval_checkpoint.py --dataset synthia --ckpt checkpoints/train_STDC2-Seg-Synthia-HI1/pths/model_maxmIOU75.pth
python eval_checkpoint.py --dataset rugd    --ckpt checkpoints/train_STDC2-Seg-RUGD-HI1/pths/model_maxmIOU75.pth --per_class
```

The script prints full mIoU, Bnd r=1/r=3 and Thin r=1/r=3, plus deltas against the first
checkpoint. Options: `--per_class` (per-class IoU and pixel support), `--out results.json`
(everything), `--size32` (a 32-divisible input instead of scale 0.75; see Protocol).
It uses the same model build, inference and metric code as the paper tables. These commands
reproduce the cached numbers in `resultData/cache/eval/` exactly (checked for Cityscapes
I0/HI1/M2, SYNTHIA HI1 and RUGD HI1).

### The paper tables and figures

The generators score every registered run, cache the results, and write plain CSV/MD/JSON to
`resultData/` (see `resultData/README.md` for every file and column). Checkpoints are named in
one place: `paper/common.py` (`STDC_RUNS`).

| Command | Output |
|---|---|
| `python gen_perclass.py --hosts stdc` | per-class tables per dataset + cross-dataset summary (`resultData/perclass/`) |
| `python gen_ablation.py` | ablation table vs the full model, with deltas in noise-floor units (`resultData/ablation/`) |
| `python gen_runtime_memory.py --hosts stdc` | params, GMACs, train/infer throughput, peak memory (`resultData/runtime/`) |
| `python gen_qualitative.py --datasets synthia rugd` | qualitative panels (`resultData/qualitative/`, read `GUIDE.md` there). The Cityscapes rows and the framework-figure panels also need the HRNet checkpoints. `--more_candidates rugd:16` renders more candidate images. |
| `python tools/make_figure_assets.py --ckpt <HI1 pth> --use_brh --image <leftImg8bit png> --label <labelIds png>` | print-quality framework-figure tiles for one image (`resultData/figure_assets/`) |
| `python gen_digest.py` | compact summaries of all results (`resultData/digest/`) |

Runs whose checkpoint is missing are skipped and marked `pending`. `python check_ablation.py`
is a quick sanity check of every RSR variant and BPM mode. It prints output shapes, parameter
counts, whether each variant starts identical to the baseline, and the BPM loss next to stock
OHEM (w = 1 must match). Run it after changing `models/` or `loss/`.

### Protocol (read before comparing numbers)

* **Input**: val images bilinearly resized to scale 0.75 (Cityscapes 768x1536). The prediction
  is upsampled back and scored against the **full-resolution** ground truth.
* **Boundary band**: pixels within r px (r = 1, 3; max-pool dilation) of a pixel whose
  ground-truth neighbour has a different class, ignoring void. It works for any label set, so
  each dataset is scored in its own classes.
* **Thin classes**: Cityscapes/SYNTHIA: pole, traffic light, traffic sign, rider, motorcycle,
  bicycle. RUGD: pole, sign, fence, bicycle, log (after the support filter, only fence and log
  remain).
* **Noise**: `train.py` sets no seed. Compare against the 3-run baseline mean and report effects
  relative to the noise floor. Full-image mIoU noise (1.2 pt) is larger than every single
  component's effect, so full mIoU is only a guard.
* **32-divisible inputs**: scale 0.75 gives 570x960 on SYNTHIA and 412x516 on RUGD. These are
  not multiples of the backbone's stride 32, which costs every model accuracy. `--size32`
  re-scores at 576x960 / 416x512. The Ours-vs-baseline deltas hold (`resultData/perclass/*/scale_check.csv`).

---

## Repository layout

```
train.py                 training (STDC-Seg + RSR/BPM flags)
evaluation.py            boundary metric reference implementation; load_net() checks RSR flags vs weights
eval_checkpoint.py       evaluate any checkpoint on cityscapes / synthia / rugd
models/model_stages.py   STDC-Seg network + BoundaryRefine (RSR) and its ablation variants
loss/loss.py             OhemCELoss + BoundaryOhemCELoss (BPM), boundary_weight_map()
loss/detail_loss.py      STDC-Seg's detail-head loss (unchanged)
nets/, modules/          STDC backbone, InPlace-ABN (from STDC-Seg)
cityscapes.py synthia.py rugd.py transform.py      datasets + augmentation
prepare_synthia.py prepare_rugd.py splits/          dataset preparation and split lists
scripts/train.sh         train any paper run by name
paper/                   shared evaluation/runtime code for the generators (run registry in common.py)
gen_*.py tools/          generators for every paper table and figure
resultData/              generated numbers and figure panels (the paper's data)
ref_images/              reference layouts of the paper tables/figures, framework diagram (.pptx)
runs/                    exact commands and logs of every run, checkpoint map (runs/README.md)
envs/                    conda environment exports
claude.md                detailed research notes: design, protocol, decisions, known pitfalls
```

## Continuing this work

* Start with **`claude.md`**. It records every design decision, the metric protocol, the
  checkpoint quirks, and pitfalls that have each cost real time. Despite the name, it is meant
  for human readers too.
* `runs/CHECKPOINTS.txt` maps every checkpoint to its paper run and lists what is still
  **pending**: ablations M3–M5, L2b, L3, L4, the bnd_weight/bnd_radius sweeps, and replicates of
  HI1. Each one is a training command from the Training section (or `scripts/train.sh <name>`),
  followed by `python gen_ablation.py`.
* Trained checkpoints are **not** in git (~100 GB). Put them under `checkpoints/` with the
  folder names in `paper/common.py`.
* To add a run: train it (Training section), register it in `paper/common.py` `STDC_RUNS`,
  and re-run the relevant `gen_*.py`.
* Never change `--use_brh` / `--brh_variant` between training and evaluation. `load_net()`
  raises on a mismatch instead of silently building the wrong model.

## Citation

If you use this code, please cite STDC-Seg, which this repository builds on (the paper above
is in preparation):

```
@InProceedings{Fan_2021_CVPR,
    author    = {Fan, Mingyuan and Lai, Shenqi and Huang, Junshi and Wei, Xiaoming and Chai, Zhenhua and Luo, Junfeng and Wei, Xiaolin},
    title     = {Rethinking BiSeNet for Real-Time Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021},
    pages     = {9716-9725}
}
```

Datasets: Cityscapes (Cordts et al., CVPR 2016), SYNTHIA (Ros et al., CVPR 2016), RUGD
(Wigness et al., IROS 2019).

## Acknowledgements and license

Built on [STDC-Seg](https://github.com/MichaelFan01/STDC-Seg) by Mingyuan Fan et al., whose
training and evaluation code derives from [BiSeNet](https://github.com/CoinCheung/BiSeNet).
InPlace-ABN is from [mapillary/inplace_abn](https://github.com/mapillary/inplace_abn).
Released under the MIT License (see `LICENSE`; the original STDC-Seg copyright is retained).
