#!/usr/bin/env bash
# Train one of the paper's runs by name. The checkpoint goes to
# checkpoints/train_STDC2-Seg-<RUN>/ -- the folder paper/common.py's run
# registry expects -- so gen_*.py pick it up without edits.
#
#   scripts/train.sh <RUN> [extra train.py flags]
#
# RUN names (append a suffix for seed replicates, e.g. HI1-rep2; train.py sets no seed):
#   Cityscapes : I0 | Baseline* (stock)   I1 (BPM)   H1 (RSR)   HI1 (RSR + BPM = Ours)
#   ablations  : ABL-M1-noHR ABL-M2-s8 ABL-M3-noRes ABL-M4-noLogit ABL-M5-noGate
#                ABL-L1-unwRank ABL-L2-noOHEM ABL-L2b-noOHEM-w1 ABL-L3-noDetail ABL-L4-noAux
#   sweeps     : SENS-w<weight> (e.g. SENS-w5)   SENS-r<radius> (e.g. SENS-r2)
#   SYNTHIA    : Synthia  Synthia-I1  Synthia-HI1
#   RUGD       : RUGD  RUGD-I1  RUGD-H1  RUGD-HI1
#
# Environment (all optional):
#   PYTHON=/path/to/env/bin/python   interpreter (default: python)
#   GPUS=1                           processes / GPUs. The paper used 1 GPU x 16 images;
#                                    more GPUs multiply the effective batch size.
#   CUDA_VISIBLE_DEVICES             which GPU(s)
# `ninja` and `nvcc` must be on PATH the first time: modules/ (InPlace-ABN) is
# compiled on first import. See README "Setup".
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:?usage: scripts/train.sh <RUN> [extra train.py flags]}
shift
PYTHON=${PYTHON:-python}
GPUS=${GPUS:-1}

OURS="--use_brh True --bnd_weight 3.0"           # RSR + BPM (w=3, r=3)
case "$RUN" in
    Synthia-HI1*) ARGS="--dataset synthia $OURS" ;;
    Synthia-I1*)  ARGS="--dataset synthia --bnd_weight 3.0" ;;
    Synthia*)     ARGS="--dataset synthia" ;;
    RUGD-HI1*)    ARGS="--dataset rugd $OURS" ;;
    RUGD-H1*)     ARGS="--dataset rugd --use_brh True" ;;
    RUGD-I1*)     ARGS="--dataset rugd --bnd_weight 3.0" ;;
    RUGD*)        ARGS="--dataset rugd" ;;
    ABL-M1-noHR*)       ARGS="$OURS --brh_variant no_hr" ;;
    ABL-M2-s8*)         ARGS="$OURS --brh_variant s8" ;;
    ABL-M3-noRes*)      ARGS="$OURS --brh_variant no_residual" ;;
    ABL-M4-noLogit*)    ARGS="$OURS --brh_variant no_logit" ;;
    ABL-M5-noGate*)     ARGS="$OURS --brh_variant no_gate" ;;
    ABL-L1-unwRank*)    ARGS="$OURS --ohem_mode unweighted_rank" ;;
    ABL-L2b-noOHEM-w1*) ARGS="--use_brh True --bnd_weight 1.0 --ohem_mode none" ;;
    ABL-L2-noOHEM*)     ARGS="$OURS --ohem_mode none" ;;
    ABL-L3-noDetail*)   ARGS="$OURS --use_detail_bce False --use_detail_dice False" ;;
    ABL-L4-noAux*)      ARGS="$OURS --use_aux False" ;;
    SENS-w*) W=${RUN#SENS-w}; ARGS="--use_brh True --bnd_weight ${W%%-*}" ;;
    SENS-r*) R=${RUN#SENS-r}; ARGS="$OURS --bnd_radius ${R%%-*}" ;;
    HI1*)    ARGS="$OURS" ;;
    H1*)     ARGS="--use_brh True" ;;
    I1*)     ARGS="--bnd_weight 3.0" ;;
    I0*|Baseline*) ARGS="--bnd_weight 1.0" ;;
    *) echo "unknown run '$RUN' (see the list at the top of $0)" >&2; exit 1 ;;
esac

set -x
"$PYTHON" -m torch.distributed.launch --nproc_per_node="$GPUS" train.py \
    --respath "checkpoints/train_STDC2-Seg-$RUN/" \
    --backbone STDCNet1446 \
    --mode train \
    --n_workers_train 12 \
    --n_workers_val 1 \
    --max_iter 60000 \
    --use_boundary_8 True \
    --pretrain_path checkpoints/STDCNet1446_76.47.tar \
    $ARGS "$@"
