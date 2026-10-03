# runs/ — what was actually run

These are the original records, kept as they were written during the project.
They contain absolute paths from the original machine (`/home/husky/...`) and
pasted terminal output. To re-run anything, use `scripts/train.sh <RUN>`
(same flags, portable). To score checkpoints, use `eval_checkpoint.py` or the
`gen_*.py` generators.

| File | What it is |
|---|---|
| `CHECKPOINTS.txt` | **Source of truth** for which checkpoint is which paper run (Cityscapes main + ablations, SYNTHIA, RUGD), with each run's training-log mIoU, plus the runs that are still pending. `paper/common.py` (`STDC_RUNS`) holds the same mapping in code. |
| `cityscapes_commands.txt` | Exact commands + final log lines for the Cityscapes runs (I0, I1, Baseline2, H1, HI1). |
| `ablation_commands.txt` | Exact commands for the ablations trained so far (M1, M2, L1, L2). |
| `synthia_commands.txt` | SYNTHIA baseline / BPM / Ours commands. |
| `rugd_commands.txt` | RUGD baseline / Ours commands. It mentions `eval_rugd.py`; that script is now `eval_checkpoint.py --dataset rugd` (identical numbers). |
| `RESULTS_early.md` | Early Cityscapes write-up (n=2 noise floor, stacking analysis, `res_scale` sign flip). The current numbers are in `resultData/`. |

Known irregularities in these records:
- `train_STDC2-Seg-Baseline` predates the cutoff and its command was not recorded. It was saved
  in torch >= 1.6 zip format, so it has to be evaluated with a torch >= 1.6 env.
- `train_STDC2-Seg-Synthia/pths/model_maxmIOU75.pth` was overwritten by an aborted restart. The
  paper uses `model_iter58000_mIOU50_0.6888_mIOU75_0.7256.pth`.
- `train_STDC2-Seg-Synthia-HI1/` holds two runs. The no-RSR one is the SYNTHIA BPM-only run
  (registered as `Synthia-I1`). `model_maxmIOU75.pth` there is the real HI1.
