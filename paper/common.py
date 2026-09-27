#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Shared plumbing for the paper-data generators.

Everything here is stdlib-only at import time (torch is imported inside the
functions that need it), so the gen_*.py drivers run under any interpreter.
GPU work happens in worker subprocesses, one per host, each launched with the
conda env that host was trained in:

    stdc  -> envs/stdcseg  (torch 1.1.0)   runs evaluation.py's model code
    hrnet -> envs/hrnet    (torch 1.1.0)   runs HRNet's lib/ via sys.path

They cannot share a process: both repos have a top-level `models` package,
and whichever is imported first shadows the other.
"""
import csv
import datetime
import json
import os
import os.path as osp
import subprocess
import sys
from collections import OrderedDict

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
STDC_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
# HRNet checkpoints/configs are read in place, relative to this repo, so no
# new code has to live in the (untracked) HRNet checkout.
HRNET_ROOT = osp.normpath(os.environ.get(
    'HRNET_ROOT', osp.join(STDC_ROOT, '..', '..', 'hrnet', 'HRNet-Semantic-Segmentation')))
RESULT_DIR = osp.join(STDC_ROOT, 'resultData')
EVAL_CACHE = osp.join(RESULT_DIR, 'cache', 'eval')

ENV_PY = {
    'stdc': os.environ.get('STDC_PY', '/home/husky/anaconda3/envs/stdcseg/bin/python'),
    'hrnet': os.environ.get('HRNET_PY', '/home/husky/anaconda3/envs/hrnet/bin/python'),
}
# checkpoints written by torch >= 1.6 (zip format, e.g. train_STDC2-Seg-Baseline)
# cannot be read by torch 1.1; those are evaluated under this env instead.
ZIP_PY = os.environ.get('STDC18_PY', '/home/husky/anaconda3/envs/stdcseg18/bin/python')

# ---------------------------------------------------------------------------
# Cityscapes classes / metric constants
# ---------------------------------------------------------------------------
# same order and short names as evaluation.py
CLASS_NAMES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'tlight',
               'tsign', 'veg', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
               'bus', 'train', 'moto', 'bicycle']
# column headers as they appear in the paper tables
PAPER_CLASS_NAMES = ['Road', 'S.walk', 'Build.', 'Wall', 'Fence', 'Pole', 'Tr.Light',
                     'Sign', 'Veget.', 'Terrain', 'Sky', 'Person', 'Rider', 'Car',
                     'Truck', 'Bus', 'Train', 'M.bike', 'Bike']
THIN = [5, 6, 7, 12, 17, 18]   # pole, tlight, tsign, rider, moto, bicycle
RADII = (1, 3)
METRICS = ('full', 'bnd_r1', 'bnd_r3', 'thin_r1', 'thin_r3')
# fallback only -- gen_ablation.py recomputes this from the baseline replicates
# (range across I0 / Baseline / Baseline2) whenever all three are on disk.
NOISE_FLOOR = dict(full=0.0123, bnd_r1=0.0032, bnd_r3=0.0048, thin_r1=0.0041, thin_r3=0.0055)

# ---------------------------------------------------------------------------
# run registry -- the single place checkpoints are named
# ---------------------------------------------------------------------------
def _stdc_ckpt(d):
    return osp.join('checkpoints', 'train_STDC2-Seg-%s' % d, 'pths', 'model_maxmIOU75.pth')


def _S(d, use_brh, desc, variant='full'):
    return dict(ckpt=_stdc_ckpt(d), use_brh=use_brh, brh_variant=variant, desc=desc)


# STDC-Seg, Cityscapes. use_brh / brh_variant MUST match training (see
# evaluation.load_net). Every entry is STDCNet1446, 60k iter, use_boundary_8.
STDC_RUNS = OrderedDict([
    ('I0',        _S('I0', False, 'baseline, bnd_weight=1.0 (lambda=1 control); paper baseline')),
    ('Baseline2', _S('Baseline2', False, 'baseline seed replicate (same config as I0)')),
    ('Baseline',  _S('Baseline', False, 'baseline seed replicate (pre-cutoff, command not recorded)')),
    ('I1',        _S('I1', False, 'BPM only: bnd_weight=3.0 (Arm I)')),
    ('H1',        _S('H1', True, 'RSR only: use_brh (Arm H)')),
    ('HI1',       _S('HI1', True, 'RSR + BPM: use_brh, bnd_weight=3.0 (headline)')),
    # module (Arm H) ablations, all otherwise == HI1
    ('ABL-M1-noHR',       _S('ABL-M1-noHR', True, 'no high-res features', 'no_hr')),
    ('ABL-M2-s8',         _S('ABL-M2-s8', True, 'refine at stride 8', 's8')),
    ('ABL-M3-noRes',      _S('ABL-M3-noRes', True, 'no residual', 'no_residual')),
    ('ABL-M4-noLogit',    _S('ABL-M4-noLogit', True, 'no logit input', 'no_logit')),
    ('ABL-M5-noGate',     _S('ABL-M5-noGate', True, 'no gate', 'no_gate')),
    # loss (Arm I) ablations, all otherwise == HI1
    ('ABL-L1-unwRank',    _S('ABL-L1-unwRank', True, 'unweighted OHEM rank')),
    ('ABL-L2-noOHEM',     _S('ABL-L2-noOHEM', True, 'no OHEM, w=3')),
    ('ABL-L2b-noOHEM-w1', _S('ABL-L2b-noOHEM-w1', True, 'no OHEM, w=1')),
    ('ABL-L3-noDetail',   _S('ABL-L3-noDetail', True, 'no detail loss')),
    ('ABL-L4-noAux',      _S('ABL-L4-noAux', True, 'no aux heads')),
    # sensitivity sweeps, all otherwise == HI1 (HI1 itself is w=3, r=3)
    ('SENS-w2', _S('SENS-w2', True, 'bnd_weight=2')),
    ('SENS-w5', _S('SENS-w5', True, 'bnd_weight=5')),
    ('SENS-w8', _S('SENS-w8', True, 'bnd_weight=8')),
    ('SENS-r1', _S('SENS-r1', True, 'bnd_radius=1')),
    ('SENS-r2', _S('SENS-r2', True, 'bnd_radius=2')),
    ('SENS-r5', _S('SENS-r5', True, 'bnd_radius=5')),
])
BASELINE_REPLICATES = ('I0', 'Baseline2', 'Baseline')

# HRNetV2-W48, Cityscapes, 120 epochs, OHEM on in BOTH arms (see claude.md
# trap 5). best.pth = best val mIoU during training, the analogue of STDC's
# model_maxmIOU75.pth. log_best is what the training log reported, used as a
# sanity check on the re-evaluation.
HRNET_RUNS = OrderedDict([
    ('baseline', dict(cfg='experiments/cityscapes/arm_baseline_w48.yaml',
                      ckpt='output/hrnet_baseline/cityscapes/arm_baseline_w48/best.pth',
                      logdir='output/hrnet_baseline/cityscapes/arm_baseline_w48',
                      log_best=0.7562, desc='HRNetV2-W48 + OHEM')),
    ('HI1',      dict(cfg='experiments/cityscapes/arm_HI1_w48.yaml',
                      ckpt='output/hrnet_HI1/cityscapes/arm_HI1_w48/best.pth',
                      logdir='output/hrnet_HI1/cityscapes/arm_HI1_w48',
                      log_best=0.7722, desc='HRNetV2-W48 + RSR + BPM')),
])

HOST_RUNS = {'stdc': STDC_RUNS, 'hrnet': HRNET_RUNS}
HOST_ROOT = {'stdc': STDC_ROOT, 'hrnet': HRNET_ROOT}

# eval protocol per host. Each host is scored with its OWN native protocol
# (what its training-time val used), and boundary bands are always computed
# on the full-resolution 1024x2048 GT, so the metric definition is identical
# across hosts; only within-host comparisons are ever made.
PROTOCOL = {
    'stdc': dict(input='bilinear resize to scale 0.75 (768x1536), align_corners=True',
                 scale=0.75, batchsize=5, radii=list(RADII)),
    'hrnet': dict(input='full resolution 1024x2048, align_corners=False',
                  scale=1.0, batchsize=1, radii=list(RADII)),
}


def ckpt_path(host, key):
    return osp.join(HOST_ROOT[host], HOST_RUNS[host][key]['ckpt'])


def available(host, key):
    return osp.isfile(ckpt_path(host, key))


# ---------------------------------------------------------------------------
# boundary-band metric (torch 1.1 safe)
# ---------------------------------------------------------------------------
def gt_boundary(label, ignore=255):
    """Verbatim copy of evaluation.gt_boundary. Duplicated rather than
    imported because importing evaluation.py pulls in STDC's `models`
    package, which collides with HRNet's in the hrnet worker."""
    import torch
    N, H, W = label.shape
    bnd = torch.zeros((N, H, W), device=label.device)
    d = label[:, :, 1:] != label[:, :, :-1]
    v = (label[:, :, 1:] != ignore) & (label[:, :, :-1] != ignore)
    e = (d & v).float()
    bnd[:, :, 1:] = torch.max(bnd[:, :, 1:], e)
    bnd[:, :, :-1] = torch.max(bnd[:, :, :-1], e)
    d = label[:, 1:, :] != label[:, :-1, :]
    v = (label[:, 1:, :] != ignore) & (label[:, :-1, :] != ignore)
    e = (d & v).float()
    bnd[:, 1:, :] = torch.max(bnd[:, 1:, :], e)
    bnd[:, :-1, :] = torch.max(bnd[:, :-1, :], e)
    return bnd


def dilate(bnd, r):
    """Verbatim copy of evaluation.dilate."""
    import torch.nn.functional as F
    k = 2 * r + 1
    return (F.max_pool2d(bnd.unsqueeze(1), kernel_size=k, stride=1,
                         padding=r).squeeze(1) > 0.5)


class SegMeter(object):
    """Accumulates one confusion matrix per region (full image, and the
    boundary band at each radius). Per-class I and U fall out of the matrix
    and are the same counts evaluation.evaluate_boundary sums class by class,
    so results match its numbers; float64 accumulation instead of float32."""

    def __init__(self, n_classes=19, radii=RADII, ignore=255):
        import torch
        self.n = n_classes
        self.radii = tuple(radii)
        self.ignore = ignore
        keys = ['full'] + ['bnd_r%d' % r for r in self.radii]
        self.hist = OrderedDict((k, torch.zeros(n_classes * n_classes, dtype=torch.float64).cuda())
                                for k in keys)

    def _add(self, key, pred, label, mask):
        import torch
        idx = label[mask] * self.n + pred[mask]
        self.hist[key] += torch.bincount(idx, minlength=self.n ** 2).double()

    def update(self, pred, label):
        """pred, label: [N,H,W] long cuda tensors at GT resolution."""
        valid = label != self.ignore
        self._add('full', pred, label, valid)
        bnd = gt_boundary(label, self.ignore)
        for r in self.radii:
            self._add('bnd_r%d' % r, pred, label, dilate(bnd, r) & valid)

    def result(self):
        """-> dict with summary metrics (evaluation.py conventions) and the
        per-class breakdown for every region."""
        res = OrderedDict()
        regions = OrderedDict()
        for key, h in self.hist.items():
            h = h.view(self.n, self.n).cpu()
            tp = h.diag()
            gt = h.sum(1)
            union = gt + h.sum(0) - tp
            iou = tp / (union + 1e-6)          # evaluation.py's epsilon
            present = union > 0
            regions[key] = OrderedDict([
                ('miou', float(iou[present].mean())),
                ('thin', float(iou[THIN].mean())),
                ('iou', [float(x) for x in iou]),
                ('gt_pixels', [int(x) for x in gt]),
                ('union_pixels', [int(x) for x in union]),
            ])
        res['full'] = regions['full']['miou']
        for r in self.radii:
            res['bnd_r%d' % r] = regions['bnd_r%d' % r]['miou']
            res['thin_r%d' % r] = regions['bnd_r%d' % r]['thin']
        res['thin_full'] = regions['full']['thin']
        res['regions'] = regions
        res['class_names'] = CLASS_NAMES
        return res


# ---------------------------------------------------------------------------
# eval cache + worker dispatch
# ---------------------------------------------------------------------------
def fingerprint(host, key):
    p = ckpt_path(host, key)
    st = os.stat(p)
    fp = dict(host=host, key=key, ckpt=p, mtime=int(st.st_mtime), size=st.st_size,
              protocol=PROTOCOL[host])
    fp.update({k: v for k, v in HOST_RUNS[host][key].items() if k in ('use_brh', 'brh_variant', 'cfg')})
    return fp


def cache_file(host, key):
    return osp.join(EVAL_CACHE, '%s__%s.json' % (host, key))


def load_cached(host, key):
    """Cached eval result if it exists AND matches the checkpoint on disk."""
    f = cache_file(host, key)
    if not (osp.isfile(f) and available(host, key)):
        return None
    with open(f) as fh:
        d = json.load(fh)
    return d if d.get('fingerprint') == fingerprint(host, key) else None


def ensure_evaluated(host, keys, force=False):
    """Evaluate (host, key) for every key not already cached, in a worker
    subprocess under that host's env. Returns {key: result} for every key
    with a checkpoint on disk; missing checkpoints are simply absent."""
    import zipfile
    keys = [k for k in keys if available(host, k)]
    todo = [k for k in keys if force or load_cached(host, k) is None]
    by_py = OrderedDict()
    for k in todo:
        py = ZIP_PY if zipfile.is_zipfile(ckpt_path(host, k)) else ENV_PY[host]
        by_py.setdefault(py, []).append(k)
    for py, ks in by_py.items():
        cmd = [py, '-m', 'paper.eval_worker', '--host', host, '--keys'] + ks
        print('[%s] evaluating %d run(s) with %s: %s' % (host, len(ks), py, ' '.join(ks)))
        subprocess.check_call(cmd, cwd=STDC_ROOT)
    out = OrderedDict()
    for k in keys:
        d = load_cached(host, k)
        if d is None:
            raise RuntimeError('worker finished but no valid cache for %s/%s' % (host, k))
        out[k] = d
    return out


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def outdir(*parts):
    d = osp.join(RESULT_DIR, *parts)
    if not osp.isdir(d):
        os.makedirs(d)
    return d


def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')


def pct(x, nd=2):
    """IoU fraction -> percent string for table CSVs ('' for missing)."""
    return '' if x is None else ('%.*f' % (nd, 100.0 * x))


def write_csv(path, header, rows):
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print('  wrote %s' % osp.relpath(path, STDC_ROOT))


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print('  wrote %s' % osp.relpath(path, STDC_ROOT))


def write_md(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('  wrote %s' % osp.relpath(path, STDC_ROOT))


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return out


def mean_result(results):
    """Element-wise mean of several SegMeter results (summary + per-class)."""
    n = float(len(results))
    avg = OrderedDict()
    for m in METRICS + ('thin_full',):
        avg[m] = sum(r[m] for r in results) / n
    regions = OrderedDict()
    for reg in results[0]['regions']:
        regions[reg] = OrderedDict([
            ('miou', sum(r['regions'][reg]['miou'] for r in results) / n),
            ('thin', sum(r['regions'][reg]['thin'] for r in results) / n),
            ('iou', [sum(r['regions'][reg]['iou'][c] for r in results) / n
                     for c in range(len(CLASS_NAMES))]),
            ('gt_pixels', results[0]['regions'][reg]['gt_pixels']),
        ])
    avg['regions'] = regions
    avg['class_names'] = CLASS_NAMES
    return avg
