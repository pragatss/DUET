#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Runtime and memory table (cf. ref_images/quantative/RuntimeAndMemory.png):
training throughput + peak GPU memory, inference throughput + peak GPU
memory, plus params and GMACs, for

    STDC-Seg, + BPM only, + RSR only, + Ours      (I0 / I1 / H1 / HI1)
    HRNet-W48, HRNet-W48 + Ours                   (baseline / HI1)

Measurement happens in paper/runtime_worker.py (one subprocess per host, in
that host's env; see its docstring for the protocol). This driver also
parses the TRAINING LOGS for the as-trained wall-clock throughput (includes
data loading and periodic val), which is what a reader would compare to
their own training runs.

Caveat: this is a T4 without clock locking, so absolute it/s and img/s move a
few percent with thermal state. The interleaved rounds protect the RELATIVE
numbers; params/GMACs are exact and are the primary cost claim.

USAGE (any python)
    python gen_runtime_memory.py                  # measure both hosts, ~25 min
    python gen_runtime_memory.py --hosts stdc
    python gen_runtime_memory.py --reuse          # rebuild tables from resultData/runtime/raw/
    python gen_runtime_memory.py --rounds 10      # quicker, noisier (default 30)

OUTPUT  resultData/runtime/
    runtime_memory.csv     one row per method (the paper table + extra columns)
    runtime_memory.md      quick-look markdown
    raw/<host>.json        worker output incl. every per-round sample and GPU temp/clock
"""
import argparse
import glob
import json
import os.path as osp
import re
import statistics
import subprocess

from paper import common

# training logs, for the as-trained throughput column
STDC_LOGDIR = {'I0': 'I0', 'I1': 'I1', 'H1': 'H1', 'HI1': 'HI1'}
STDC_MSG_ITER = 50       # train.py logs every 50 iters; 'time' = seconds for those 50


def _largest(paths, pattern):
    best, best_n = None, 0
    for p in paths:
        with open(p, errors='ignore') as f:
            n = sum(1 for l in f if pattern in l)
        if n > best_n:
            best, best_n = p, n
    return best


def log_throughput(host, key):
    """median it/s over the run's log (the log with the most progress lines).
    Median, so the intervals that include periodic validation drop out."""
    if host == 'stdc':
        d = osp.join(common.STDC_ROOT, 'checkpoints', 'train_STDC2-Seg-%s' % STDC_LOGDIR.get(key, key))
        log = _largest(glob.glob(osp.join(d, '*.log')), ' it: ')
        if not log:
            return None, None
        ts = [float(m.group(1)) for m in
              (re.search(r' it: \d+/\d+,.*time: ([\d.]+)', l) for l in open(log, errors='ignore')) if m]
        return (STDC_MSG_ITER / statistics.median(ts) if ts else None), osp.relpath(log, common.STDC_ROOT)
    d = osp.join(common.HRNET_ROOT, common.HRNET_RUNS[key]['logdir'])
    log = _largest(glob.glob(osp.join(d, '*.log')), 'Iter:[')
    if not log:
        return None, None
    # HRNet 'Time' is the running mean s/iter within the epoch; skip Iter 0
    ts = [float(m.group(1)) for m in
          (re.search(r'Iter:\[(?!0/)\d+/\d+\], Time: ([\d.]+)', l) for l in open(log, errors='ignore')) if m]
    return (1.0 / statistics.median(ts) if ts else None), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hosts', nargs='+', default=['stdc', 'hrnet'], choices=('stdc', 'hrnet'))
    ap.add_argument('--reuse', action='store_true', help='do not measure; use existing raw/<host>.json')
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--force', action='store_true', help='measure even if the GPU is busy')
    args = ap.parse_args()

    od = common.outdir('runtime')
    raw = {}
    for host in args.hosts:
        f = osp.join(od, 'raw', '%s.json' % host)
        if not args.reuse:
            cmd = [common.ENV_PY[host], '-m', 'paper.runtime_worker', '--host', host,
                   '--out', f, '--rounds', str(args.rounds)] + (['--force'] if args.force else [])
            print('[%s] measuring: %s' % (host, ' '.join(cmd)))
            subprocess.check_call(cmd, cwd=common.STDC_ROOT)
        if osp.isfile(f):
            with open(f) as fh:
                raw[host] = json.load(fh)
        else:
            print('[%s] no %s; skipped' % (host, f))

    header = ['Method', 'Host',
              'Train throughput (it/s)', 'Train it/s std', 'Train throughput (img/s)',
              'Train GPU mem peak alloc (GB)', 'Train GPU mem reserved (GB)',
              'Infer throughput (img/s)', 'Infer img/s std', 'Infer latency (ms)',
              'Infer GPU mem peak alloc (GB)', 'Infer GPU mem reserved (GB)',
              'Params (M)', 'GMACs', 'thop GFLOPs',
              'd Train it/s vs baseline (% paired)', 'd Infer img/s vs baseline (% paired)',
              'd Params (M)', 'd GMACs (%)',
              'As-trained it/s (log median)', 'Train batch (NCHW)', 'Infer input (model)', 'GPU', 'Log']
    rows, md = [], []
    for host, d in raw.items():
        res = d['results']
        base = res[0]                      # CONFIGS lists the baseline first
        for r in res:
            s, tr, inf = r['static'], r['train'], r['infer']
            log_its, log = log_throughput(host, r['key'])
            rel = lambda a, b: '' if r is base else '%+.1f' % (100.0 * (a - b) / b)
            paired = lambda ph: '' if r is base else '%+.1f' % (100.0 * (ph['ratio_vs_baseline'] - 1))
            rows.append([
                r['name'], host,
                '%.3f' % tr['it_s']['pooled'], '%.3f' % tr['it_s']['std'], '%.2f' % tr['img_s'],
                '%.2f' % tr['peak_alloc_gb'], '%.2f' % tr['peak_reserved_gb'],
                '%.2f' % inf['img_s']['pooled'], '%.2f' % inf['img_s']['std'], '%.1f' % inf['ms'],
                '%.2f' % inf['peak_alloc_gb'], '%.2f' % inf['peak_reserved_gb'],
                '%.3f' % s['params_m'], '%.2f' % s['gmacs'],
                '' if s.get('thop_gflops') is None else '%.2f' % s['thop_gflops'],
                paired(tr), paired(inf),
                '' if r is base else '%+.3f' % (s['params_m'] - base['static']['params_m']),
                rel(s['gmacs'], base['static']['gmacs']),
                '' if log_its is None else '%.3f' % log_its,
                'x'.join(str(v) for v in d['meta']['train_batch_shape']),
                'x'.join(str(v) for v in s['input']),
                d['meta']['gpu'], log or ''])
            md.append([r['name'], '%.3f it/s' % tr['it_s']['pooled'], '%.2f GB' % tr['peak_alloc_gb'],
                       '%.2f img/s' % inf['img_s']['pooled'], '%.2f GB' % inf['peak_alloc_gb'],
                       '%.2fM' % s['params_m'], '%.1f' % s['gmacs']])
    common.write_csv(osp.join(od, 'runtime_memory.csv'), header, rows)
    common.write_md(osp.join(od, 'runtime_memory.md'), [
        '# Runtime and memory', '',
        'generated %s by gen_runtime_memory.py; %s' % (
            common.now(), '; '.join('%s: %s, torch %s, measured %s' % (h, d['meta']['gpu'], d['meta']['torch'],
                                                                    d['meta']['measured_at'])
                                    for h, d in raw.items())), '',
        'Training = one full train iteration (fwd + all losses + bwd + SGD) at the host training batch; '
        'inference = batch 1, 1024x2048 image -> label map. Throughput pooled over interleaved rounds '
        'after a heat soak; d% = median per-round ratio to the host baseline. Peak memory = torch '
        'max_memory_allocated, isolated pass, cudnn.benchmark off.', ''] +
        common.md_table(['Method', 'Train', 'Train mem', 'Infer', 'Infer mem', 'Params', 'GMACs'], md))


if __name__ == '__main__':
    main()
