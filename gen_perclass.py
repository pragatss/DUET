#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Per-class IoU tables (cf. ref_images/quantative/PerClass.png) and a
cross-dataset summary, full-image mIoU and boundary metrics side by side.

    cityscapes : STDC-Seg (baseline = mean of 3 seed replicates), STDC-Seg + Ours,
                 HRNet-W48, HRNet-W48 + Ours
    synthia    : STDC-Seg, STDC-Seg + Ours (+ BPM-only if its checkpoint exists)
    rugd       : STDC-Seg, STDC-Seg + Ours

Regions: full image (mIoU = regression guard) and the boundary band at r=1 and
r=3 (the primary metric). Each dataset is scored in its OWN classes (no mapping
to Cityscapes); the thin set and rare-class handling per dataset live in
paper/common.py DATASETS:
  * SYNTHIA uses Cityscapes trainIds; terrain/truck/train never occur in val
    and drop out of every mean.
  * RUGD means use only classes with >= 0.1% of val GT pixels (the support
    filter, as in eval_checkpoint.py); 'mIoU (all classes)' is the unfiltered number train.py logs.
For SYNTHIA and RUGD, scale 0.75 is not 32-divisible, so each checkpoint is
also scored at a nearby 32-divisible size (scale_check.csv) to confirm the
conclusions don't depend on that resize.

Checkpoints are evaluated once and cached in resultData/cache/eval/.

USAGE (any python; GPU work runs in each host's conda env)
    python gen_perclass.py                              # all datasets
    python gen_perclass.py --datasets synthia rugd
    python gen_perclass.py --hosts stdc                 # Cityscapes: skip HRNet
    python gen_perclass.py --stdc_baseline I0           # Cityscapes: single-run baseline
    python gen_perclass.py --force                      # re-evaluate even if cached

OUTPUT  resultData/perclass/  (IoU in %, see resultData/README.md)
    datasets_summary.csv / .md     every dataset x method: full mIoU (+ all-class), bnd r1/r3, thin r1/r3, deltas
    <dataset>/perclass_{full,bnd_r1,bnd_r3}.csv   one table per region
    <dataset>/perclass_delta.csv   Ours - baseline, per class
    <dataset>/perclass_long.csv    tidy: one row per (method, region, class)
    <dataset>/perclass_summary.json
    <dataset>/perclass.md
    <dataset>/scale_check.csv      (synthia, rugd) scale 0.75 vs 32-divisible input
"""
import argparse
import os.path as osp
from collections import OrderedDict

from paper import common
from paper.common import DATASETS, pct

REGIONS = OrderedDict([('full', 'Full image'), ('bnd_r1', 'Boundary band r=1'),
                       ('bnd_r3', 'Boundary band r=3')])
SUMMARY_METRICS = OrderedDict([('full', 'Full mIoU'), ('full_all', 'Full mIoU (all classes)'),
                               ('bnd_r1', 'Bnd r=1'), ('bnd_r3', 'Bnd r=3'),
                               ('thin_r1', 'Thin r=1'), ('thin_r3', 'Thin r=3')])


def _row(host, method, row_type, keys, res, note):
    return dict(host=host, method=method, row_type=row_type, keys=keys, res=res, note=note)


def rows_cityscapes(hosts, stdc_baseline, force):
    rows = []
    if 'stdc' in hosts:
        ev = common.ensure_evaluated('stdc', list(common.BASELINE_REPLICATES) + ['HI1'], force)
        reps = [k for k in common.BASELINE_REPLICATES if k in ev]
        if stdc_baseline == 'mean' and len(reps) > 1:
            rows.append(_row('stdc', 'STDC-Seg', 'main', reps, common.mean_result([ev[k] for k in reps]),
                             'mean of %d seed replicates' % len(reps)))
        elif 'I0' in ev:
            rows.append(_row('stdc', 'STDC-Seg', 'main', ['I0'], ev['I0'], 'single run (I0)'))
        if 'HI1' in ev:
            rows.append(_row('stdc', 'STDC-Seg + Ours', 'main', ['HI1'], ev['HI1'], 'RSR + BPM, single run'))
        for k in reps:
            rows.append(_row('stdc', 'STDC-Seg [%s]' % k, 'replicate', [k], ev[k],
                             common.STDC_RUNS[k]['desc']))
    if 'hrnet' in hosts:
        ev = common.ensure_evaluated('hrnet', list(common.HRNET_RUNS), force)
        if 'baseline' in ev:
            rows.append(_row('hrnet', 'HRNet-W48', 'main', ['baseline'], ev['baseline'], 'single run'))
        if 'HI1' in ev:
            rows.append(_row('hrnet', 'HRNet-W48 + Ours', 'main', ['HI1'], ev['HI1'], 'RSR + BPM, single run'))
    return rows


def rows_other(dataset, force, size32=False):
    spec = common.DATASET_RUNS[dataset]
    keys = [spec['baseline'], spec['ours']] + spec['extra']
    ev = common.ensure_evaluated('stdc', keys, force, size32=size32)
    rows = []
    if spec['baseline'] in ev:
        rows.append(_row('stdc', 'STDC-Seg', 'main', [spec['baseline']], ev[spec['baseline']], 'single run'))
    if spec['ours'] in ev:
        rows.append(_row('stdc', 'STDC-Seg + Ours', 'main', [spec['ours']], ev[spec['ours']],
                         'RSR + BPM, single run'))
    for k in spec['extra']:
        if k in ev:
            rows.append(_row('stdc', 'STDC-Seg [%s]' % k, 'arm', [k], ev[k], common.STDC_RUNS[k]['desc']))
    missing = [k for k in keys if k not in ev]
    if missing:
        print('[%s] no checkpoint for: %s' % (dataset, ', '.join(missing)))
    return rows


def host_baseline(rows, host):
    for r in rows:
        if r['host'] == host and r['row_type'] == 'main' and '+ Ours' not in r['method']:
            return r
    return None


def host_ours(rows, host):
    for r in rows:
        if r['host'] == host and r['row_type'] == 'main' and '+ Ours' in r['method']:
            return r
    return None


def metric(res, m):
    if m == 'full_all':       # old Cityscapes caches predate full_all; every class is present there
        return res.get('full_all', res['regions']['full'].get('miou_all', res['full']))
    return res[m]


def fmt_d(a, b):
    return '' if a is None or b is None else '%+.2f' % (100 * (a - b))


def write_dataset(dataset, rows, od):
    ds = DATASETS[dataset]
    names = ds['paper_classes']
    ref = rows[0]['res']
    # which classes enter the means (from any evaluated row; identical across rows
    # of the same dataset since it depends only on GT)
    used = ref['regions']['full'].get('used', [True] * len(names))
    present = ref['regions']['full'].get('present', [True] * len(names))
    support = ref.get('support')
    md = ['# Per-class IoU (%%) -- %s val' % dataset, '',
          'generated %s by gen_perclass.py' % common.now(), '']

    for reg, title in REGIONS.items():
        header = ['Method', 'Host', 'Row type', 'Runs'] + names + \
                 ['mIoU', 'mIoU (all classes)', 'Thin mean', 'Delta mIoU vs host baseline']
        out = []
        if support is not None:
            out.append(['GT support (% of val px)', '', 'support', ''] +
                       ['%.4f' % (100 * s) for s in support] + ['', '', '', ''])
        out.append(['Used in means (1=yes)', '', 'mask', ''] + ['1' if u else '0' for u in used] +
                   ['', '', '', ''])
        for r in rows:
            R = r['res']['regions'][reg]
            base = host_baseline(rows, r['host'])
            d = '' if base is None or base is r else fmt_d(R['miou'], base['res']['regions'][reg]['miou'])
            iou = [pct(x) if p else '' for x, p in zip(R['iou'], present)]
            out.append([r['method'], r['host'], r['row_type'], '+'.join(r['keys'])] + iou +
                       [pct(R['miou']), pct(R.get('miou_all', R['miou'])), pct(R['thin']), d])
        common.write_csv(osp.join(od, 'perclass_%s.csv' % reg), header, out)
        md += ['## %s' % title, ''] + common.md_table(
            ['Method'] + names + ['mIoU', 'Thin', 'd'],
            [[o[0]] + o[4:4 + len(names)] + [o[-4], o[-2], o[-1]] for o in out
             if o[2] in ('main', 'arm')]) + ['']

    # per-class delta
    header = ['Host', 'Region', 'Baseline runs', 'Ours runs'] + names + ['mIoU', 'Thin mean']
    out = []
    for host in ('stdc', 'hrnet'):
        base, ours = host_baseline(rows, host), host_ours(rows, host)
        if base is None or ours is None:
            continue
        for reg in REGIONS:
            B, O = base['res']['regions'][reg], ours['res']['regions'][reg]
            out.append([host, reg, '+'.join(base['keys']), '+'.join(ours['keys'])] +
                       [fmt_d(o, b) if p else '' for o, b, p in zip(O['iou'], B['iou'], present)] +
                       [fmt_d(O['miou'], B['miou']), fmt_d(O['thin'], B['thin'])])
    common.write_csv(osp.join(od, 'perclass_delta.csv'), header, out)
    md += ['## Ours - baseline (points)', ''] + common.md_table(
        ['Host', 'Region'] + names + ['mIoU', 'Thin'], [o[:2] + o[4:] for o in out]) + ['']

    # tidy long format (iou as fraction)
    out = []
    for r in rows:
        for reg in REGIONS:
            R = r['res']['regions'][reg]
            for c, name in enumerate(ds['classes']):
                out.append([dataset, r['host'], r['method'], r['row_type'], '+'.join(r['keys']), reg, name,
                            '%.6f' % R['iou'][c] if present[c] else '', int(bool(used[c])), R['gt_pixels'][c]])
    common.write_csv(osp.join(od, 'perclass_long.csv'),
                     ['dataset', 'host', 'method', 'row_type', 'runs', 'region', 'class', 'iou',
                      'used_in_mean', 'gt_pixels'], out)

    common.write_json(osp.join(od, 'perclass_summary.json'), OrderedDict([
        ('generated', common.now()), ('dataset', dataset),
        ('thin_classes', [ds['classes'][c] for c in ds['thin']]),
        ('min_support', ds['min_support']),
        ('classes_used_in_means', [n for n, u in zip(ds['classes'], used) if u]),
        ('rows', [OrderedDict([('host', r['host']), ('method', r['method']), ('row_type', r['row_type']),
                               ('runs', r['keys']), ('note', r['note'])] +
                              [(m, metric(r['res'], m)) for m in SUMMARY_METRICS] +
                              [('per_class', {reg: OrderedDict(zip(ds['classes'], r['res']['regions'][reg]['iou']))
                                              for reg in REGIONS})])
                  for r in rows]),
    ]))
    notes = ['Thin classes: %s.' % ', '.join(ds['classes'][c] for c in ds['thin'])]
    if ds['min_support'] is not None:
        notes.append('Means use only classes with >= %.1f%% of val GT pixels: %s.' % (
            100 * ds['min_support'], ', '.join(n for n, u in zip(ds['classes'], used) if u)))
    absent = [n for n, p in zip(ds['classes'], present) if not p]
    if absent:
        notes.append('Absent from val (blank, excluded): %s.' % ', '.join(absent))
    common.write_md(osp.join(od, 'perclass.md'), md + notes)


def summary_rows(dataset, rows):
    out = []
    for r in rows:
        if r['row_type'] not in ('main', 'arm'):
            continue
        base = host_baseline(rows, r['host'])
        res = r['res']
        row = [dataset, r['host'], r['method'], r['row_type'], '+'.join(r['keys']), r['note']]
        row += [pct(metric(res, m)) for m in SUMMARY_METRICS]
        row += ['' if base is r else fmt_d(metric(res, m), metric(base['res'], m)) for m in SUMMARY_METRICS]
        used = res['regions']['full'].get('used')
        row += [sum(used) if used else len(res['class_names']),
                ', '.join(res.get('thin_classes', []) or [res['class_names'][c] for c in common.THIN])]
        out.append(row)
    return out


def scale_check(dataset, rows75, rows32, od):
    """0.75 vs 32-divisible input: the deltas (ours - baseline) should agree."""
    out = []
    for label, rows in (('scale 0.75', rows75), ('32-div %dx%d' % DATASETS[dataset]['size32'], rows32)):
        base, ours = host_baseline(rows, 'stdc'), host_ours(rows, 'stdc')
        for r in (base, ours):
            if r:
                out.append([label, r['method']] + [pct(metric(r['res'], m)) for m in SUMMARY_METRICS])
        if base and ours:
            out.append([label, 'delta (Ours - baseline)'] +
                       [fmt_d(metric(ours['res'], m), metric(base['res'], m)) for m in SUMMARY_METRICS])
    common.write_csv(osp.join(od, 'scale_check.csv'), ['Input', 'Method'] + list(SUMMARY_METRICS.values()), out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument('--hosts', nargs='+', default=['stdc', 'hrnet'], choices=('stdc', 'hrnet'),
                    help='Cityscapes only; other datasets have STDC runs only')
    ap.add_argument('--stdc_baseline', default='mean', choices=('mean', 'I0'),
                    help="Cityscapes: 'mean' = mean of I0/Baseline2/Baseline; 'I0' = single run")
    ap.add_argument('--no_scale_check', action='store_true', help='skip the 32-divisible re-evaluation')
    ap.add_argument('--force', action='store_true', help='re-evaluate cached checkpoints')
    args = ap.parse_args()

    root = common.outdir('perclass')
    summary, checks = [], []
    for dataset in args.datasets:
        print('\n##### %s' % dataset)
        od = common.outdir('perclass', dataset)
        if dataset == 'cityscapes':
            rows = rows_cityscapes(args.hosts, args.stdc_baseline, args.force)
        else:
            rows = rows_other(dataset, args.force)
        if not rows:
            print('[%s] nothing evaluated' % dataset)
            continue
        write_dataset(dataset, rows, od)
        summary += summary_rows(dataset, rows)
        if dataset != 'cityscapes' and not args.no_scale_check:
            checks.append((dataset, scale_check(dataset, rows, rows_other(dataset, args.force, size32=True), od)))

    header = ['Dataset', 'Host', 'Method', 'Row type', 'Runs', 'Note'] + list(SUMMARY_METRICS.values()) + \
             ['d ' + v for v in SUMMARY_METRICS.values()] + ['# classes in mean', 'Thin classes']
    if summary:
        # merge with rows of datasets not regenerated this time
        f = osp.join(root, 'datasets_summary.csv')
        if osp.isfile(f):
            import csv
            with open(f) as fh:
                old = [r for r in csv.reader(fh)][1:]
            summary = [r for r in old if r and r[0] not in args.datasets] + summary
            order = list(DATASETS)
            summary.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
        common.write_csv(f, header, summary)
        md = ['# All datasets -- full mIoU and boundary metrics (%, deltas in points vs host baseline)', '',
              'generated %s by gen_perclass.py. Full mIoU = regression guard; boundary/thin = primary. '
              'RUGD means are support-filtered (>= 0.1%% of val px); "all classes" is what train.py logs.'
              % common.now(), '']
        md += common.md_table(['Dataset', 'Method'] + list(SUMMARY_METRICS.values()),
                              [[r[0], r[2]] + ['%s (%s)' % (v, d) if d else v
                                               for v, d in zip(r[6:12], r[12:18])] for r in summary])
        for dataset, rows in checks:
            md += ['', '## %s: scale 0.75 vs 32-divisible input' % dataset, ''] + \
                common.md_table(['Input', 'Method'] + list(SUMMARY_METRICS.values()), rows)
        common.write_md(osp.join(root, 'datasets_summary.md'), md)


if __name__ == '__main__':
    main()
