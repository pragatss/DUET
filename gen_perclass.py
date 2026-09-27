#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Per-class IoU tables (paper Table: per-class comparison, cf.
ref_images/quantative/PerClass.png) for Cityscapes val:

    STDC-Seg            (baseline; default = mean of the 3 seed replicates)
    STDC-Seg + Ours     (HI1)
    HRNet-W48           (baseline)
    HRNet-W48 + Ours    (HI1)

for three regions: full image (the classic table; mIoU is the regression
guard) and the boundary band at r=1 and r=3 (the primary metric). Each host is
scored with its own native protocol (paper/common.py PROTOCOL); every number
comes from one evaluator, so STDC and HRNet boundary numbers are computed the
same way.

Checkpoints are evaluated once and cached in resultData/cache/eval/ (keyed on
checkpoint mtime+size), so re-running only rebuilds the tables. Missing
checkpoints are skipped with a note.

USAGE (any python; GPU work runs in each host's conda env)
    python gen_perclass.py                       # both hosts
    python gen_perclass.py --hosts stdc          # STDC only
    python gen_perclass.py --stdc_baseline I0    # single-run baseline instead of n=3 mean
    python gen_perclass.py --force               # re-evaluate even if cached

OUTPUT  resultData/perclass/  (all IoU values in %, see resultData/README.md)
    perclass_full.csv  perclass_bnd_r1.csv  perclass_bnd_r3.csv   one table per region
    perclass_delta.csv          Ours - baseline, per class, per host and region
    perclass_long.csv           tidy: one row per (method, region, class)
    perclass_summary.json       everything, full precision
    perclass.md                 quick-look markdown of the three tables
"""
import argparse
import os.path as osp
from collections import OrderedDict

from paper import common
from paper.common import CLASS_NAMES, PAPER_CLASS_NAMES, pct

REGIONS = OrderedDict([('full', 'Full image'), ('bnd_r1', 'Boundary band r=1'),
                       ('bnd_r3', 'Boundary band r=3')])


def build_rows(hosts, stdc_baseline, force):
    """-> list of row dicts: host, method, row_type, keys, res, ref (index of host baseline row)"""
    rows = []
    if 'stdc' in hosts:
        keys = list(common.BASELINE_REPLICATES) + ['HI1']
        ev = common.ensure_evaluated('stdc', keys, force)
        reps = [k for k in common.BASELINE_REPLICATES if k in ev]
        if stdc_baseline == 'mean' and len(reps) > 1:
            base = dict(host='stdc', method='STDC-Seg', row_type='main', keys=reps,
                        res=common.mean_result([ev[k] for k in reps]),
                        note='mean of %d seed replicates' % len(reps))
        elif 'I0' in ev:
            base = dict(host='stdc', method='STDC-Seg', row_type='main', keys=['I0'],
                        res=ev['I0'], note='single run (I0)')
        else:
            base = None
            print('[stdc] no baseline checkpoint found')
        if base:
            rows.append(base)
        if 'HI1' in ev:
            rows.append(dict(host='stdc', method='STDC-Seg + Ours', row_type='main', keys=['HI1'],
                             res=ev['HI1'], note='RSR + BPM'))
        for k in reps:
            rows.append(dict(host='stdc', method='STDC-Seg [%s]' % k, row_type='replicate',
                             keys=[k], res=ev[k], note=common.STDC_RUNS[k]['desc']))
    if 'hrnet' in hosts:
        ev = common.ensure_evaluated('hrnet', list(common.HRNET_RUNS), force)
        if 'baseline' in ev:
            rows.append(dict(host='hrnet', method='HRNet-W48', row_type='main', keys=['baseline'],
                             res=ev['baseline'], note='single run'))
        if 'HI1' in ev:
            rows.append(dict(host='hrnet', method='HRNet-W48 + Ours', row_type='main', keys=['HI1'],
                             res=ev['HI1'], note='RSR + BPM, single run'))
    for h in hosts:
        missing = [k for k in common.HOST_RUNS[h] if not common.available(h, k)
                   and (h == 'hrnet' or k in common.BASELINE_REPLICATES + ('HI1',))]
        if missing:
            print('[%s] no checkpoint for: %s' % (h, ', '.join(missing)))
    return rows


def host_baseline(rows, host):
    for r in rows:
        if r['host'] == host and r['row_type'] == 'main' and '+ Ours' not in r['method']:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hosts', nargs='+', default=['stdc', 'hrnet'], choices=('stdc', 'hrnet'))
    ap.add_argument('--stdc_baseline', default='mean', choices=('mean', 'I0'),
                    help="'mean' = mean of I0/Baseline2/Baseline (claude.md); 'I0' = single run")
    ap.add_argument('--force', action='store_true', help='re-evaluate cached checkpoints')
    args = ap.parse_args()

    rows = build_rows(args.hosts, args.stdc_baseline, args.force)
    od = common.outdir('perclass')
    md = ['# Per-class IoU (%) -- Cityscapes val', '', 'generated %s by gen_perclass.py' % common.now(), '']

    # --- one table per region ------------------------------------------------
    for reg, title in REGIONS.items():
        header = ['Method', 'Host', 'Row type', 'Runs'] + PAPER_CLASS_NAMES + \
                 ['mIoU', 'Thin mean', 'Delta mIoU vs host baseline']
        out = []
        for r in rows:
            R = r['res']['regions'][reg]
            base = host_baseline(rows, r['host'])
            d = '' if base is None or base is r else \
                '%+.2f' % (100 * (R['miou'] - base['res']['regions'][reg]['miou']))
            out.append([r['method'], r['host'], r['row_type'], '+'.join(r['keys'])] +
                       [pct(x) for x in R['iou']] + [pct(R['miou']), pct(R['thin']), d])
        common.write_csv(osp.join(od, 'perclass_%s.csv' % reg), header, out)
        md += ['## %s' % title, ''] + common.md_table(
            ['Method'] + PAPER_CLASS_NAMES + ['mIoU', 'Thin', 'd'],
            [[o[0]] + o[4:] for o, r in zip(out, rows) if r['row_type'] == 'main']) + ['']

    # --- per-class delta (Ours - baseline) ------------------------------------
    header = ['Host', 'Region', 'Baseline runs', 'Ours runs'] + PAPER_CLASS_NAMES + ['mIoU', 'Thin mean']
    out = []
    for host in args.hosts:
        base = host_baseline(rows, host)
        ours = [r for r in rows if r['host'] == host and '+ Ours' in r['method']]
        if base is None or not ours:
            continue
        ours = ours[0]
        for reg in REGIONS:
            B, O = base['res']['regions'][reg], ours['res']['regions'][reg]
            out.append([host, reg, '+'.join(base['keys']), '+'.join(ours['keys'])] +
                       ['%+.2f' % (100 * (o - b)) for o, b in zip(O['iou'], B['iou'])] +
                       ['%+.2f' % (100 * (O['miou'] - B['miou'])), '%+.2f' % (100 * (O['thin'] - B['thin']))])
    common.write_csv(osp.join(od, 'perclass_delta.csv'), header, out)
    md += ['## Ours - baseline (points)', ''] + common.md_table(
        ['Host', 'Region'] + PAPER_CLASS_NAMES + ['mIoU', 'Thin'], [o[:2] + o[4:] for o in out]) + ['']

    # --- tidy long format -------------------------------------------------------
    out = []
    for r in rows:
        for reg in REGIONS:
            R = r['res']['regions'][reg]
            for c, name in enumerate(CLASS_NAMES):
                out.append([r['host'], r['method'], r['row_type'], '+'.join(r['keys']), reg, name,
                            '%.6f' % R['iou'][c], R['gt_pixels'][c]])
    common.write_csv(osp.join(od, 'perclass_long.csv'),
                     ['host', 'method', 'row_type', 'runs', 'region', 'class', 'iou', 'gt_pixels'], out)

    # --- summary json ----------------------------------------------------------
    summary = OrderedDict([
        ('generated', common.now()),
        ('dataset', 'cityscapes val (500 images)'),
        ('protocol', {h: common.PROTOCOL[h] for h in args.hosts}),
        ('stdc_baseline', args.stdc_baseline),
        ('rows', [OrderedDict([('host', r['host']), ('method', r['method']), ('row_type', r['row_type']),
                               ('runs', r['keys']), ('note', r['note'])] +
                              [(m, r['res'][m]) for m in common.METRICS] +
                              [('per_class', {reg: OrderedDict(zip(CLASS_NAMES, r['res']['regions'][reg]['iou']))
                                              for reg in REGIONS})])
                  for r in rows]),
    ])
    common.write_json(osp.join(od, 'perclass_summary.json'), summary)
    md += ['Notes: STDC baseline = %s. HRNet rows are n=1. Full-image mIoU is a regression '
           'guard; the boundary tables carry the claim.' % host_baseline(rows, 'stdc')['note']
           if host_baseline(rows, 'stdc') else '']
    common.write_md(osp.join(od, 'perclass.md'), md)


if __name__ == '__main__':
    main()
