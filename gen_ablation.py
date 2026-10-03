#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Ablation table (cf. ref_images/quantative/Ablation.png) -- STDC-Seg,
Cityscapes val, every row compared to the full model (HI1).

Unlike the reference table, which reports full mIoU only, every row carries
all five metrics, because on this project full mIoU cannot carry an ablation
claim: its seed-to-seed spread (~1.2 pts) is larger than every ablation delta
we have. The primary columns are boundary IoU r=1 / r=3 and the thin-class
subset; full mIoU is reported as the regression guard. Each delta also comes
with |delta| / noise floor, where the noise floor is the range of that metric
across the baseline seed replicates (I0, Baseline, Baseline2).

Delta convention (same as the reference table's delta column):
    delta = variant - full model      (negative = removing the part hurts)

Rows whose checkpoint does not exist yet are written with status=pending, so
the CSV doubles as a tracker; re-run after each new run finishes and only the
new checkpoint is evaluated (others come from resultData/cache/eval/).

USAGE (any python; GPU work runs in the stdcseg env)
    python gen_ablation.py
    python gen_ablation.py --force     # re-evaluate cached checkpoints

OUTPUT  resultData/ablation/  (IoU in %, deltas in points)
    ablation.csv               main table, all rows incl. pending
    ablation_sensitivity.csv   bnd_weight and bnd_radius sweeps (HI1 = w3, r3)
    ablation_thin_r1.csv       per thin class IoU at r=1, per row
    ablation_summary.json      everything, full precision, plus the noise floor used
    ablation.md                quick-look markdown
"""
import argparse
import os.path as osp
from collections import OrderedDict

from paper import common
from paper.common import CLASS_NAMES, THIN, pct

REF = 'HI1'
BASE = '__baseline_mean__'

# (group, key, paper label). key BASE = mean of the baseline seed replicates.
# "RSR" = Arm H (BoundaryRefine head), "BPM" = Arm I (boundary-priority OHEM).
ROWS = [
    ('reference', REF,                 'Full model (RSR + BPM)'),
    ('baseline',  BASE,                'Baseline (no RSR, no BPM)'),
    ('component', 'I1',                'w/o RSR (BPM only)'),
    ('component', 'H1',                'w/o BPM (RSR only)'),
    ('module',    'ABL-M1-noHR',       'RSR w/o high-res features'),
    ('module',    'ABL-M2-s8',         'RSR at stride 8 instead of 4'),
    ('module',    'ABL-M3-noRes',      'RSR w/o residual'),
    ('module',    'ABL-M4-noLogit',    'RSR w/o logit input'),
    ('module',    'ABL-M5-noGate',     'RSR w/o gate'),
    ('loss',      'ABL-L1-unwRank',    'BPM w/ unweighted OHEM rank'),
    ('loss',      'ABL-L2-noOHEM',     'BPM w/o OHEM (w=3)'),
    ('loss',      'ABL-L2b-noOHEM-w1', 'w/o OHEM, w=1'),
    ('loss',      'ABL-L3-noDetail',   'w/o detail-head loss'),
    ('loss',      'ABL-L4-noAux',      'w/o auxiliary heads'),
]
# (param, value, key); HI1 is the w=3 / r=3 point of both sweeps
SENSITIVITY = [
    ('bnd_weight', 1, 'H1'), ('bnd_weight', 2, 'SENS-w2'), ('bnd_weight', 3, REF),
    ('bnd_weight', 5, 'SENS-w5'), ('bnd_weight', 8, 'SENS-w8'),
    ('bnd_radius', 1, 'SENS-r1'), ('bnd_radius', 2, 'SENS-r2'), ('bnd_radius', 3, REF),
    ('bnd_radius', 5, 'SENS-r5'),
]
# primary first, guard last
COLS = ('bnd_r1', 'bnd_r3', 'thin_r1', 'thin_r3', 'full')
COL_NAME = dict(bnd_r1='Bnd r=1', bnd_r3='Bnd r=3', thin_r1='Thin r=1', thin_r3='Thin r=3',
                full='Full mIoU')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='re-evaluate cached checkpoints')
    args = ap.parse_args()

    keys = [k for _, k, _ in ROWS if k != BASE] + [k for _, _, k in SENSITIVITY] + \
        list(common.BASELINE_REPLICATES)
    keys = list(OrderedDict.fromkeys(keys))
    ev = common.ensure_evaluated('stdc', keys, args.force)
    if REF not in ev:
        raise SystemExit('reference %s has no checkpoint; nothing to compare against' % REF)

    reps = [k for k in common.BASELINE_REPLICATES if k in ev]
    if reps:
        ev[BASE] = common.mean_result([ev[k] for k in reps])
    if len(reps) >= 2:
        noise = {m: max(ev[k][m] for k in reps) - min(ev[k][m] for k in reps) for m in common.METRICS}
        noise_src = 'range across %s' % ', '.join(reps)
    else:
        noise, noise_src = dict(common.NOISE_FLOOR), 'paper/common.py NOISE_FLOOR (fewer than 2 replicates on disk)'
    ref = ev[REF]

    def desc(key):
        return 'mean of %s' % '+'.join(reps) if key == BASE else common.STDC_RUNS[key]['desc']

    od = common.outdir('ablation')

    # --- main table ------------------------------------------------------------
    header = ['Group', 'Key', 'Model variation', 'Status']
    for m in COLS:
        header += [COL_NAME[m], 'd ' + COL_NAME[m], 'd/noise ' + COL_NAME[m]]
    header += ['Description']
    out, md_rows = [], []
    for group, key, label in ROWS:
        if key not in ev:
            out.append([group, key, label, 'pending'] + [''] * (3 * len(COLS)) + [desc(key)])
            md_rows.append([label, 'pending'] + [''] * len(COLS))
            continue
        r, row, mrow = ev[key], [group, key, label, 'done'], [label, 'done']
        for m in COLS:
            d = r[m] - ref[m]
            same = key == REF
            row += [pct(r[m]), '' if same else '%+.2f' % (100 * d),
                    '' if same else '%.1f' % (abs(d) / noise[m])]
            mrow.append(pct(r[m]) + ('' if same else ' (%+.2f)' % (100 * d)))
        out.append(row + [desc(key)])
        md_rows.append(mrow)
    common.write_csv(osp.join(od, 'ablation.csv'), header, out)

    # --- sensitivity sweeps ------------------------------------------------------
    s_out = []
    for param, val, key in SENSITIVITY:
        r = ev.get(key)
        s_out.append([param, val, key, 'done' if r else 'pending'] +
                     [pct(r[m]) if r else '' for m in COLS])
    common.write_csv(osp.join(od, 'ablation_sensitivity.csv'),
                     ['Param', 'Value', 'Key', 'Status'] + [COL_NAME[m] for m in COLS], s_out)

    # --- thin classes at r=1 -------------------------------------------------------
    t_out = []
    for group, key, label in ROWS:
        if key in ev:
            iou = ev[key]['regions']['bnd_r1']['iou']
            t_out.append([key, label] + [pct(iou[c]) for c in THIN] +
                         ['%+.2f' % (100 * (iou[c] - ref['regions']['bnd_r1']['iou'][c])) for c in THIN])
    common.write_csv(osp.join(od, 'ablation_thin_r1.csv'),
                     ['Key', 'Model variation'] + [CLASS_NAMES[c] for c in THIN] +
                     ['d ' + CLASS_NAMES[c] for c in THIN], t_out)

    # --- summary json + markdown --------------------------------------------------
    common.write_json(osp.join(od, 'ablation_summary.json'), OrderedDict([
        ('generated', common.now()),
        ('reference', REF),
        ('protocol', common.PROTOCOL['stdc']),
        ('noise_floor', noise), ('noise_floor_source', noise_src),
        ('baseline_replicates', reps),
        ('rows', [OrderedDict([('group', g), ('key', k), ('label', l),
                               ('status', 'done' if k in ev else 'pending')] +
                              ([(m, ev[k][m]) for m in common.METRICS] +
                               [('delta_vs_ref', {m: ev[k][m] - ref[m] for m in common.METRICS})]
                               if k in ev else []))
                  for g, k, l in ROWS]),
    ]))
    md = ['# Ablations -- STDC-Seg, Cityscapes val (IoU %, delta vs full model in points)', '',
          'generated %s by gen_ablation.py; noise floor = %s: %s' % (
              common.now(), noise_src, ', '.join('%s %.2f' % (m, 100 * noise[m]) for m in COLS)), ''] + \
        common.md_table(['Model variation', 'Status'] + [COL_NAME[m] for m in COLS], md_rows)
    common.write_md(osp.join(od, 'ablation.md'), md)
    pend = [k for _, k, _ in ROWS if k not in ev] + \
        list(OrderedDict.fromkeys(k for _, _, k in SENSITIVITY if k not in ev))
    if pend:
        print('pending (no checkpoint yet): %s' % ', '.join(pend))


if __name__ == '__main__':
    main()
