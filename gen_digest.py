#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Compact, self-describing digests of the evaluation results, meant to be
handed to whoever builds the paper tables
without access to this repo. Keeps only the numbers that carry the paper's
argument, plus the context and caveats needed to read them correctly.

Reads resultData/cache/eval/ only (no GPU; run gen_perclass.py /
gen_ablation.py first so the cache is complete).

USAGE
    python gen_digest.py

OUTPUT  resultData/digest/
    eval_digest.md       headline results for every dataset/host, noise floor,
                         component analysis, ablations, thin classes, robustness checks
    perclass_digest.md   per-class IoU tables (full image and boundary band) for
                         Cityscapes (STDC + HRNet), SYNTHIA and RUGD, with seed-noise marks
"""
import json
import os.path as osp
from collections import OrderedDict

from paper import common
from paper.common import CLASS_NAMES, DATASETS, PAPER_CLASS_NAMES, THIN

M5 = ('full', 'bnd_r1', 'bnd_r3', 'thin_r1', 'thin_r3')
MNAME = dict(full='Full mIoU', full_all='Full mIoU (all cls)', bnd_r1='Bnd r=1', bnd_r3='Bnd r=3',
             thin_r1='Thin r=1', thin_r3='Thin r=3')

HEADER = [
    'Paper: "DUET" -- two plug-and-play additions for real-time semantic segmentation that improve',
    'BOUNDARY and THIN-STRUCTURE accuracy: RSR (Residual Sub-grid Refinement: a stride-4 refinement head,',
    '+0.086M params) fixes output resolution; BPM (Boundary-Priority Mining: boundary-weighted OHEM loss,',
    'zero inference cost) fixes the training objective. "Ours" = RSR + BPM.',
    '',
    'How to read the numbers:',
    '- All values are IoU in %; deltas in percentage points.',
    '- Full mIoU = standard full-image mIoU. It is a REGRESSION GUARD, not the claim: on Cityscapes its',
    '  seed-to-seed spread (1.23 pts over 3 identical baseline runs) exceeds several effects.',
    '- Bnd r=N = IoU restricted to pixels within N px of a ground-truth class boundary (a trimap IoU).',
    '  PRIMARY metric. Thin r=N = mean IoU of thin classes inside that band.',
    '- Thin classes: Cityscapes/SYNTHIA = pole, traffic light, traffic sign, rider, motorcycle, bicycle.',
    '  RUGD = pole, sign, fence, bicycle, log (same rule: structures thinner than one stride-8 cell).',
    '- STDC-Seg = STDC2 backbone, scale 0.75 eval. HRNet = HRNetV2-W48, full-res eval (reduced 120-epoch',
    '  schedule, identical for both arms). Compare within a host/dataset only.',
    '- Cityscapes STDC baseline = mean of 3 identical-config seeds; every other row is a single run (n=1).',
]


def load(host, key, size32=False):
    return common.load_cached(host, key, size32)


def v(res, m):
    if res is None:
        return None
    if m == 'full_all':
        return res.get('full_all', res['full'])
    return res.get(m)


def f(x):
    return '' if x is None else '%.2f' % (100 * x)


def fd(a, b):
    return '' if a is None or b is None else '%+.2f' % (100 * (a - b))


def cell(a, b=None):
    """'77.25 (+1.43)' or '77.25'"""
    if a is None:
        return 'n/a'
    return f(a) if b is None else '%s (%s)' % (f(a), fd(a, b))


def table(header, rows):
    return common.md_table(header, rows) + ['']


def stdc_baseline():
    reps = [(k, load('stdc', k)) for k in common.BASELINE_REPLICATES]
    reps = [(k, r) for k, r in reps if r]
    return reps, (common.mean_result([r for _, r in reps]) if len(reps) > 1 else None)


def noise_floor(reps):
    return {m: max(r[m] for _, r in reps) - min(r[m] for _, r in reps) for m in M5} if len(reps) > 1 \
        else dict(common.NOISE_FLOOR)


# ---------------------------------------------------------------------------
def eval_digest():
    reps, base = stdc_baseline()
    noise = noise_floor(reps)
    S = {k: load('stdc', k) for k in common.STDC_RUNS}
    Hr = {k: load('hrnet', k) for k in common.HRNET_RUNS}
    out = ['# Evaluation digest -- key results for the paper', '',
           'generated %s from resultData/cache/eval by gen_digest.py' % common.now(), ''] + HEADER + ['']

    # 1 -- headline
    out += ['## 1. Main results: baseline vs Ours on every dataset and host', '',
            'Scale 0.75 for STDC (the protocol the training logs use). RUGD means use only classes with >= 0.1% '
            'of val pixels (RUGD val is two videos; rarer classes give meaningless IoU); "all cls" is the '
            'unfiltered mean that the training log reports.', '']
    rows = []

    def pair(ds, host, bname, b, oname, o, n):
        rows.append([ds, bname, n] + [cell(v(b, m)) for m in ('full', 'full_all') + M5[1:]])
        rows.append([ds, oname, '1'] + [cell(v(o, m), v(b, m)) for m in ('full', 'full_all') + M5[1:]])
    if base and S.get('HI1'):
        pair('Cityscapes', 'stdc', 'STDC-Seg', base, 'STDC-Seg + Ours', S['HI1'], str(len(reps)))
    if Hr.get('baseline') and Hr.get('HI1'):
        pair('Cityscapes', 'hrnet', 'HRNet-W48', Hr['baseline'], 'HRNet-W48 + Ours', Hr['HI1'], '1')
    for ds, spec in common.DATASET_RUNS.items():
        if S.get(spec['baseline']) and S.get(spec['ours']):
            pair(ds.upper() if ds == 'rugd' else ds.capitalize(), 'stdc', 'STDC-Seg', S[spec['baseline']],
                 'STDC-Seg + Ours', S[spec['ours']], '1')
    out += table(['Dataset', 'Method', 'n'] + [MNAME[m] for m in ('full', 'full_all') + M5[1:]], rows)

    if base and S.get('HI1'):
        out += ['Cityscapes STDC-Seg effect size as a multiple of seed noise (|delta| / range across the 3 '
                'baseline seeds):', '']
        out += table(['Metric', 'Delta', 'Seed noise (range)', 'x noise'],
                     [[MNAME[m], fd(S['HI1'][m], base[m]), '%.2f' % (100 * noise[m]),
                       '%.1f' % (abs(S['HI1'][m] - base[m]) / noise[m])] for m in M5])

    # 2 -- noise floor
    out += ['## 2. Seed noise floor (Cityscapes, STDC-Seg baseline, identical config, 3 seeds)', '']
    out += table(['Run', 'Note'] + [MNAME[m] for m in M5],
                 [[k, common.STDC_RUNS[k]['desc']] + [f(r[m]) for m in M5] for k, r in reps] +
                 [['range', 'noise floor'] + ['%.2f' % (100 * noise[m]) for m in M5]])

    # 3 -- components
    if base and all(S.get(k) for k in ('I1', 'H1', 'HI1')):
        out += ['## 3. Component analysis (Cityscapes, STDC-Seg, gains vs 3-seed baseline mean)', '']
        rows = [['Baseline', 'mean of %d seeds' % len(reps)] + [f(base[m]) for m in M5]]
        for k, lab in (('I1', '+ BPM only'), ('H1', '+ RSR only'), ('HI1', '+ RSR + BPM (Ours)')):
            rows.append([lab, k] + [cell(S[k][m], base[m]) for m in M5])
        rows.append(['sum of single-arm gains', 'I1 + H1 (additive prediction)'] +
                    ['%+.2f' % (100 * ((S['I1'][m] - base[m]) + (S['H1'][m] - base[m]))) for m in M5])
        rows.append(['Ours minus additive prediction', '> noise = super-additive'] +
                    ['%+.2f' % (100 * ((S['HI1'][m] - base[m]) - (S['I1'][m] - base[m]) - (S['H1'][m] - base[m])))
                     for m in M5])
        out += table(['Configuration', 'Run'] + [MNAME[m] for m in M5], rows)
        out += ['Reading: each arm alone gives about half the boundary gain; together they are additive on '
                'overall boundary IoU and super-additive only on thin classes (single runs). Do not use the '
                'sign of res_scale as evidence of interaction (the output is res_scale * conv, so the sign is '
                'not identifiable).', '']

    # 4 -- ablations
    ref = S.get('HI1')
    if ref:
        out += ['## 4. Ablations (Cityscapes, STDC-Seg; every row = full model with one thing changed; '
                'delta vs full model)', '',
                'Primary columns are boundary/thin. Full-mIoU ablation deltas are all below its 1.23-pt seed '
                'noise, so full mIoU cannot rank ablations; report it as the guard.', '']
        abl = [('Full model (RSR + BPM)', 'HI1'), ('w/o RSR (BPM only)', 'I1'), ('w/o BPM (RSR only)', 'H1'),
               ('RSR w/o high-res features', 'ABL-M1-noHR'), ('RSR at stride 8 instead of 4', 'ABL-M2-s8'),
               ('RSR w/o residual', 'ABL-M3-noRes'), ('RSR w/o logit input', 'ABL-M4-noLogit'),
               ('RSR w/o gate', 'ABL-M5-noGate'), ('BPM w/ unweighted OHEM rank', 'ABL-L1-unwRank'),
               ('BPM w/o OHEM (w=3)', 'ABL-L2-noOHEM'), ('w/o OHEM, w=1', 'ABL-L2b-noOHEM-w1'),
               ('w/o detail-head loss', 'ABL-L3-noDetail'), ('w/o auxiliary heads', 'ABL-L4-noAux')]
        cols = ('bnd_r1', 'bnd_r3', 'thin_r1', 'thin_r3', 'full')
        rows, pending = [], []
        if base:
            rows.append(['Baseline (no RSR, no BPM)', '3-seed mean'] + [cell(base[m], ref[m]) for m in cols] +
                        ['%.1f' % (abs(base['bnd_r1'] - ref['bnd_r1']) / noise['bnd_r1'])])
        for lab, k in abl:
            r = S.get(k)
            if r is None:
                pending.append(lab)
                continue
            same = k == 'HI1'
            rows.append([lab, k] + [cell(r[m], None if same else ref[m]) for m in cols] +
                        ['' if same else '%.1f' % (abs(r['bnd_r1'] - ref['bnd_r1']) / noise['bnd_r1'])])
        out += table(['Model variation', 'Run'] + [MNAME[m] for m in cols] + ['abs(d Bnd r=1) / noise'], rows)
        sens = [(k, lab) for k, lab in (('SENS-w2', 'w=2'), ('SENS-w5', 'w=5'), ('SENS-w8', 'w=8'),
                                        ('SENS-r1', 'r=1'), ('SENS-r2', 'r=2'), ('SENS-r5', 'r=5'))]
        done = [(k, lab) for k, lab in sens if S.get(k)]
        if done:
            out += ['Sensitivity (bnd_weight w, bnd_radius r; HI1 = w3, r3):', '']
            out += table(['Setting', 'Run'] + [MNAME[m] for m in cols],
                         [[lab, k] + [f(S[k][m]) for m in cols] for k, lab in done])
        pending += [lab for k, lab in sens if not S.get(k)]
        if pending:
            out += ['Not yet trained (no checkpoint): %s.' % ', '.join(pending), '']

    # 5 -- thin classes
    out += ['## 5. Thin classes, boundary band r=1 (Cityscapes)', '']
    thin_rows = []
    for lab, r in (('STDC-Seg (3-seed mean)', base), ('STDC-Seg + BPM only', S.get('I1')),
                   ('STDC-Seg + RSR only', S.get('H1')), ('STDC-Seg + Ours', S.get('HI1')),
                   ('HRNet-W48', Hr.get('baseline')), ('HRNet-W48 + Ours', Hr.get('HI1'))):
        if r:
            thin_rows.append([lab] + [f(r['regions']['bnd_r1']['iou'][c]) for c in THIN] +
                             [f(r['thin_r1'])])
    out += table(['Method'] + [PAPER_CLASS_NAMES[c] for c in THIN] + ['Thin mean'], thin_rows)

    # 6 -- robustness
    out += ['## 6. Robustness checks', '']
    if S.get('Synthia-I1') and S.get('Synthia'):
        out += ['SYNTHIA single arm (BPM only; this run\'s flags were not logged, identity inferred from its '
                'weights having no RSR module):', '']
        out += table(['Method'] + [MNAME[m] for m in M5],
                     [['STDC-Seg', ] + [f(S['Synthia'][m]) for m in M5],
                      ['+ BPM only'] + [cell(S['Synthia-I1'][m], S['Synthia'][m]) for m in M5]] +
                     ([['+ RSR + BPM (Ours)'] + [cell(S['Synthia-HI1'][m], S['Synthia'][m]) for m in M5]]
                      if S.get('Synthia-HI1') else []))
    rows = []
    for ds, spec in common.DATASET_RUNS.items():
        for size32, lab in ((False, 'scale 0.75 (%s)' % ('570x960' if ds == 'synthia' else '412x516')),
                            (True, '32-divisible (%dx%d)' % DATASETS[ds]['size32'])):
            b, o = load('stdc', spec['baseline'], size32), load('stdc', spec['ours'], size32)
            if b and o:
                rows.append([ds, lab, f(b['full']), f(o['full'])] + [fd(o[m], b[m]) for m in M5])
    if rows:
        out += ['Input-size check: scale 0.75 on SYNTHIA/RUGD is not a multiple of 32 (the network stride), '
                'which distorts predictions. Re-evaluated at the nearest 32-divisible size; the improvement '
                'holds at both. (SYNTHIA: every model scores ~5.8 mIoU higher at 576x960.)', '']
        out += table(['Dataset', 'Input', 'Base full', 'Ours full'] + ['d ' + MNAME[m] for m in M5], rows)

    # 7 -- cost
    rows = []
    for host in ('stdc', 'hrnet'):
        fn = osp.join(common.RESULT_DIR, 'runtime', 'raw', '%s.json' % host)
        if not osp.isfile(fn):
            continue
        with open(fn) as fh:
            d = json.load(fh)
        for r in d['results']:
            if r['name'].endswith('only'):
                continue
            first = r is d['results'][0]
            pc = lambda x: '' if first else '%+.1f%%' % (100.0 * (x - 1))
            rows.append([r['name'], '%.3f' % r['static']['params_m'], '%.1f' % r['static']['gmacs'],
                         '%.3f' % r['train']['it_s']['pooled'], pc(r['train']['ratio_vs_baseline']),
                         '%.2f' % r['train']['peak_alloc_gb'],
                         '%.2f' % r['infer']['img_s']['pooled'], pc(r['infer']['ratio_vs_baseline']),
                         '%.2f' % r['infer']['peak_alloc_gb']])
    if rows:
        out += ['## 7. Cost (Tesla T4; training = one full iteration at the training batch -- STDC 16x512x1024, '
                'HRNet 3x512x1024; inference = batch 1, 1024x2048 image -> label map)', '',
                'Params/GMACs are exact and are the primary cost claim. Throughput is pooled over 30 interleaved '
                'rounds on a thermally throttling T4; the % columns are paired per-round ratios vs the baseline '
                '(about +/-3% noise). Memory = peak allocated tensors.', '']
        out += table(['Method', 'Params (M)', 'GMACs', 'Train it/s', 'd train', 'Train mem (GB)',
                      'Infer img/s', 'd infer', 'Infer mem (GB)'], rows)

    # 8 -- caveats
    out += ['## 8. Caveats to state in the paper (so reviewers do not find them first)', '',
            '- All "Ours" rows and all non-Cityscapes/HRNet rows are single runs (n=1); HI1 seed replicates '
            'are pending. HRNet has no seed-noise estimate of its own.',
            '- Full-mIoU gains: Cityscapes STDC +1.43 is only ~1.2x seed noise; claim "no regression", not '
            '"improves mIoU".',
            '- Checkpoints are the best-val-mIoU checkpoint for every arm (same rule for all), evaluated on val.',
            '- HRNet-W48 baseline (75.6) is below the published number because of the reduced 120-epoch '
            'schedule; both arms share it.',
            '- RUGD: several classes are rare in val; means are support-filtered (>= 0.1% of val pixels). '
            'RUGD thin mean covers only fence and log (bicycle absent; pole, sign below threshold -- pole '
            'still shows the largest RUGD boundary gain, 15.7 -> 27.6, but is excluded from the mean).',
            '- Cost is not zero for a real-time method (section 7). The strongest defence is an equal-latency '
            'comparison (baseline at a larger input scale vs Ours); not in this digest yet.', '']
    return out


# ---------------------------------------------------------------------------
def perclass_block(title, names, present, rows, noise_row=None, used=None):
    """rows: [(label, iou list, miou, is_delta)]"""
    idx = [i for i, p in enumerate(present) if p and (used is None or used[i])]
    header = ['Method'] + [names[i] for i in idx] + ['mean']
    body = []
    for lab, iou, mean, delta in rows:
        body.append([lab] + [(('%+.2f' % (100 * iou[i])) if delta else f(iou[i])) for i in idx] +
                    [('%+.2f' % (100 * mean)) if delta else f(mean)])
    if noise_row:
        body.append([noise_row[0]] + ['%.2f' % (100 * noise_row[1][i]) for i in idx] + [''])
    return ['### ' + title, ''] + table(header, body)


def perclass_digest():
    reps, base = stdc_baseline()
    S = {k: load('stdc', k) for k in common.STDC_RUNS}
    Hr = {k: load('hrnet', k) for k in common.HRNET_RUNS}
    out = ['# Per-class digest -- per-class IoU (%) for the paper tables', '',
           'generated %s from resultData/cache/eval by gen_digest.py' % common.now(), ''] + HEADER + [
        '',
        'Per-class table layout follows a standard "Method x class + mIoU" table. "Delta" rows are Ours minus '
        'baseline. For Cityscapes STDC, the "seed range" row is the spread of that class across 3 identical '
        'baseline seeds; a delta marked * exceeds it (i.e. is larger than seed noise). Classes absent from a '
        'val set are omitted.', '']

    def stars(delta, rng):
        return ['%+.2f%s' % (100 * d, '*' if abs(d) > r else '') for d, r in zip(delta, rng)]

    # Cityscapes
    out += ['## Cityscapes', '']
    for reg, title in (('full', 'Full image'), ('bnd_r1', 'Boundary band r=1'), ('bnd_r3', 'Boundary band r=3')):
        rows = []
        if base and S.get('HI1'):
            rows += [('STDC-Seg (3-seed mean)', base['regions'][reg]['iou'], base['regions'][reg]['miou'], False),
                     ('STDC-Seg + Ours', S['HI1']['regions'][reg]['iou'], S['HI1']['regions'][reg]['miou'], False)]
        if Hr.get('baseline') and Hr.get('HI1'):
            rows += [('HRNet-W48', Hr['baseline']['regions'][reg]['iou'], Hr['baseline']['regions'][reg]['miou'], False),
                     ('HRNet-W48 + Ours', Hr['HI1']['regions'][reg]['iou'], Hr['HI1']['regions'][reg]['miou'], False)]
        if reg == 'bnd_r3':
            rows = []      # r=3 only as deltas below, to keep the file short
        if rows:
            out += perclass_block(title, PAPER_CLASS_NAMES, [True] * 19, rows)
        # deltas with noise marks
        drows = []
        if base and S.get('HI1'):
            rng = [max(r['regions'][reg]['iou'][c] for _, r in reps) - min(r['regions'][reg]['iou'][c] for _, r in reps)
                   for c in range(19)]
            d = [S['HI1']['regions'][reg]['iou'][c] - base['regions'][reg]['iou'][c] for c in range(19)]
            dm = S['HI1']['regions'][reg]['miou'] - base['regions'][reg]['miou']
            drows.append(['STDC-Seg: Ours - base'] + stars(d, rng) + ['%+.2f' % (100 * dm)])
            drows.append(['STDC-Seg seed range'] + ['%.2f' % (100 * x) for x in rng] + [''])
        if Hr.get('baseline') and Hr.get('HI1'):
            d = [Hr['HI1']['regions'][reg]['iou'][c] - Hr['baseline']['regions'][reg]['iou'][c] for c in range(19)]
            dm = Hr['HI1']['regions'][reg]['miou'] - Hr['baseline']['regions'][reg]['miou']
            drows.append(['HRNet-W48: Ours - base'] + ['%+.2f' % (100 * x) for x in d] + ['%+.2f' % (100 * dm)])
        if drows:
            out += ['### %s -- deltas (points)' % title, ''] + table(['Row'] + PAPER_CLASS_NAMES + ['mean'], drows)
    summ = []
    for lab, bb, oo in (('STDC-Seg', base, S.get('HI1')), ('HRNet-W48', Hr.get('baseline'), Hr.get('HI1'))):
        if bb and oo:
            d = [oo['regions']['bnd_r1']['iou'][c] - bb['regions']['bnd_r1']['iou'][c] for c in range(19)]
            top = sorted(range(19), key=lambda c: -d[c])[:3]
            summ.append('%s + Ours improves %d/19 classes at boundary r=1 (largest: %s)%s.' % (
                lab, sum(x > 0 for x in d), ', '.join('%s %+.1f' % (PAPER_CLASS_NAMES[c], 100 * d[c]) for c in top),
                '' if all(x > 0 for x in d) else '; decreases: ' + ', '.join(
                    '%s %+.1f' % (PAPER_CLASS_NAMES[c], 100 * d[c]) for c in range(19) if d[c] <= 0)))
    out += ['Summary: ' + ' '.join(summ) + ' Rare classes (truck, bus, train, motorcycle) swing several points '
            'between identical seeds; make no per-class claims about them.', '']

    # SYNTHIA / RUGD
    for ds, spec in common.DATASET_RUNS.items():
        b, o = S.get(spec['baseline']), S.get(spec['ours'])
        if not (b and o):
            continue
        names = DATASETS[ds]['paper_classes']
        present = b['regions']['full'].get('present', [True] * len(names))
        used = b['regions']['full'].get('used') if DATASETS[ds]['min_support'] else None
        out += ['## %s (STDC-Seg, n=1)' % ('RUGD' if ds == 'rugd' else 'SYNTHIA'), '']
        if ds == 'rugd':
            sup = b['support']
            out += ['Only classes with >= 0.1%% of val pixels are shown/averaged: %s. Excluded rare classes: %s.'
                    % (', '.join(n for n, u in zip(names, used) if u),
                       ', '.join('%s (%.3f%%)' % (n, 100 * s) for n, s, u, p in zip(names, sup, used, present)
                                 if p and not u)), '']
        else:
            out += ['SYNTHIA labels use Cityscapes classes; terrain, truck and train do not occur in val.', '']
        for reg, title in (('full', 'Full image'), ('bnd_r1', 'Boundary band r=1')):
            B, O = b['regions'][reg], o['regions'][reg]
            rows = [('STDC-Seg', B['iou'], B['miou'], False), ('STDC-Seg + Ours', O['iou'], O['miou'], False),
                    ('Ours - base', [x - y for x, y in zip(O['iou'], B['iou'])], O['miou'] - B['miou'], True)]
            out += perclass_block(title, names, present, rows, used=used)
        if ds == 'rugd':
            pi = names.index('pole')
            out += ['Excluded but notable -- pole (0.083%% of val px): full %s -> %s, boundary r=1 %s -> %s.' % (
                f(b['regions']['full']['iou'][pi]), f(o['regions']['full']['iou'][pi]),
                f(b['regions']['bnd_r1']['iou'][pi]), f(o['regions']['bnd_r1']['iou'][pi])), '']
    return out


def main():
    od = common.outdir('digest')
    common.write_md(osp.join(od, 'eval_digest.md'), eval_digest())
    common.write_md(osp.join(od, 'perclass_digest.md'), perclass_digest())


if __name__ == '__main__':
    main()
