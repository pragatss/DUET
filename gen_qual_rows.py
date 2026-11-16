#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Better rows for Figure A (the qualitative comparison), chosen from the data: the val images
where the baseline is far from the GT but the same network + Ours is close to it.
Two rows per dataset, because the method improves the overall segmentation, with a
specific focus on thin classes:

  overall  largest class-weighted IoU gain over all classes in the frame
  thin     largest class-weighted IoU gain over the thin classes, GT thin pixels >= 0.5% of
           the frame, thin gain >= 5 points (STDC-Seg) and >= 2 points (HRNet-W48)

Class-weighted IoU gain, per image and network: per-class IoU, Ours minus baseline, for every
class whose GT covers >= 0.1% of the frame, averaged with weight sqrt(GT class size) -- a class 100x larger counts 10x more, not 100x, so a gain in sky or grass
cannot hide a loss in a pole or sign, and a speck cannot outweigh the road. Losses count
double: gain = sum w * max(d, 0) - 2 * sum w * max(-d, 0), over sum w.

Every candidate, both rows and both networks: the error drops, no class loses more than 3
IoU points, the class-weighted gain is positive and the thin gain is not negative. Clean
images rank first: Ours also does not visibly hurt an area that was right -- visible broken
pixels <= 0.2 x visible fixed (STDC-Seg), <= 0.35 x (HRNet-W48), where visible = in an area
big enough to see in a figure panel (survives an opening of radius width / 800 px).
Score = STDC-Seg gain + 0.5 x HRNet-W48 gain, the HRNet term capped at the STDC-Seg one
(STDC-Seg is the main comparison).
Columns as in fig_qualitative/: RGB | GT | STDC-Seg | STDC-Seg + Ours | HRNet-W48 |
HRNet-W48 + Ours. Reads the gen_qualitative.py prediction cache; no GPU work.

Every run writes a new dated folder next to fig_qualitative/ and never touches older ones:
  resultData/qualitative/fig_qualitative_<YYYY-MM-DD>/   (same day again: ..._2, ...)
    SELECTION.md            rows, why each was chosen, alternatives by name, file list
    PREVIEW_qualitative.png the rows assembled (reference only)
    rowN_<dataset>_<kind>__<image>/  1_rgb ... 6_hrnet_ours.png (+ extra/ fixed/broken maps)
    legend_*.png, candidates/candidates_<dataset>_<kind>.png, ranking_<dataset>.csv
  resultData/qualitative/ROWS_LOG.md   one entry per run: folder, rows, command

USAGE -- stdcseg env:
    /home/husky/anaconda3/envs/stdcseg/bin/python gen_qual_rows.py
    ... --pick synthia:thin:0000815     # force one row (dataset:kind:image), repeatable
"""
import argparse
import csv
import datetime
import json
import os.path as osp
import sys
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw

from paper import common
from gen_qualitative import (QROOT, QCACHE, PALETTE, CITYSCAPES_PAL, RUGD_PAL, LEGEND_NAMES_CS, DS_TITLE, RUNS,
                             COL_TITLES, colorize, load_label, load_rgb, to_size, resize, save, overlay, fit, font,
                             legend, mkdir, gt_path, pred_path, rgb_index, load_stats, group_of, preview,
                             boundary_focus)

METRICS_VERSION = 7
KINDS = OrderedDict([('overall', 'overall segmentation'), ('thin', 'thin classes')])
MIN_THIN_SHARE = 0.005    # thin row: GT thin pixels must cover >= 0.5% of the labelled pixels
MIN_THIN_IOU_GAIN = dict(stdc=0.05, hrnet=0.02)   # thin row: thin gain must be >= 5 / 2 points
HRNET_WEIGHT = 0.5        # score = STDC-Seg gain + HRNET_WEIGHT x HRNet-W48 gain
HOSTS = ('stdc', 'hrnet')
HOST_TITLE = dict(stdc='STDC-Seg', hrnet='HRNet-W48')
VISIBLE_DIV = 800         # visible-area kernel radius = image width / 800 px (Cityscapes 3, SYNTHIA 2, RUGD 1)
BAND_RADIUS = 3           # GT boundary band, as the paper's bnd_r3
MIN_CLASS_SHARE = 0.001   # a class counts in the class-weighted gain if it covers >= 0.1% of the frame
LOSS_WEIGHT = 2.0         # class-weighted gain: a lost IoU point counts double
MAX_CLASS_DROP = 0.03     # no counted class may lose more than 3 IoU points, either network
MAX_VIS_BROKEN = dict(stdc=0.2, hrnet=0.35)    # clean: visible broken <= this x visible fixed
MIN_FRAME_GAP = 50        # RUGD (two sequences): min frame gap between rows of one sequence


def thin_classes(ds):
    """thin classes that also gained in the per-class table (bnd r1 >= +1.0 and full >= 0) -- drops e.g. RUGD log"""
    with open(osp.join(common.RESULT_DIR, 'perclass', ds, 'perclass_delta.csv')) as fh:
        rows = {r['Region']: r for r in csv.DictReader(fh) if r['Host'] == 'stdc'}
    cols = common.DATASETS[ds]['paper_classes']          # the CSV uses the paper's class names
    return [c for c in common.DATASETS[ds]['thin']
            if rows['bnd_r1'].get(cols[c]) and float(rows['bnd_r1'][cols[c]]) >= 1.0
            and float(rows['full'][cols[c]]) >= 0.0]


# ---------------------------------------------------------------------------
# per-image metrics, every host (cached)
# ---------------------------------------------------------------------------
def class_ious(gt, valid, pb, po, n_cls):
    """{class: [share of the frame (union of GT and both predictions), IoU baseline, IoU Ours, GT share]} for every
    class in the GT or either prediction; IoU = 1 where a model neither predicts the class nor has it in the GT"""
    g = gt[valid].astype(np.int64)
    cms = [np.bincount(g * n_cls + p[valid], minlength=n_cls * n_cls).reshape(n_cls, n_cls) for p in (pb, po)]
    out = {}
    for c in range(n_cls):
        ng = cms[0][c].sum()
        unions = [ng + cm[:, c].sum() - cm[c, c] for cm in cms]
        if not any(unions):
            continue
        ious = [cm[c, c] / float(u) if u else 1.0 for cm, u in zip(cms, unions)]
        union = ((gt == c) | (pb == c) | (po == c)) & valid
        out[str(c)] = [union.sum() / float(valid.sum())] + ious + [ng / float(valid.sum())]
    return out


def pair_metrics(gt, valid, band, pb, po, thin, n_cls):
    """fix / brk: pixels Ours fixed (baseline wrong, Ours right) / broke (baseline right, Ours wrong), per frame
    px; bnd_*: the same inside the GT boundary band; blob_brk: largest connected broken patch, per frame px;
    vis_fix / vis_brk: fixed / broken pixels that survive an opening with a VISIBLE_DIV-scaled kernel -- the areas
    big enough to see in a figure panel (1-2 px edge slivers drop out), per frame px"""
    nv, nband = float(valid.sum()), float(band.sum())
    tg = np.isin(gt, thin) & valid
    wb, wo = (pb != gt) & valid, (po != gt) & valid
    brk = wo & ~wb
    r = max(1, int(round(gt.shape[1] / VISIBLE_DIV)))
    vis = lambda x: cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    out = dict(err_b=wb.sum() / nv, err_o=wo.sum() / nv, fix=(wb & ~wo).sum() / nv, brk=brk.sum() / nv,
               bnd_err_b=(wb & band).sum() / nband, bnd_err_o=(wo & band).sum() / nband,
               bnd_brk=(brk & band).sum() / nband,
               blob_brk=cv2.connectedComponentsWithStats(brk.astype(np.uint8), connectivity=8)[2][1:, 4].max(
                   initial=0) / nv,
               vis_fix=vis(wb & ~wo).sum() / nv, vis_brk=vis(brk).sum() / nv)
    out['cls'] = class_ious(gt, valid, pb, po, n_cls)
    for tag, p in (('b', pb), ('o', po)):
        tp = (tg & (p == gt)).sum()
        union = (tg | (np.isin(p, thin) & valid)).sum()
        out['thin_' + tag] = tp / float(union) if union else None
        out['thin_net_' + tag] = (tp - (union - tg.sum())) / nv     # right thin px minus false thin px, per frame px
    return out


def metrics(ds):
    thin = thin_classes(ds)
    keys = [(h, k) for h, b, o, _ in RUNS[ds] for k in (b, o)]
    key = dict(version=METRICS_VERSION, thin=thin, runs=[
        json.load(open(osp.join(QCACHE, 'stats', '%s__%s.json' % hk)))['fingerprint'] for hk in keys])
    f = osp.join(QCACHE, 'rows_metrics_%s.json' % ds)
    if osp.isfile(f):
        with open(f) as fh:
            d = json.load(fh)
        if d['key'] == key:
            return [dict(m, thin_cls=thin) for m in d['images']]
    names = set.intersection(*[set(load_stats(h, k)) for h, k in keys])
    out = []
    for i, n in enumerate(sorted(names)):
        gt = load_label(gt_path(ds, n))
        valid = gt != 255
        if not valid.any():
            continue
        m = dict(name=n, thin_share=float((np.isin(gt, thin) & valid).sum()) / valid.sum())
        band = (boundary_focus(gt, thin, BAND_RADIUS) > 0) & valid
        for h, b, o, _ in RUNS[ds]:
            m[h] = pair_metrics(gt, valid, band, load_label(pred_path(h, b, n)), load_label(pred_path(h, o, n)), thin,
                                 len(common.DATASETS[ds]['classes']))
        out.append(m)
        if (i + 1) % 200 == 0:
            print('  [%s] %d / %d' % (ds, i + 1, len(names)))
    with open(f, 'w') as fh:
        json.dump(dict(key=key, images=out), fh, default=float)
    return [dict(m, thin_cls=thin) for m in out]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def class_deltas(m, host, thin_only=False):
    """[(class, weight sqrt(GT share), IoU Ours - baseline)] for the classes whose GT covers >= MIN_CLASS_SHARE of
    the frame. Classes absent from the GT are left out (their IoU only jumps 0 <-> 1); their false pixels still
    lower the IoU of the GT class they cover"""
    return [(int(c), np.sqrt(v[3]), v[2] - v[1]) for c, v in m[host]['cls'].items()
            if v[3] >= MIN_CLASS_SHARE and (not thin_only or int(c) in m['thin_cls'])]


def gain(m, kind, host):
    """class-weighted IoU gain over all counted classes (overall) or the thin ones (thin); None if none counted"""
    cd = class_deltas(m, host, kind == 'thin')
    if not cd:
        return None
    w = sum(x[1] for x in cd)
    return sum(wc * (max(d, 0) - LOSS_WEIGHT * max(-d, 0)) for _, wc, d in cd) / w


def worst(m, host):
    """largest per-class IoU loss among the counted classes (<= 0), and its class"""
    return min([(min(d, 0.0), c) for c, _, d in class_deltas(m, host)] or [(0.0, None)])


def score(m, kind):
    """HRNet's term is capped at STDC-Seg's, so a row is never chosen for HRNet alone"""
    s = gain(m, kind, 'stdc')
    return s + HRNET_WEIGHT * min(gain(m, kind, 'hrnet'), s)


def clean(m):
    """Ours does not visibly hurt an area that was right: visible broken small next to visible fixed, both networks"""
    return all(m[h]['vis_brk'] <= MAX_VIS_BROKEN[h] * m[h]['vis_fix'] for h in HOSTS)


def rank(ms):
    """both rows, both networks: error drops, no class loses > MAX_CLASS_DROP, class-weighted gain > 0, thin gain not
    negative. Clean images first, each part by score"""
    ok = [m for m in ms if all(m[h]['err_o'] < m[h]['err_b'] and worst(m, h)[0] >= -MAX_CLASS_DROP
                               and gain(m, 'overall', h) > 0 and (gain(m, 'thin', h) or 0) >= 0 for h in HOSTS)]
    return OrderedDict([
        ('overall', sorted(ok, key=lambda m: (not clean(m), -score(m, 'overall')))),
        ('thin', sorted([m for m in ok if m['thin_share'] >= MIN_THIN_SHARE
                         and all((gain(m, 'thin', h) or 0) >= MIN_THIN_IOU_GAIN[h] for h in HOSTS)],
                        key=lambda m: (not clean(m), -score(m, 'thin'))))])


def too_close(ds, a, b):
    if group_of(ds, a) != group_of(ds, b):
        return False
    if ds == 'rugd':                       # only two sequences: a frame gap is enough
        return abs(int(a.rsplit('_', 1)[1]) - int(b.rsplit('_', 1)[1])) < MIN_FRAME_GAP
    return True


def choose(ds, ranked, picks, by_name):
    chosen = OrderedDict()
    for kind in KINDS:
        p = picks.get((ds, kind))
        if p:
            if p not in by_name:
                raise SystemExit('--pick %s:%s:%s: image not in the %s val set' % (ds, kind, p, ds))
            chosen[kind] = by_name[p]
    for kind in KINDS:
        if kind not in chosen:
            # clean beats diverse: clean and apart, clean, apart, anything (one image never fills both rows)
            others = [m['name'] for m in chosen.values()]
            apart = lambda m: not any(too_close(ds, m['name'], o) for o in others)
            m = next((m for test in (lambda m: clean(m) and apart(m), clean, apart, lambda m: True)
                      for m in ranked[kind] if test(m) and m['name'] not in others), None)
            if m:
                chosen[kind] = m
    return OrderedDict((k, chosen[k]) for k in KINDS if k in chosen)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def panels_of(ds, name):
    pal = PALETTE[ds]
    gt = load_label(gt_path(ds, name))
    valid = gt != 255
    rgb = load_rgb(rgb_index(ds)[name])
    if rgb.shape[:2] != gt.shape:
        rgb = to_size(rgb, gt.shape, nearest=False)
    panels = OrderedDict([('rgb', rgb), ('gt', colorize(gt, pal))])
    extra = OrderedDict()
    present = set(np.unique(gt).tolist())
    for host, b, o, col in RUNS[ds]:
        pb, po = load_label(pred_path(host, b, name)), load_label(pred_path(host, o, name))
        for lbl, key in ((pb, col), (po, col + '_ours')):
            c = colorize(lbl, pal)
            c[~valid] = 0
            panels[key] = c
            present |= set(np.unique(lbl[valid]).tolist())
        wb, wo = (pb != gt) & valid, (po != gt) & valid
        extra['%s_fixed_green_broken_red' % host] = overlay(rgb, [((wb & ~wo).astype(np.uint8), (0, 230, 0)),
                                                                 ((wo & ~wb).astype(np.uint8), (255, 40, 40))])
    return panels, extra, sorted(present - {255})


def render_row(i, ds, kind, m, outdir):
    """1_rgb ... 6_hrnet_ours.png enlarged by an integer factor to a long side >= 1600 px, + extra/"""
    panels, extra, present = panels_of(ds, m['name'])
    H, W = panels['gt'].shape[:2]
    k = int(np.ceil(1600.0 / max(H, W)))
    folder = mkdir(osp.join(outdir, 'row%d_%s_%s__%s' % (i, ds, kind, m['name'])))
    mkdir(osp.join(folder, 'extra'))
    files = OrderedDict()
    for j, (key, arr) in enumerate(panels.items(), 1):
        files[key] = '%d_%s.png' % (j, key)
        save(resize(arr, W * k, key != 'rgb') if k > 1 else arr, osp.join(folder, files[key]))
    for key, arr in extra.items():
        save(resize(arr, W * k, True) if k > 1 else arr, osp.join(folder, 'extra', key + '.png'))
    return dict(dataset=ds, kind=kind, name=m['name'], folder=osp.relpath(folder, QROOT), files=files,
                present=present, k=k, size=[H, W])


def contact_sheet(ds, kind, pool, outpath, n):
    cw, ch, pad, txt_w = 220, 130, 4, 520
    cols = list(COL_TITLES.values())
    rows = pool[:n]
    im = Image.new('RGB', (len(cols) * (cw + pad) + txt_w, 30 + len(rows) * (ch + pad)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for j, t in enumerate(cols):
        d.text((j * (cw + pad) + 6, 6), t, fill=(0, 0, 0), font=font(15))
    for i, m in enumerate(rows):
        panels, _, _ = panels_of(ds, m['name'])
        y = 30 + i * (ch + pad)
        for j, a in enumerate(panels.values()):
            im.paste(fit(a, cw, ch), (j * (cw + pad), y))
        d.text((len(cols) * (cw + pad) + 8, y + 6), '#%d  %s\n%s' % (i + 1, m['name'], describe(m, '\n')),
               fill=(0, 0, 0), font=font(14))
    im.save(outpath)


def describe(m, sep):
    out = []
    for h in HOSTS:
        g, t = m[h], gain(m, 'thin', h)
        out.append('%s: class gain %+.1f, thin %s, worst class %+.1f, error %.1f%% -> %.1f%%' % (
            HOST_TITLE[h], 100 * gain(m, 'overall', h), '%+.1f' % (100 * t) if t is not None else 'n/a',
            100 * worst(m, h)[0], 100 * g['err_b'], 100 * g['err_o']))
    return sep.join(out)


def class_changes(ds, m, host):
    """every counted class, largest gain first: 'pole +14.0, ... , sidewalk -1.7'"""
    names = common.DATASETS[ds]['classes']
    return ', '.join('%s %+.1f' % (names[c], 100 * d) for c, _, d in sorted(class_deltas(m, host), key=lambda x: -x[2]))


def pct(x):
    return '' if x is None else '%.2f' % (100 * x)


# ---------------------------------------------------------------------------
def run_folder(date):
    d = osp.join(QROOT, 'fig_qualitative_%s' % date)
    i = 2
    while osp.exists(d):
        d = osp.join(QROOT, 'fig_qualitative_%s_%d' % (date, i))
        i += 1
    return mkdir(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(RUNS), choices=list(RUNS))
    ap.add_argument('--pick', action='append', default=[], help='<dataset>:<overall|thin>:<image>, repeatable')
    ap.add_argument('--top', type=int, default=10, help='alternatives per dataset and kind in candidates/')
    ap.add_argument('--date', default=datetime.date.today().isoformat())
    args = ap.parse_args()
    picks = {}
    for p in args.pick:
        ds, kind, name = p.split(':', 2)
        if ds not in RUNS or kind not in KINDS:
            raise SystemExit('--pick %s: expected <dataset>:<overall|thin>:<image>' % p)
        picks[(ds, kind)] = name

    out = run_folder(args.date)
    cand = mkdir(osp.join(out, 'candidates'))
    rows, alts = [], OrderedDict()
    for ds in args.datasets:
        print('[%s] scoring' % ds)
        ms = metrics(ds)
        ranked = rank(ms)
        for kind, m in choose(ds, ranked, picks, {m['name']: m for m in ms}).items():
            r = render_row(len(rows) + 1, ds, kind, m, out)
            r['m'] = m
            r['src'] = ('rank %d of %d' % (ranked[kind].index(m) + 1, len(ranked[kind])) if m in ranked[kind]
                        else 'not ranked') + (', picked by eye' if (ds, kind) in picks else '')
            rows.append(r)
            print('  row %d %-8s %-24s %s' % (len(rows), kind, m['name'], describe(m, ' | ')))
        for kind, pool in ranked.items():
            contact_sheet(ds, kind, pool, osp.join(cand, 'candidates_%s_%s.png' % (ds, kind)), args.top)
            alts[(ds, kind)] = [m['name'] for m in pool[:args.top]]
        common.write_csv(osp.join(out, 'ranking_%s.csv' % ds),
                         ['image', 'thin_share'] + ['%s_%s' % (h, c) for h in HOSTS for c in
                                                    ('error_baseline', 'error_ours', 'fixed', 'broken',
                                                     'visible_fixed', 'visible_broken', 'band_fixed', 'band_broken', 'largest_broken_patch',
                                                     'thin_iou_baseline', 'thin_iou_ours')] +
                         ['%s_%s' % (h, c) for h in HOSTS for c in ('class_gain', 'thin_class_gain', 'worst_class_drop')] +
                         ['clean', 'overall_score', 'thin_score'],
                         [[m['name'], pct(m['thin_share'])] +
                          [pct(v) for h in HOSTS for v in (
                              m[h]['err_b'], m[h]['err_o'], m[h]['fix'], m[h]['brk'], m[h]['vis_fix'], m[h]['vis_brk'],
                              m[h]['bnd_err_b'] - m[h]['bnd_err_o'] + m[h]['bnd_brk'], m[h]['bnd_brk'],
                              m[h]['blob_brk'], m[h]['thin_b'], m[h]['thin_o'])] +
                          [pct(v) for h in HOSTS for v in (gain(m, 'overall', h), gain(m, 'thin', h), worst(m, h)[0])] +
                          [int(clean(m)), pct(score(m, 'overall')),
                           pct(score(m, 'thin')) if all(gain(m, 'thin', h) is not None for h in HOSTS) else '']
                          for m in sorted(ms, key=lambda m: -score(m, 'overall'))])

    preview(rows, osp.join(out, 'PREVIEW_qualitative.png'))
    legends = []
    if any(r['dataset'] != 'rugd' for r in rows):
        legend(LEGEND_NAMES_CS + ['n/a'], CITYSCAPES_PAL + [(0, 0, 0)],
               osp.join(out, 'legend_cityscapes_synthia.png'), per_row=10)
        legends.append('legend_cityscapes_synthia.png')
    rugd_cls = sorted(set(c for r in rows if r['dataset'] == 'rugd' for c in r['present']))
    if rugd_cls:
        legend([common.RUGD_CLASSES[c] for c in rugd_cls] + ['n/a'], [RUGD_PAL[c] for c in rugd_cls] + [(0, 0, 0)],
               osp.join(out, 'legend_rugd.png'), cell_w=130, per_row=8)
        legends.append('legend_rugd.png')
    write_selection(out, rows, alts, legends, args)
    write_log(out, rows)
    print('done -> %s' % osp.relpath(out, common.STDC_ROOT))


def write_selection(out, rows, alts, legends, args):
    names = lambda ds: common.DATASETS[ds]['classes']
    md = ['# Figure A rows, %s' % osp.basename(out), '',
          'Generated %s by `gen_qual_rows.py`. All paths are relative to this folder. The earlier rows in '
          '`../fig_qualitative/` are untouched.' % common.now(), '',
          '## How the rows were chosen', '',
          'The method improves the overall segmentation, with a specific focus on thin classes, so every dataset '
          'gets one row of each:', '',
          '* **Class-weighted IoU gain** (per image, per network): per-class IoU, Ours minus baseline, for every '
          'class whose GT covers >= %.1f%% of the frame, averaged with weight sqrt(GT class size). Classes absent '
          'from the GT are left out (their IoU only jumps between 0 and 1); their false pixels still lower the IoU '
          'of the GT class they cover. '
          'A class 100x larger counts 10x more, not 100x, so a gain in sky or grass cannot hide a loss in a pole or '
          'sign, and a speck cannot outweigh the road. Lost IoU points count %.0f x: gain = (sum w max(d, 0) - %.0f '
          'sum w max(-d, 0)) / sum w.' % (100 * MIN_CLASS_SHARE, LOSS_WEIGHT, LOSS_WEIGHT),
          '* **overall**: the largest class-weighted gain over all counted classes.',
          '* **thin**: the largest class-weighted gain over the thin classes; GT thin pixels >= %.1f%% of the frame '
          'and thin gain >= %.0f points (STDC-Seg) / >= %.0f (HRNet-W48).'
          % (100 * MIN_THIN_SHARE, 100 * MIN_THIN_IOU_GAIN['stdc'], 100 * MIN_THIN_IOU_GAIN['hrnet']),
          '* Every candidate, both kinds and both networks: the error drops, **no counted class loses more than %.0f '
          'IoU points**, the class-weighted gain is positive and the thin gain is not negative.' % (100 * MAX_CLASS_DROP),
          '* Clean images rank first: Ours also does not visibly hurt an area that was right (a class can gain in one '
          'place and lose in another, which per-class IoU cannot see). Visible broken pixels (baseline right, Ours '
          'wrong, in an area big enough to see -- survives an opening of radius image width / %d px) <= %.2f x the '
          'visible fixed pixels for STDC-Seg and <= %.2f x for HRNet-W48. `extra/` shows fixed (green) / broken '
          '(red) per network.' % (VISIBLE_DIV, MAX_VIS_BROKEN['stdc'], MAX_VIS_BROKEN['hrnet']),
          '* Score = STDC-Seg gain + %.1f x HRNet-W48 gain, the HRNet term capped at the STDC-Seg one (STDC-Seg is '
          'the main comparison).' % HRNET_WEIGHT,
          '* Rows of one dataset come from different cities / SYNTHIA frame ranges, or RUGD frames >= %d apart, '
          'unless only a non-clean image would satisfy that (clean comes first).' % MIN_FRAME_GAP,
          '* Thin classes: %s.' % '; '.join('%s %s' % (DS_TITLE[ds], ', '.join(names(ds)[c] for c in thin_classes(ds)))
                                            for ds in args.datasets) +
          ' (The dataset\'s thin classes that also gained in the per-class table, bnd r1 >= +1.0 and full >= 0.)', '',
          '## Rows', '',
          '| Row | Dataset | Kind | Image | Class gain (STDC-Seg / HRNet-W48) | Thin gain | Worst class | '
          'STDC-Seg error | HRNet-W48 error | Clean | Chosen | Folder |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    f = lambda b, o, e: ('%.1f%% -> %.1f%%' if e else '%.1f -> %.1f') % (100 * b, 100 * o) if o is not None else 'n/a'
    for i, r in enumerate(rows, 1):
        m = r['m']
        p = lambda v: '%+.1f' % (100 * v) if v is not None else 'n/a'
        md.append('| %d | %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s | `%s/` |' % (
            i, DS_TITLE[r['dataset']], r['kind'], r['name'],
            ' / '.join(p(gain(m, 'overall', h)) for h in HOSTS), ' / '.join(p(gain(m, 'thin', h)) for h in HOSTS),
            ' / '.join(p(worst(m, h)[0]) for h in HOSTS),
            f(m['stdc']['err_b'], m['stdc']['err_o'], 1), f(m['hrnet']['err_b'], m['hrnet']['err_o'], 1),
            'yes' if clean(m) else 'no', r['src'], osp.basename(r['folder'])))
    md += ['', 'Gains and worst class in IoU points, Ours minus baseline (worst class: largest per-class loss, 0 if '
           'none lost). Error: share of labelled pixels wrong, baseline -> Ours.', '',
           '### Per-class IoU change, every counted class (Ours minus baseline, IoU points)', '']
    for i, r in enumerate(rows, 1):
        md += ['* row %d `%s`' % (i, r['name'])] + ['  * %s: %s' % (HOST_TITLE[h], class_changes(r['dataset'], r['m'], h))
                                                     for h in HOSTS]
    md += ['',
           '## Alternatives (best first; swap one in with `--pick <dataset>:<kind>:<image>`)', '']
    for (ds, kind), ns in alts.items():
        md.append('* %s %s: %s' % (DS_TITLE[ds], kind, ', '.join('`%s`' % n for n in ns)))
    md += ['', '## Files', '',
           '| File | What it is |', '|---|---|',
           '| `PREVIEW_qualitative.png` | all rows assembled, reference only (the figure is built in PowerPoint) |']
    for r in rows:
        md.append('| `%s/` | row: %s (x%d enlargement, nearest for label maps); `extra/` = pixels Ours fixed '
                  '(green) / broke (red), per network, for choosing only |' % (
                      osp.basename(r['folder']), ', '.join(r['files'].values()), r['k']))
    md += ['| `%s` | colour legend |' % l for l in legends]
    md += ['| `candidates/candidates_<dataset>_<kind>.png` | top %d alternatives with their numbers |' % args.top,
           '| `ranking_<dataset>.csv` | every val image: errors, fixed / broken pixels, thin IoU, class-weighted gains, worst class drop and both scores, sorted by overall score |',
           '', 'Columns of every row folder: 1 RGB, 2 GT (black = ignore), 3 STDC-Seg, 4 STDC-Seg + Ours, '
           '5 HRNet-W48, 6 HRNet-W48 + Ours. Predictions are black where the GT is ignore.']
    common.write_md(osp.join(out, 'SELECTION.md'), md)


def write_log(out, rows):
    logf = osp.join(QROOT, 'ROWS_LOG.md')
    head = [] if osp.isfile(logf) else [
        '# Figure A row sets', '', 'One entry per `gen_qual_rows.py` run, newest last. Every folder is kept; '
        'its SELECTION.md says why each row was chosen. `fig_qualitative/` is the original set from '
        '`gen_qualitative.py`.', '']
    with open(logf, 'a') as fh:
        fh.write('\n'.join(head + ['## %s  (`%s/`)' % (common.now(), osp.basename(out)), '',
                                   '`%s`' % ' '.join(['python gen_qual_rows.py'] + sys.argv[1:]), ''] +
                           ['* row %d: %s %s `%s`' % (i, DS_TITLE[r['dataset']], r['kind'], r['name'])
                            for i, r in enumerate(rows, 1)] + ['', '']))
    print('  wrote %s' % osp.relpath(logf, common.STDC_ROOT))


if __name__ == '__main__':
    main()
