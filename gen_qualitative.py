#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Panels for the two image figures, to be assembled in PowerPoint:

  A. Qualitative comparison (cf. ref_images/qualitative/QualitativeData.png)
     rows    : 2 Cityscapes, 2 SYNTHIA, 2 RUGD (val images)
     columns : RGB | GT | STDC-Seg | STDC-Seg + Ours | HRNet-W48 | HRNet-W48 + Ours
               (HRNet was trained on Cityscapes only -> its columns exist for Cityscapes rows only)
     models  : baseline and HI (RSR + BPM) only: STDC I0/HI1, Synthia/Synthia-HI1,
               RUGD/RUGD-HI1, HRNet baseline/HI1.

  B. Framework diagram (cf. ref_images/qualitative/duet_framework.png)
     every hatched placeholder of the diagram, filled with the REAL tensors of
     STDC-Seg + Ours (HI1) on one Cityscapes val image: input, context
     features, F, Z, Z', prediction, GT, BPM weight map, loss map, OHEM-selected
     pixels, plus optional extras (Z~', P, Delta, pixels changed by RSR, ...).

Images are chosen automatically by the per-image boundary-band (r=1) accuracy
gain of Ours over the baseline (for Cityscapes, the smaller of the STDC and
HRNet gains, so both pairs show a visible difference), one per city / sequence.
Override with --pick; browse alternatives in fig_qualitative/candidates/.
The GUIDE.md written next to the panels explains every file and how to assemble
both figures.

USAGE -- run with the stdcseg env (needs numpy, PIL, matplotlib):
    /home/husky/anaconda3/envs/stdcseg/bin/python gen_qualitative.py
    ... --pick cityscapes:frankfurt_000001_054640 --pick rugd:park-8_01234   # force rows
    ... --strategy median          # typical instead of best-case images
    ... --framework_image munster_000071_000019                              # framework image
    ... --render_only              # skip GPU work, re-render from cache

OUTPUT  resultData/qualitative/  (see GUIDE.md there)
"""
import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from paper import common

QROOT = osp.join(common.RESULT_DIR, 'qualitative')
QCACHE = osp.join(QROOT, 'cache')

CITYSCAPES_PAL = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                  (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                  (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                  (0, 80, 100), (0, 0, 230), (119, 11, 32)]
LEGEND_NAMES_CS = ['road', 'sidew.', 'build.', 'wall', 'fence', 'pole', 'tr. light', 'tr. sign', 'veget.',
                   'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'm.bike', 'bike']
# RUGD colours from data/rugd/meta.json (DatasetNinja export), in trainId order
RUGD_HEX = ['404040', '00FF80', '66FFFF', 'FF0000', 'FF99CC', '65650B', 'FF007F', '6C4014', '6600CC',
            '006600', 'FF8000', '660000', '994C00', 'CC99FF', '72552F', '009999', '99CCFF', '666600',
            'FFE5CC', '006666', '0000FF', '00FF00', 'FFFF00', '0080FF']
RUGD_PAL = [tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) for h in RUGD_HEX]
PALETTE = dict(cityscapes=CITYSCAPES_PAL, synthia=CITYSCAPES_PAL, rugd=RUGD_PAL)

# (host, baseline key, ours key, column labels) per dataset -- baseline and HI only
RUNS = OrderedDict([
    ('cityscapes', [('stdc', 'I0', 'HI1', 'stdc'), ('hrnet', 'baseline', 'HI1', 'hrnet')]),
    ('synthia',    [('stdc', 'Synthia', 'Synthia-HI1', 'stdc')]),
    ('rugd',       [('stdc', 'RUGD', 'RUGD-HI1', 'stdc')]),
])
COL_TITLES = OrderedDict([('rgb', 'RGB'), ('gt', 'GT'), ('stdc', 'STDC-Seg'), ('stdc_ours', 'STDC-Seg + Ours'),
                          ('hrnet', 'HRNet-W48'), ('hrnet_ours', 'HRNet-W48 + Ours')])
DS_TITLE = dict(cityscapes='Cityscapes', synthia='SYNTHIA', rugd='RUGD')
FRAMEWORK_KEY = 'HI1'

FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def mkdir(d):
    if not osp.isdir(d):
        os.makedirs(d)
    return d


# ---------------------------------------------------------------------------
# GPU work (delegated to paper/qual_worker.py in each host's env)
# ---------------------------------------------------------------------------
def stats_file(host, key):
    return osp.join(QCACHE, 'stats', '%s__%s.json' % (host, key))


def stats_valid(host, key):
    f = stats_file(host, key)
    if not osp.isfile(f):
        return False
    with open(f) as fh:
        return json.load(fh).get('fingerprint') == common.fingerprint(host, key)


def ensure_predictions(datasets, force):
    todo = OrderedDict()
    for ds in datasets:
        for host, b, o, _ in RUNS[ds]:
            for k in (b, o):
                if force or not stats_valid(host, k):
                    todo.setdefault(host, []).append(k)
    for host in ('stdc', 'hrnet'):          # stdc first: it also writes GT + RGB index
        if todo.get(host):
            cmd = [common.ENV_PY[host], '-m', 'paper.qual_worker', '--host', host, '--keys'] + todo[host]
            print('[%s] predicting: %s' % (host, ' '.join(todo[host])))
            subprocess.check_call(cmd, cwd=common.STDC_ROOT)


def ensure_framework(name, force):
    p = osp.join(QCACHE, 'framework_%s_%s.npz' % (FRAMEWORK_KEY, name))
    if force or not osp.isfile(p):
        cmd = [common.ENV_PY['stdc'], '-m', 'paper.qual_worker', '--host', 'stdc', '--keys', FRAMEWORK_KEY,
               '--framework', name]
        print('[stdc] framework tensors for %s' % name)
        subprocess.check_call(cmd, cwd=common.STDC_ROOT)
    return p


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def load_stats(host, key):
    with open(stats_file(host, key)) as fh:
        return json.load(fh)['images']


def group_of(ds, name):
    if ds == 'cityscapes':
        return name.split('_')[0]                      # city
    if ds == 'rugd':
        return name.rsplit('_', 1)[0]                  # video sequence
    return int(name) // 150 if name.isdigit() else name   # SYNTHIA: nearby frames look alike


def score_images(ds):
    """-> list of dict(name, score, gains) sorted by score desc."""
    per_host = []
    for host, b, o, _ in RUNS[ds]:
        per_host.append((host, load_stats(host, b), load_stats(host, o)))
    names = set(per_host[0][1])
    for _, sb, so in per_host:
        names &= set(sb) & set(so)
    out = []
    for n in names:
        gains, scores = {}, []
        for host, sb, so in per_host:
            b, o = sb[n], so[n]
            if b[2] == 0:
                break
            g_band = (o[3] - b[3]) / float(b[2])
            g_thin = (o[5] - b[5]) / float(b[4]) if b[4] >= 500 else 0.0
            g_full = (o[1] - b[1]) / float(max(b[0], 1))
            gains[host] = dict(band=g_band, thin=g_thin, full=g_full)
            scores.append(g_band + 0.5 * g_thin)
        else:
            out.append(dict(name=n, score=min(scores), gains=gains))
    out.sort(key=lambda d: -d['score'])
    for i, d in enumerate(out):
        d['rank'] = i + 1
        d['n'] = len(out)
    return out


def choose(ds, ranked, n_rows, strategy, picks):
    by_name = {d['name']: d for d in ranked}
    chosen = [by_name[p] for p in picks if p in by_name]
    for p in picks:
        if p not in by_name:
            print('[%s] --pick %s not found in val set' % (ds, p))
    if strategy == 'median':
        med = ranked[len(ranked) // 2]['score']
        pool = sorted(ranked, key=lambda d: abs(d['score'] - med))
    else:
        pool = [d for d in ranked if all(g['band'] > 0 for g in d['gains'].values())]
    used = set(group_of(ds, d['name']) for d in chosen)
    for d in pool:
        if len(chosen) >= n_rows:
            break
        if d in chosen or group_of(ds, d['name']) in used:
            continue
        chosen.append(d)
        used.add(group_of(ds, d['name']))
    for d in pool:                     # fewer groups than rows: allow repeats
        if len(chosen) >= n_rows:
            break
        if d not in chosen:
            chosen.append(d)
    return chosen[:n_rows]


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------
def colorize(lbl, pal):
    lut = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(pal):
        lut[i] = c
    return lut[lbl]                    # 255 (ignore) stays black


def load_label(path):
    return np.array(Image.open(path))


def load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def resize(arr, w, nearest):
    h = int(round(arr.shape[0] * w / float(arr.shape[1])))
    return np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST if nearest else Image.LANCZOS))


def to_size(arr, hw, nearest=True):
    return np.array(Image.fromarray(arr).resize((hw[1], hw[0]), Image.NEAREST if nearest else Image.BILINEAR))


def save(arr, path, width=None, nearest=True):
    if width is not None and arr.shape[1] > width:
        arr = resize(arr, width, nearest)
    Image.fromarray(arr).save(path)


def dashed_box(arr, box, color=(255, 255, 255)):
    im = Image.fromarray(arr.copy())
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = box
    lw = max(2, arr.shape[1] // 320)
    dash = max(6, arr.shape[1] // 90)
    for (a, b, horiz, fixed) in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        p = a
        while p < b:
            q = min(p + dash, b)
            d.line([(p, fixed), (q, fixed)] if horiz else [(fixed, p), (fixed, q)], fill=color, width=lw)
            p += 2 * dash
    return np.array(im)


def best_window(score, bh, bw, step=4):
    """top-left of the bh x bw window with the largest sum of `score` (coarse grid)."""
    s = score[::step, ::step]
    h, w = s.shape
    kh, kw = max(1, bh // step), max(1, bw // step)
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = s.cumsum(0).cumsum(1)
    sums = ii[kh:, kw:] - ii[:-kh, kw:] - ii[kh:, :-kw] + ii[:-kh, :-kw]
    y, x = np.unravel_index(np.argmax(sums), sums.shape)
    x, y = int(x) * step, int(y) * step
    return x, y, x + bw, y + bh


def boundary_focus(gt, thin, radius=3):
    """weight map: 0 away from GT boundaries, 1 within `radius` px of one, 3 on thin classes there"""
    import cv2
    valid = gt != 255
    e = np.zeros(gt.shape, np.uint8)
    dx = (gt[:, 1:] != gt[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    dy = (gt[1:, :] != gt[:-1, :]) & valid[1:, :] & valid[:-1, :]
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:, :] |= dy
    e[:-1, :] |= dy
    band = cv2.dilate(e, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    return band.astype(np.float32) * (1.0 + 2.0 * np.isin(gt, thin))


def crop(arr, box):
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def heat(x, cmap='magma', vmax=None, mask=None):
    from matplotlib import cm
    x = x.astype(np.float32)
    vmax = vmax if vmax is not None else max(np.percentile(x[mask > 0] if mask is not None else x, 99), 1e-6)
    rgb = (cm.get_cmap(cmap)(np.clip(x / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    if mask is not None:
        rgb[mask == 0] = 0
    return rgb


def overlay(rgb, masks_colors, dim=0.35):
    out = (rgb.astype(np.float32) * dim).astype(np.uint8)
    for m, c in masks_colors:
        out[m > 0] = c
    return out


def text_color(c):
    return (0, 0, 0) if 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] > 140 else (255, 255, 255)


def legend(names, colors, path, cell_w=120, cell_h=44, per_row=None):
    per_row = per_row or len(names)
    rows = (len(names) + per_row - 1) // per_row
    im = Image.new('RGB', (cell_w * per_row, cell_h * rows), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fnt = font(18)
    for i, (n, c) in enumerate(zip(names, colors)):
        x, y = (i % per_row) * cell_w, (i // per_row) * cell_h
        d.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], fill=c)
        tw, th = d.textsize(n, font=fnt)
        d.text((x + (cell_w - tw) // 2, y + (cell_h - th) // 2), n, fill=text_color(c), font=fnt)
    im.save(path)


# ---------------------------------------------------------------------------
# figure A
# ---------------------------------------------------------------------------
def gt_path(ds, name):
    return osp.join(QCACHE, 'gt', ds, name + '.png')


def pred_path(host, key, name):
    return osp.join(QCACHE, 'preds', '%s__%s' % (host, key), name + '.png')


def rgb_index(ds):
    with open(osp.join(QCACHE, 'gt', ds, 'index.json')) as fh:
        return json.load(fh)


def render_row(row_i, ds, d, width, mask_ignore, outdir):
    name = d['name']
    pal = PALETTE[ds]
    gt = load_label(gt_path(ds, name))
    rgb = load_rgb(rgb_index(ds)[name])
    if rgb.shape[:2] != gt.shape:
        rgb = to_size(rgb, gt.shape, nearest=False)
    valid = gt != 255
    panels = OrderedDict([('rgb', rgb), ('gt', colorize(gt, pal))])
    improve = np.zeros(gt.shape, np.float32)
    extras = OrderedDict()
    for host, b, o, col in RUNS[ds]:
        pb, po = load_label(pred_path(host, b, name)), load_label(pred_path(host, o, name))
        okb, oko = (pb == gt) & valid, (po == gt) & valid
        imp = oko.astype(np.float32) - okb.astype(np.float32)
        improve += imp
        for lbl, key in ((pb, col), (po, col + '_ours')):
            c = colorize(lbl, pal)
            if mask_ignore:
                c[~valid] = 0
            panels[key] = c
        # green = pixel fixed by Ours, red = pixel broken by Ours (valid pixels only)
        extras['%s_fixed_green_broken_red' % host] = overlay(rgb, [((imp > 0).astype(np.uint8), (0, 230, 0)),
                                                                  ((imp < 0).astype(np.uint8), (255, 40, 40))])
    H, W = gt.shape
    # place the box on BOUNDARY / THIN-structure corrections (the paper's claim), not on
    # large interior regions: only pixels within 3 px of a GT boundary count, thin classes x3
    box = best_window(improve * boundary_focus(gt, THIN_OF[ds]), int(0.32 * H), int(0.22 * W))
    folder = mkdir(osp.join(outdir, 'row%d_%s__%s' % (row_i, ds, name)))
    for sub in ('boxed', 'zoom', 'extra'):
        mkdir(osp.join(folder, sub))
    files = OrderedDict()
    for j, (key, arr) in enumerate(panels.items(), 1):
        fn = '%d_%s.png' % (j, key)
        nearest = key != 'rgb'
        save(arr, osp.join(folder, fn), width, nearest)
        save(dashed_box(arr, box), osp.join(folder, 'boxed', fn), width, nearest)
        z = crop(arr, box)
        save(resize(z, 512, nearest), osp.join(folder, 'zoom', fn))
        files[key] = fn
    for key, arr in extras.items():
        save(arr, osp.join(folder, 'extra', key + '.png'), width, False)
    return dict(folder=osp.relpath(folder, QROOT), files=files, box=list(box), size=[H, W],
                present=sorted(set(np.unique(gt).tolist()) - {255}))


def fit(arr, w, h, bg=(235, 235, 235)):
    im = Image.fromarray(arr)
    im.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new('RGB', (w, h), bg)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def preview(rows, outpath, boxed=True):
    cw, ch, pad, left, top = 320, 170, 6, 120, 40
    cols = list(COL_TITLES)
    W = left + len(cols) * (cw + pad)
    Hh = top + len(rows) * (ch + pad)
    im = Image.new('RGB', (W, Hh), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fh, fs = font(20), font(15)
    for j, c in enumerate(cols):
        t = COL_TITLES[c]
        tw, _ = d.textsize(t, font=fh)
        d.text((left + j * (cw + pad) + (cw - tw) // 2, 8), t, fill=(0, 0, 0), font=fh)
    for i, r in enumerate(rows):
        y = top + i * (ch + pad)
        d.text((8, y + ch // 2 - 10), DS_TITLE[r['dataset']], fill=(0, 0, 0), font=fh)
        for j, c in enumerate(cols):
            x = left + j * (cw + pad)
            if c in r['files']:
                p = osp.join(QROOT, r['folder'], 'boxed' if boxed else '', r['files'][c])
                im.paste(fit(np.array(Image.open(p)), cw, ch), (x, y))
            else:
                cell = Image.new('RGB', (cw, ch), (235, 235, 235))
                cd = ImageDraw.Draw(cell)
                t = 'not trained on %s' % DS_TITLE[r['dataset']]
                tw, _ = cd.textsize(t, font=fs)
                cd.text(((cw - tw) // 2, ch // 2 - 8), t, fill=(120, 120, 120), font=fs)
                im.paste(cell, (x, y))
    im.save(outpath)


def contact_sheet(ds, ranked, outpath, k=10):
    cw, ch, pad = 240, 120, 4
    cols = ['rgb', 'gt'] + [c for h, b, o, col in RUNS[ds] for c in (col, col + '_ours')]
    pal = PALETTE[ds]
    idx = rgb_index(ds)
    im = Image.new('RGB', (len(cols) * (cw + pad) + 330, k * (ch + pad)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fs = font(14)
    for i, r in enumerate(ranked[:k]):
        n = r['name']
        gt = load_label(gt_path(ds, n))
        arrs = [load_rgb(idx[n]), colorize(gt, pal)]
        for h, b, o, col in RUNS[ds]:
            for key in (b, o):
                c = colorize(load_label(pred_path(h, key, n)), pal)
                c[gt == 255] = 0
                arrs.append(c)
        y = i * (ch + pad)
        for j, a in enumerate(arrs):
            im.paste(fit(a, cw, ch), (j * (cw + pad), y))
        txt = '#%d  %s\n%s' % (r['rank'], n, '\n'.join(
            '%s: band acc %+.1f, thin %+.1f' % (h, 100 * g['band'], 100 * g['thin']) for h, g in r['gains'].items()))
        d.text((len(cols) * (cw + pad) + 8, y + 8), txt, fill=(0, 0, 0), font=fs)
    im.save(outpath)


# ---------------------------------------------------------------------------
# figure B
# ---------------------------------------------------------------------------
def render_framework(npz_path, width, outdir):
    z = np.load(npz_path)
    pal = CITYSCAPES_PAL
    lab = z['label']
    H, W = lab.shape
    valid = z['valid']
    rgb = load_rgb(str(z['rgb_path']))
    up = lambda a: to_size(a, (H, W), nearest=True)              # show true feature-map resolution
    upf = lambda a: to_size((np.asarray(a, np.float32) * 255).astype(np.uint8), (H, W), nearest=True)
    band = z['band']
    sel_colors = lambda sel: overlay(rgb, [(((sel > 0) & (band == 0)).astype(np.uint8), (255, 210, 0)),
                                           (((sel > 0) & (band > 0)).astype(np.uint8), (230, 30, 30))])
    w = z['weight'].astype(np.float32)
    wmap = np.zeros((H, W, 3), np.uint8)
    wmap[w <= 1.0] = (40, 60, 110)
    wmap[w > 1.0] = (255, 200, 40)
    wmap[valid == 0] = 0
    loss = np.log1p(z['loss'].astype(np.float32))
    wl = np.log1p(z['loss'].astype(np.float32) * w)
    vmax = np.percentile(wl[valid > 0], 99.5)
    bnd_vis = np.array(Image.fromarray((z['boundary'] * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)))
    main = OrderedDict([
        ('F01_input_image', (rgb, False)),
        ('F02_context_feat', (upf(z['ctx_pca']), True)),
        ('F03_Z_stride8_logits', (up(colorize(z['Z'], pal)), True)),
        ('F04_F_stride4_features', (upf(z['F_pca']), True)),
        ('F05_Zprime_stride4_refined', (up(colorize(z['Z_prime'], pal)), True)),
        ('F06_prediction', (colorize(z['pred'], pal), True)),
        ('F07_ground_truth', (colorize(lab, pal), True)),
        ('F08_weight_map_w', (wmap, True)),
        ('F09_loss_map', (heat(loss, 'magma', np.percentile(loss[valid > 0], 99.5), valid), False)),
        ('F10_selected_pixels_BPM', (sel_colors(z['sel_bpm']), True)),
    ])
    extra = OrderedDict([
        ('X1_Ztilde_bilinear_x2', (up(colorize(z['Z_tilde'], pal)), True)),
        ('X2_P_projected_features', (upf(z['P_pca']), True)),
        ('X3_Delta_correction_magnitude', (up(heat(z['delta_mag'], 'inferno')), True)),
        ('X4_pixels_changed_by_RSR', (overlay(rgb, [(up(z['rsr_changed']), (0, 255, 255))]), True)),
        ('X5_boundary_set_B', (np.stack([bnd_vis] * 3, -1), True)),
        ('X6_boundary_band_Br', (np.stack([z['band'] * 255] * 3, -1).astype(np.uint8), True)),
        ('X7_weighted_loss_w_times_l', (heat(wl, 'magma', vmax, valid), False)),
        ('X8_selected_pixels_stock_OHEM', (sel_colors(z['sel_stock']), True)),
    ])
    # zoom window: square around where RSR changed the most pixels on thin classes
    changed = up(z['rsr_changed']).astype(np.float32)
    thin = np.isin(lab, THIN_CS).astype(np.float32)
    side = H // 3
    box = best_window(changed * (0.25 + thin), side, side)
    for sub in ('', 'zoom', 'extras', 'extras/zoom'):
        mkdir(osp.join(outdir, sub))
    for group, sub in ((main, ''), (extra, 'extras')):
        for k, (arr, nearest) in group.items():
            save(arr, osp.join(outdir, sub, k + '.png'), width, nearest)
            save(resize(crop(arr, box), 512, nearest), osp.join(outdir, sub, 'zoom', k + '.png'))
    save(dashed_box(rgb, box), osp.join(outdir, 'zoom', 'where_the_zoom_is.png'), width, False)
    sel_w, sel_u = z['sel_bpm'] > 0, z['sel_stock'] > 0
    meta = dict(image=str(z['name']), model=str(z['key']), gamma_res_scale=float(z['gamma']),
                model_input_hw=[int(v) for v in z['input_hw']], label_hw=[H, W],
                Z_hw=list(z['Z'].shape), Zprime_hw=list(z['Z_prime'].shape),
                bnd_radius=int(z['radius']), w_bnd=float(z['w_bnd']), ohem_n_min=int(z['n_min']),
                ohem_kept_bpm=int(sel_w.sum()), ohem_kept_stock=int(sel_u.sum()),
                pct_kept_in_boundary_band_bpm=100.0 * float((sel_w & (band > 0)).sum()) / max(int(sel_w.sum()), 1),
                pct_kept_in_boundary_band_stock=100.0 * float((sel_u & (band > 0)).sum()) / max(int(sel_u.sum()), 1),
                pct_of_valid_pixels_in_band=100.0 * float((band > 0).sum()) / max(int(valid.sum()), 1),
                pixels_changed_by_rsr_at_stride4=int(z['rsr_changed'].sum()),
                zoom_box_xyxy=list(box))
    with open(osp.join(outdir, 'framework_meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    return list(main), list(extra), meta


THIN_CS = common.THIN
THIN_OF = {ds: common.DATASETS[ds]['thin'] for ds in RUNS}


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------
def write_guide(rows, fw, strategy, mask_ignore):
    main, extra, meta = fw if fw else ([], [], None)
    L = ['# Qualitative figures -- what every file is and how to assemble them', '',
         'Generated %s by `gen_qualitative.py`. All paths are relative to `resultData/qualitative/`.' % common.now(),
         'Models: baseline and "Ours" (= HI: RSR + BPM) only. STDC-Seg: I0 / HI1 (Cityscapes), Synthia / '
         'Synthia-HI1, RUGD / RUGD-HI1. HRNet-W48: baseline / HI1 (Cityscapes only -- the only dataset HRNet was '
         'trained on).', '',
         '## Folder map', '',
         '```',
         'fig_qualitative/                  Figure A (cf. ref_images/qualitative/QualitativeData.png)',
         '  PREVIEW_qualitative.png         auto-assembled preview of the whole figure (reference only)',
         '  PREVIEW_qualitative_clean.png   same without the dashed boxes',
         '  legend_cityscapes_synthia.png   colour legend for Cityscapes + SYNTHIA rows (same classes)',
         '  legend_rugd.png                 colour legend for RUGD rows',
         '  rowN_<dataset>__<image>/        one folder per figure row',
         '    1_rgb.png 2_gt.png 3_stdc.png 4_stdc_ours.png [5_hrnet.png 6_hrnet_ours.png]',
         '    boxed/   same panels with a dashed box around the region where Ours helps most',
         '    zoom/    the boxed region cropped (512 px wide) -- for insets',
         '    extra/   diagnostics: green = pixel fixed by Ours, red = pixel broken by Ours (not for the paper grid)',
         '  candidates/candidates_<dataset>.png   top-10 alternative images per dataset, with scores',
         'fig_framework/                    Figure B (cf. ref_images/qualitative/duet_framework.png)',
         '  F01..F10_*.png                  one image per hatched box in the diagram (table below)',
         '  zoom/                           the same images cropped to one square region (recommended for the boxes)',
         '  extras/ (+ extras/zoom/)        optional panels for tensors the diagram names but has no box for',
         '  framework_meta.json             numbers for the caption (gamma, OHEM statistics, sizes)',
         'manifest.json                     machine-readable version of this guide',
         '```', '']

    # figure A
    L += ['## Figure A -- qualitative comparison', '',
          '### Rows (chosen automatically, strategy = `%s`)' % strategy, '']
    L += common.md_table(
        ['Row', 'Dataset', 'Image', 'Folder', 'Rank / pool', 'Boundary-band acc. gain (points)'],
        [[str(i + 1), DS_TITLE[r['dataset']], r['name'], '`%s/`' % r['folder'], '%d / %d' % (r['rank'], r['n']),
          ', '.join('%s %+.1f' % ('STDC' if h == 'stdc' else 'HRNet', 100 * g['band']) for h, g in r['gains'].items())]
         for i, r in enumerate(rows)]) + ['']
    L += ['"Boundary-band acc. gain" = change in the fraction of correctly labelled pixels within 1 px of a GT '
          'boundary in that image (Ours minus baseline). Rank 1 = the largest gain in the val set. For Cityscapes '
          'the score is the smaller of the STDC and HRNet gains, so both model pairs show a visible difference. '
          'One image per city (Cityscapes), video sequence (RUGD) or frame range (SYNTHIA).', '',
          '### Columns (same file names in every row folder)', '']
    L += common.md_table(['#', 'File', 'Column title in the figure', 'What it is'], [
        ['1', '1_rgb.png', 'RGB', 'input image'],
        ['2', '2_gt.png', 'GT', 'ground truth; black = ignore / unlabeled'],
        ['3', '3_stdc.png', 'STDC-Seg', 'baseline prediction'],
        ['4', '4_stdc_ours.png', 'STDC-Seg + Ours', 'RSR + BPM prediction'],
        ['5', '5_hrnet.png', 'HRNet-W48', 'baseline prediction (Cityscapes rows only)'],
        ['6', '6_hrnet_ours.png', 'HRNet-W48 + Ours', 'RSR + BPM prediction (Cityscapes rows only)']]) + ['']
    L += ['Predictions are %s. Colours: Cityscapes palette for Cityscapes and SYNTHIA (SYNTHIA uses Cityscapes '
          'classes), RUGD\'s own palette for RUGD.' % ('masked to black wherever the GT is ignore, so they read '
                                                       'like the GT' if mask_ignore else 'shown unmasked'), '',
          '### How to assemble it in PowerPoint', '',
          '1. Open `fig_qualitative/PREVIEW_qualitative.png` to see the target layout.',
          '2. Layout choice for the missing HRNet cells on SYNTHIA/RUGD rows (HRNet exists for Cityscapes only):',
          '   - **Recommended:** two blocks. Top block: the 2 Cityscapes rows x 6 columns. Bottom block: the 4 '
          'SYNTHIA/RUGD rows x 4 columns (RGB, GT, STDC-Seg, STDC-Seg + Ours), separated by a dashed line like the '
          'reference figure. No empty cells.',
          '   - Alternative: one 6-column grid with the HRNet cells on SYNTHIA/RUGD rows left blank or marked "--".',
          '3. Insert the row images left to right in file order 1..6. Use either the plain files or the `boxed/` '
          'versions (the box is the same in every column of a row, placed where Ours fixes the most pixels). If you '
          'prefer your own boxes, use the plain files and draw a dashed rectangle in PowerPoint.',
          '4. Aspect ratios differ: Cityscapes 2:1, SYNTHIA 1280x760 (1.68:1), RUGD 688x550 (1.25:1). Either crop '
          'every image in a row identically (Picture Format -> Crop, then copy the crop to the other panels of the '
          'row) or let rows have different heights. Always crop all panels of a row the same way.',
          '5. Put a row label on the left (Cityscapes / SYNTHIA / RUGD), column titles on top (table above), and the '
          'legend strip(s) at the bottom: `legend_cityscapes_synthia.png` and `legend_rugd.png`.',
          '6. Optional insets: the `zoom/` crops can go below or beside a row to magnify the boxed region.',
          '7. Other image choices: look at `candidates/candidates_<dataset>.png`, then re-run with '
          '`--pick <dataset>:<image>`.', '',
          'Caption suggestion: "Qualitative results on Cityscapes, SYNTHIA and RUGD val. Dashed boxes mark regions '
          'where Ours corrects boundaries and thin structures. HRNet-W48 was trained on Cityscapes only." If the '
          'rows are best-case picks (strategy `top`), say they were "selected for visible differences"; '
          '`--strategy median` gives typical images instead.', '']

    # figure B
    if meta:
        L += ['## Figure B -- framework diagram (STDC-Seg + Ours, image `%s`)' % meta['image'], '',
              'Every image is a real intermediate tensor of the HI1 model on this Cityscapes val image. The model '
              'sees the image at %dx%d (scale 0.75). Feature maps are drawn at their true resolution '
              '(nearest-neighbour upsampling), so the blockiness of Z vs Z\' IS the point of RSR. For the boxes in '
              'the diagram, use the `zoom/` versions: the full frames are 2:1, the boxes are square, and the zoom '
              'region (see `zoom/where_the_zoom_is.png`) is where RSR changed the most thin-structure pixels.'
              % tuple(meta['model_input_hw']), '',
              '### Main slots (one per hatched box in duet_framework.png)', '']
        L += common.md_table(['Diagram box (label in the figure)', 'File', 'What it shows', 'How it was made'], [
            ['Input image *I*', 'F01_input_image.png', 'the val image', 'RGB as loaded'],
            ['context feat.', 'F02_context_feat.png', 'context-path features fed to the decoder (FFM)',
             'feat_cp8 (128 ch, stride 8) -> first 3 PCA components as RGB'],
            ['*Z* (stride rho)', 'F03_Z_stride8_logits.png', 'host logits before RSR, as a label map',
             'argmax of conv_out output, stride 8 (%dx%d grid)' % tuple(meta['Z_hw'])],
            ['*F* (stride rho/2)', 'F04_F_stride4_features.png', 'high-res detail features RSR consumes',
             'feat_res4 (64 ch, stride 4) -> PCA RGB'],
            ["*Z'* (stride rho/2)", 'F05_Zprime_stride4_refined.png', 'logits after RSR, as a label map',
             "argmax of Z' = Z~' + gamma*Delta, stride 4 (%dx%d grid)" % tuple(meta['Zprime_hw'])],
            ['Prediction *Y^*', 'F06_prediction.png', 'final full-resolution prediction',
             "Z' upsampled to 1024x2048, argmax (not masked)"],
            ['Ground truth *Y*', 'F07_ground_truth.png', 'GT labels', 'black = ignore'],
            ['*w* (Eq. 9) -- "weight map"', 'F08_weight_map_w.png', 'BPM boundary weights',
             'yellow = w_bnd (%.0f) within r=%d px of a GT boundary, blue = 1, black = ignore'
             % (meta['w_bnd'], meta['bnd_radius'])],
            ['*l_i* -- "loss map"', 'F09_loss_map.png', 'per-pixel cross-entropy', 'log(1+CE), magma colour map'],
            ['bottom-left box ("Output image *I*")', 'F10_selected_pixels_BPM.png',
             'pixels BPM\'s OHEM keeps (S_w)', 'red = kept & inside boundary band, yellow = kept elsewhere, '
             'on a darkened RGB; n_min = %d (1/16 of pixels, as in train.py)' % meta['ohem_n_min']]]) + ['']
        L += ['**The bottom-left box is mislabeled in the current diagram.** It reads "Output image *I*" with an '
              '"input image" placeholder, but BPM outputs a loss, not an image. Relabel it "OHEM-selected pixels '
              '*S_w*" and fill it with F10. That makes the BPM row end in a picture of what BPM does. Stronger '
              'option: put `extras/X8_selected_pixels_stock_OHEM.png` next to it as "stock OHEM" for contrast. '
              'Otherwise delete the box and let the arrow end at *L_BPM*.', '',
              'Numbers for the caption (from `framework_meta.json`): on this image, %.0f%% of valid pixels lie in the '
              'r=%d boundary band. Stock OHEM keeps %.0f%% of its selected pixels in that band; BPM keeps %.0f%%. '
              'RSR changed %d stride-4 pixels; gamma (res_scale) = %+.2f.' % (
                  meta['pct_of_valid_pixels_in_band'], meta['bnd_radius'], meta['pct_kept_in_boundary_band_stock'],
                  meta['pct_kept_in_boundary_band_bpm'], meta['pixels_changed_by_rsr_at_stride4'],
                  meta['gamma_res_scale']), '',
              '### Optional extras (`extras/`, tensors the diagram names but has no box for)', '']
        L += common.md_table(['Where it would go', 'File', 'What it shows'], [
            ["*Z~'* (after \"bilinear x2\")", 'X1_Ztilde_bilinear_x2.png',
             'Z upsampled x2 before correction: compare with F05 to see what RSR adds'],
            ['*P* (after ConvBNReLU)', 'X2_P_projected_features.png', 'projected F inside RSR (PCA RGB)'],
            ['*Delta* (before x gamma)', 'X3_Delta_correction_magnitude.png',
             '|gamma*Delta| per pixel: where the correction acts (inferno)'],
            ["next to *Z'*, or as an inset", 'X4_pixels_changed_by_RSR.png',
             "cyan = pixels whose label differs between Z~' and Z' (what RSR fixes)"],
            ['boundary set *B* (Eq. 8)', 'X5_boundary_set_B.png', 'GT boundary pixels (drawn 3 px wide for visibility)'],
            ['dilate to *B_r*', 'X6_boundary_band_Br.png', 'band of radius r (white)'],
            ['after the (x) node, *w_i l_i*', 'X7_weighted_loss_w_times_l.png', 'weighted loss map (magma)'],
            ['contrast panel for F10', 'X8_selected_pixels_stock_OHEM.png', 'what stock OHEM (w = 1) would keep']])
        L += ['', '### How to fill the diagram in PowerPoint', '',
              '1. Open `ref_images/qualitative/duet_framework.pptx`.',
              '2. For each hatched placeholder: right-click -> Change Picture (or insert the file and send the '
              'placeholder backward), using the file in the table above. Use the `zoom/` version for square boxes.',
              '3. Keep the label text above each box (e.g. "*Z* (stride rho)"). The images replace only the hatching.',
              '4. Optional: add a small inset of `extras/zoom/X4_pixels_changed_by_RSR.png` beside *Z\'*, and swap the '
              'bottom-left box for F10 + X8 as described above.',
              '5. Label maps (Z, Z\', prediction, GT) use the Cityscapes palette; the legend is '
              '`fig_qualitative/legend_cityscapes_synthia.png` if the figure needs one.', '']
    L += ['## Regenerating', '',
          '```',
          '/home/husky/anaconda3/envs/stdcseg/bin/python gen_qualitative.py                 # everything',
          '... --pick cityscapes:<image> --pick rugd:<image>                                   # choose rows',
          '... --framework_image <cityscapes image>                                          # framework image',
          '... --strategy median                                                             # typical rows',
          '... --render_only                                                                 # no GPU, re-render',
          '```']
    common.write_md(osp.join(QROOT, 'GUIDE.md'), L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=list(RUNS), choices=list(RUNS))
    ap.add_argument('--rows_per_dataset', type=int, default=2)
    ap.add_argument('--strategy', default='top', choices=('top', 'median'))
    ap.add_argument('--pick', action='append', default=[], help='<dataset>:<image id>, repeatable')
    ap.add_argument('--framework_image', default=None,
                    help='Cityscapes val image for the framework figure (default: first Cityscapes row)')
    ap.add_argument('--width', type=int, default=1024, help='max width of saved full-frame panels')
    ap.add_argument('--no_mask', action='store_true', help='do not black out GT-ignore pixels in predictions')
    ap.add_argument('--render_only', action='store_true', help='skip GPU work; use the cache')
    ap.add_argument('--force', action='store_true', help='recompute predictions / framework tensors')
    args = ap.parse_args()

    if not args.render_only:
        ensure_predictions(args.datasets, args.force)
    figA = osp.join(QROOT, 'fig_qualitative')
    if osp.isdir(figA):
        shutil.rmtree(figA)            # generated content only; old row folders would linger
    mkdir(osp.join(figA, 'candidates'))

    picks = {}
    for p in args.pick:
        ds, n = p.split(':', 1)
        picks.setdefault(ds, []).append(n)
    rows, manifest_rows = [], []
    for ds in args.datasets:
        ranked = score_images(ds)
        contact_sheet(ds, ranked, osp.join(figA, 'candidates', 'candidates_%s.png' % ds))
        for d in choose(ds, ranked, args.rows_per_dataset, args.strategy, picks.get(ds, [])):
            r = render_row(len(rows) + 1, ds, d, args.width, not args.no_mask, figA)
            r.update(dataset=ds, name=d['name'], rank=d['rank'], n=d['n'], score=d['score'], gains=d['gains'])
            rows.append(r)
            print('  row %d  %-10s %-32s rank %d/%d' % (len(rows), ds, d['name'], d['rank'], d['n']))
    preview(rows, osp.join(figA, 'PREVIEW_qualitative.png'), boxed=True)
    preview(rows, osp.join(figA, 'PREVIEW_qualitative_clean.png'), boxed=False)
    legend(LEGEND_NAMES_CS + ['n/a'], CITYSCAPES_PAL + [(0, 0, 0)],
           osp.join(figA, 'legend_cityscapes_synthia.png'), per_row=10)
    rugd_cls = sorted(set(c for r in rows if r['dataset'] == 'rugd' for c in r['present']))
    if rugd_cls:
        legend([common.RUGD_CLASSES[c] for c in rugd_cls] + ['n/a'], [RUGD_PAL[c] for c in rugd_cls] + [(0, 0, 0)],
               osp.join(figA, 'legend_rugd.png'), cell_w=130, per_row=8)

    fw = None
    if 'cityscapes' in args.datasets:
        name = args.framework_image or next(r['name'] for r in rows if r['dataset'] == 'cityscapes')
        npz = osp.join(QCACHE, 'framework_%s_%s.npz' % (FRAMEWORK_KEY, name))
        if not args.render_only or osp.isfile(npz):
            npz = ensure_framework(name, args.force) if not args.render_only else npz
            figB = osp.join(QROOT, 'fig_framework')
            if osp.isdir(figB):
                shutil.rmtree(figB)
            fw = render_framework(npz, args.width, mkdir(figB))
    with open(osp.join(QROOT, 'manifest.json'), 'w') as fh:
        json.dump(dict(generated=common.now(), strategy=args.strategy, rows=rows,
                       framework=fw[2] if fw else None), fh, indent=2)
    write_guide(rows, fw, args.strategy, not args.no_mask)
    print('done -> %s' % osp.relpath(QROOT, common.STDC_ROOT))


if __name__ == '__main__':
    main()
