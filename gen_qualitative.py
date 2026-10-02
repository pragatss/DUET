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


def render_row(row_i, ds, d, width, mask_ignore, outdir, extras=False):
    """writes 1_rgb, 2_gt, 3_stdc, 4_stdc_ours[, 5_hrnet, 6_hrnet_ours] for one row.
    extras=True also writes boxed/ (dashed box on the boundary/thin corrections),
    zoom/ (that box cropped) and extra/ (green fixed / red broken diagnostics)."""
    name = d['name']
    pal = PALETTE[ds]
    gt = load_label(gt_path(ds, name))
    rgb = load_rgb(rgb_index(ds)[name])
    if rgb.shape[:2] != gt.shape:
        rgb = to_size(rgb, gt.shape, nearest=False)
    valid = gt != 255
    panels = OrderedDict([('rgb', rgb), ('gt', colorize(gt, pal))])
    improve = np.zeros(gt.shape, np.float32)
    diag = OrderedDict()
    for host, b, o, col in RUNS[ds]:
        pb, po = load_label(pred_path(host, b, name)), load_label(pred_path(host, o, name))
        imp = ((po == gt) & valid).astype(np.float32) - ((pb == gt) & valid).astype(np.float32)
        improve += imp
        for lbl, key in ((pb, col), (po, col + '_ours')):
            c = colorize(lbl, pal)
            if mask_ignore:
                c[~valid] = 0
            panels[key] = c
        diag['%s_fixed_green_broken_red' % host] = overlay(rgb, [((imp > 0).astype(np.uint8), (0, 230, 0)),
                                                                ((imp < 0).astype(np.uint8), (255, 40, 40))])
    H, W = gt.shape
    folder = mkdir(osp.join(outdir, 'row%d_%s__%s' % (row_i, ds, name)))
    files = OrderedDict()
    for j, (key, arr) in enumerate(panels.items(), 1):
        fn = '%d_%s.png' % (j, key)
        save(arr, osp.join(folder, fn), width, key != 'rgb')
        files[key] = fn
    box = None
    if extras:
        # box on BOUNDARY / THIN-structure corrections (within 3 px of a GT boundary, thin classes x3)
        box = best_window(improve * boundary_focus(gt, THIN_OF[ds]), int(0.32 * H), int(0.22 * W))
        for sub in ('boxed', 'zoom', 'extra'):
            mkdir(osp.join(folder, sub))
        for key, arr in panels.items():
            nearest = key != 'rgb'
            save(dashed_box(arr, box), osp.join(folder, 'boxed', files[key]), width, nearest)
            save(resize(crop(arr, box), 512, nearest), osp.join(folder, 'zoom', files[key]))
        for key, arr in diag.items():
            save(arr, osp.join(folder, 'extra', key + '.png'), width, False)
        box = list(box)
    return dict(folder=osp.relpath(folder, QROOT), files=files, box=box, size=[H, W],
                present=sorted(set(np.unique(gt).tolist()) - {255}))


def fit(arr, w, h, bg=(235, 235, 235)):
    im = Image.fromarray(arr)
    im.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new('RGB', (w, h), bg)
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def preview(rows, outpath):
    """Recommended layout: block 1 = rows that have every column (Cityscapes, 6 columns);
    block 2 = rows without HRNet (SYNTHIA, RUGD, 4 columns), under a dashed separator,
    with its own column titles. Reference only -- the paper figure is built in PowerPoint."""
    cw, ch, pad, left, title_h, gap = 320, 170, 6, 120, 34, 30
    blocks = [[r for r in rows if len(r['files']) == len(COL_TITLES)],
              [r for r in rows if len(r['files']) < len(COL_TITLES)]]
    blocks = [b for b in blocks if b]
    ncol = max(len(b[0]['files']) for b in blocks)
    Hh = sum(title_h + len(b) * (ch + pad) for b in blocks) + gap * (len(blocks) - 1)
    im = Image.new('RGB', (left + ncol * (cw + pad), Hh), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fh = font(20)
    y = 0
    for bi, block in enumerate(blocks):
        if bi:
            yy = y - gap // 2
            for x in range(8, im.width - 8, 24):
                d.line([(x, yy), (x + 12, yy)], fill=(120, 120, 120), width=2)
        cols = list(block[0]['files'])
        for j, c in enumerate(cols):
            t = COL_TITLES[c]
            tw, _ = d.textsize(t, font=fh)
            d.text((left + j * (cw + pad) + (cw - tw) // 2, y + 6), t, fill=(0, 0, 0), font=fh)
        y += title_h
        for r in block:
            d.text((8, y + ch // 2 - 10), DS_TITLE[r['dataset']], fill=(0, 0, 0), font=fh)
            for j, c in enumerate(cols):
                p = osp.join(QROOT, r['folder'], r['files'][c])
                im.paste(fit(np.array(Image.open(p)), cw, ch), (left + j * (cw + pad), y))
            y += ch + pad
        y += gap
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
# figure B -- framework panels (several candidate images, high resolution)
# ---------------------------------------------------------------------------
FW_N = 8          # candidate images
ZOOM_LABEL = 512  # zoom side in GT px = 384 model-input px = 48 stride-8 cells = 96 stride-4 cells
ZOOM_OUT = 1536   # zoom output side: every enlargement is an exact integer (GT x3, stride-4 x16, stride-8 x32)
ALIGN = 32        # zoom origin on a 32-GT-px lattice = 24 input px = 3 stride-8 cells: no grid cell is ever split
DPI = (300, 300)
FW_LABELS = OrderedDict([
    ('F01_input_image', 'Input image I'), ('F02_context_feat', 'context feat.'),
    ('F03_Z_stride8_logits', 'Z (stride 8)'), ('F04_F_stride4_features', 'F (stride 4)'),
    ('F05_Zprime_stride4_refined', "Z' (stride 4)"), ('F06_prediction', 'Prediction'),
    ('F07_ground_truth', 'Ground truth Y'), ('F08_weight_map_w', 'w (Eq. 9)'),
    ('F09_loss_map', 'loss map l_i'), ('F10_selected_pixels_BPM', 'S_w: BPM-selected'),
    ('X1_Ztilde_bilinear_x2', "Z~' (bilinear x2)"), ('X2_P_projected_features', 'P'),
    ('X3_Delta_correction_magnitude', '|gamma * Delta|'), ('X4_pixels_changed_by_RSR', 'changed by RSR'),
    ('X5_boundary_set_B', 'B (Eq. 8)'), ('X6_boundary_band_Br', 'B_r'),
    ('X7_weighted_loss_w_times_l', 'w * l'), ('X8_selected_pixels_stock_OHEM', 'stock OHEM-selected'),
    ('X9_prediction_unmasked', 'Prediction (unmasked)'),
])


def fw_npz(name):
    return osp.join(QCACHE, 'framework_%s_%s.npz' % (FRAMEWORK_KEY, name))


def ensure_framework(names, force):
    todo = [n for n in names if force or not osp.isfile(fw_npz(n))]
    if todo:
        cmd = [common.ENV_PY['stdc'], '-m', 'paper.qual_worker', '--host', 'stdc', '--keys', FRAMEWORK_KEY,
               '--framework'] + todo
        print('[stdc] framework tensors for %d image(s)' % len(todo))
        subprocess.check_call(cmd, cwd=common.STDC_ROOT)


def framework_candidates(n, picks):
    """Cityscapes val images where RSR+BPM most improves thin structures at the boundary
    (per-image thin-band accuracy gain, HI1 vs I0), at most ceil(n/3) per city."""
    base, ours = load_stats('stdc', 'I0'), load_stats('stdc', 'HI1')
    info, scored = {}, []
    for nm in base:
        b, o = base[nm], ours[nm]
        info[nm] = dict(thin_gain=(o[5] - b[5]) / float(max(b[4], 1)), band_gain=(o[3] - b[3]) / float(max(b[2], 1)),
                        thin_band_px=b[4])
        if b[4] >= 3000:           # enough thin-structure boundary pixels to show something
            scored.append((info[nm]['thin_gain'] + 0.5 * info[nm]['band_gain'], nm))
    scored.sort(reverse=True)
    out = [p for p in picks if p in base]
    cap = max(1, -(-n // 3))
    per_city = {}
    for p in out:
        per_city[p.split('_')[0]] = per_city.get(p.split('_')[0], 0) + 1
    for _, nm in scored:
        if len(out) >= n:
            break
        city = nm.split('_')[0]
        if nm in out or per_city.get(city, 0) >= cap:
            continue
        out.append(nm)
        per_city[city] = per_city.get(city, 0) + 1
    return out, info


class FwView(object):
    """One output frame: the full image at GT resolution, or a square zoom whose origin
    lies on the ALIGN lattice so stride-4/8 cells map to exact integer pixel blocks."""

    def __init__(self, z, rgb, origin=None):
        self.H, self.W = z['label'].shape
        self.scale = int(z['input_hw'][0]) / float(self.H)          # 0.75
        self.rgb_src = rgb
        if origin is None:
            self.box, self.out = (0, 0, self.W, self.H), (self.H, self.W)
        else:
            y0, x0 = origin
            self.box, self.out = (x0, y0, x0 + ZOOM_LABEL, y0 + ZOOM_LABEL), (ZOOM_OUT, ZOOM_OUT)

    def _resize(self, a, nearest):
        return np.array(Image.fromarray(np.ascontiguousarray(a)).resize(
            (self.out[1], self.out[0]), Image.NEAREST if nearest else Image.LANCZOS))

    def lab(self, a, nearest=True):
        """array at GT resolution"""
        x0, y0, x1, y1 = self.box
        return self._resize(a[y0:y1, x0:x1], nearest)

    def grid(self, a):
        """array on a stride-s grid of the model input (s inferred from its size)"""
        s = int(round(self.H * self.scale)) // a.shape[0]
        f = self.scale / s                                        # GT px -> grid cells
        x0, y0, x1, y1 = [int(round(v * f)) for v in self.box]
        return self._resize(a[y0:y1, x0:x1], True)

    def rgb(self):
        return self.lab(self.rgb_src, nearest=False)


def fw_zoom_origins(z, k=2):
    """k square windows (ALIGN lattice, overlap < 25%) with the most RSR-changed pixels,
    weighted toward thin classes."""
    lab, valid = z['label'], z['valid']
    H, W = lab.shape
    changed = to_size(z['rsr_changed'], (H, W), nearest=True).astype(np.float64)
    score = changed * (0.25 + np.isin(lab, THIN_CS)) * valid
    ii = np.zeros((H + 1, W + 1))
    ii[1:, 1:] = score.cumsum(0).cumsum(1)
    S = ZOOM_LABEL
    cands = sorted(((ii[y + S, x + S] - ii[y, x + S] - ii[y + S, x] + ii[y, x], y, x)
                    for y in range(0, H - S + 1, ALIGN) for x in range(0, W - S + 1, ALIGN)), reverse=True)
    picked = []
    for s, y, x in cands:
        if all(max(0, S - abs(y - py)) * max(0, S - abs(x - px)) < 0.25 * S * S for _, py, px in picked):
            picked.append((s, y, x))
        if len(picked) == k:
            break
    return [(y, x) for _, y, x in picked]


def fw_panels(z, v, full_maps):
    pal = CITYSCAPES_PAL
    band = z['band']
    dim = (v.rgb().astype(np.float32) * 0.35).astype(np.uint8)

    def paint(masks):
        out = dim.copy()
        for m, c in masks:
            out[m > 0] = c
        return out

    def selected(sel):
        return paint([(v.lab(((sel > 0) & (band == 0)).astype(np.uint8)), (255, 210, 0)),
                      (v.lab(((sel > 0) & (band > 0)).astype(np.uint8)), (230, 30, 30))])

    feat = lambda a: v.grid((np.clip(np.asarray(a, np.float32), 0, 1) * 255).astype(np.uint8))
    bnd = z['boundary'] * 255
    if v.out == (v.H, v.W):             # full frame: 1-px lines vanish when scaled down in PPT
        bnd = np.array(Image.fromarray(bnd.astype(np.uint8)).filter(ImageFilter.MaxFilter(3)))
    gray = lambda m: np.stack([m] * 3, -1).astype(np.uint8)
    # final prediction masked like the GT: black where the GT category is not one of the 19
    # evaluated classes (parking, ground, static/dynamic, ego vehicle, borders). Those pixels
    # are ignored by the loss and every metric; internal tensors (Z, Z', ...) stay unmasked.
    masked_pred = colorize(z['pred'], pal)
    masked_pred[z['label'] == 255] = 0
    main = OrderedDict([
        ('F01_input_image', v.rgb()),
        ('F02_context_feat', feat(z['ctx_pca'])),
        ('F03_Z_stride8_logits', v.grid(colorize(z['Z'], pal))),
        ('F04_F_stride4_features', feat(z['F_pca'])),
        ('F05_Zprime_stride4_refined', v.grid(colorize(z['Z_prime'], pal))),
        ('F06_prediction', v.lab(masked_pred)),
        ('F07_ground_truth', v.lab(colorize(z['label'], pal))),
        ('F08_weight_map_w', v.lab(full_maps['wmap'])),
        ('F09_loss_map', v.lab(full_maps['loss'], nearest=False)),
        ('F10_selected_pixels_BPM', selected(z['sel_bpm'])),
    ])
    extra = OrderedDict([
        ('X1_Ztilde_bilinear_x2', v.grid(colorize(z['Z_tilde'], pal))),
        ('X2_P_projected_features', feat(z['P_pca'])),
        ('X3_Delta_correction_magnitude', v.grid(full_maps['delta'])),
        ('X4_pixels_changed_by_RSR', paint([(v.grid(z['rsr_changed']), (0, 255, 255))])),
        ('X5_boundary_set_B', v.lab(gray(bnd))),
        ('X6_boundary_band_Br', v.lab(gray(band * 255))),
        ('X7_weighted_loss_w_times_l', v.lab(full_maps['wloss'], nearest=False)),
        ('X8_selected_pixels_stock_OHEM', selected(z['sel_stock'])),
        ('X9_prediction_unmasked', v.lab(colorize(z['pred'], pal))),
    ])
    return main, extra


def fw_full_maps(z):
    """colour maps normalised over the WHOLE image, so full frame and zooms share one scale"""
    valid = z['valid']
    w = z['weight'].astype(np.float32)
    wmap = np.zeros(w.shape + (3,), np.uint8)
    wmap[w <= 1.0] = (40, 60, 110)
    wmap[w > 1.0] = (255, 200, 40)
    wmap[valid == 0] = 0
    loss = np.log1p(z['loss'].astype(np.float32))
    wl = np.log1p(z['loss'].astype(np.float32) * w)
    return dict(wmap=wmap,
                loss=heat(loss, 'magma', np.percentile(loss[valid > 0], 99.5), valid),
                wloss=heat(wl, 'magma', np.percentile(wl[valid > 0], 99.5), valid),
                delta=heat(z['delta_mag'], 'inferno'))


def save_hq(arr, path):
    Image.fromarray(arr).save(path, dpi=DPI)


def sheet(panels, path, tile_w):
    """labelled contact sheet of one view's panels (quick look only)"""
    items = list(panels.items())
    a0 = items[0][1]
    tile_h = int(round(tile_w * a0.shape[0] / float(a0.shape[1])))
    cols = 6
    rows = (len(items) + cols - 1) // cols
    cap = 26
    im = Image.new('RGB', (cols * (tile_w + 6), rows * (tile_h + cap + 6)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fnt = font(17)
    for i, (k, a) in enumerate(items):
        x, y = (i % cols) * (tile_w + 6), (i // cols) * (tile_h + cap + 6)
        d.text((x + 2, y + 3), '%s  %s' % (k.split('_')[0], FW_LABELS[k]), fill=(0, 0, 0), font=fnt)
        im.paste(Image.fromarray(a).resize((tile_w, tile_h), Image.LANCZOS), (x, y + cap))
    im.save(path)


def render_framework_one(idx, name, info, outroot):
    z = np.load(fw_npz(name))
    rgb = load_rgb(str(z['rgb_path']))
    H, W = z['label'].shape
    # accuracy check: must match the evaluation pipeline's prediction for this image
    ref = load_label(pred_path('stdc', FRAMEWORK_KEY, name))
    agree = float((ref == z['pred']).mean())
    if agree < 0.9999:
        print('  [warn] %s: framework prediction agrees with eval cache on only %.4f%% of pixels' % (name, 100 * agree))
    full_maps = fw_full_maps(z)
    origins = fw_zoom_origins(z)
    folder = mkdir(osp.join(outroot, 'cand%02d__%s' % (idx, name)))
    views = [('full', FwView(z, rgb))] + [('zoom%d' % (i + 1), FwView(z, rgb, o)) for i, o in enumerate(origins)]
    thumbs = {}
    for vname, v in views:
        main, extra = fw_panels(z, v, full_maps)
        d = mkdir(osp.join(folder, vname))
        dx = mkdir(osp.join(d, 'extras'))
        for k, a in main.items():
            save_hq(a, osp.join(d, k + '.png'))
        for k, a in extra.items():
            save_hq(a, osp.join(dx, k + '.png'))
        allp = OrderedDict(list(main.items()) + list(extra.items()))
        sheet(allp, osp.join(d, 'SHEET_%s.png' % vname), 512 if vname == 'full' else 300)
        thumbs[vname] = allp
    # where the zooms are
    where = Image.fromarray(rgb.copy())
    dr = ImageDraw.Draw(where)
    for i, (y, x) in enumerate(origins):
        arr = dashed_box(np.array(where), (x, y, x + ZOOM_LABEL, y + ZOOM_LABEL), (255, 255, 0))
        where = Image.fromarray(arr)
        dr = ImageDraw.Draw(where)
        dr.text((x + 12, y + 8), 'zoom%d' % (i + 1), fill=(255, 255, 0), font=font(48))
    where.save(osp.join(folder, 'where_the_zooms_are.png'), dpi=DPI)
    band = z['band'] > 0
    sw, su = z['sel_bpm'] > 0, z['sel_stock'] > 0
    meta = OrderedDict([
        ('candidate', idx), ('image', name), ('folder', osp.relpath(folder, QROOT)), ('model', str(z['key'])),
        ('gamma_res_scale', float(z['gamma'])), ('model_input_hw', [int(v) for v in z['input_hw']]),
        ('label_hw', [H, W]), ('Z_hw', list(z['Z'].shape)), ('Zprime_hw', list(z['Z_prime'].shape)),
        ('bnd_radius', int(z['radius'])), ('w_bnd', float(z['w_bnd'])), ('ohem_n_min', int(z['n_min'])),
        ('pct_valid_pixels_in_band', 100.0 * float(band.sum()) / max(int((z['valid'] > 0).sum()), 1)),
        ('pct_bpm_selected_in_band', 100.0 * float((sw & band).sum()) / max(int(sw.sum()), 1)),
        ('pct_stock_selected_in_band', 100.0 * float((su & band).sum()) / max(int(su.sum()), 1)),
        ('rsr_changed_stride4_cells', int(z['rsr_changed'].sum())),
        ('thin_band_acc_gain_vs_baseline_pts', 100.0 * info[name]['thin_gain']),
        ('band_acc_gain_vs_baseline_pts', 100.0 * info[name]['band_gain']),
        ('prediction_agreement_with_eval_cache', agree),
        ('zoom_origins_yx_gt_px', [list(o) for o in origins]),
    ])
    with open(osp.join(folder, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    return meta, thumbs, np.array(where)


def framework_index(results, path):
    """one row per candidate: where the zooms are + key zoom1 panels + numbers"""
    keys = ['F01_input_image', 'F03_Z_stride8_logits', 'F05_Zprime_stride4_refined', 'F06_prediction',
            'F07_ground_truth', 'X4_pixels_changed_by_RSR', 'F10_selected_pixels_BPM']
    t, ww, pad, cap, txt = 230, 460, 6, 24, 330
    width = ww + len(keys) * (t + pad) + txt
    im = Image.new('RGB', (width, cap + len(results) * (t + pad)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fb, fs = font(16), font(14)
    d.text((4, 4), 'image (zoom boxes)', fill=(0, 0, 0), font=fb)
    for j, k in enumerate(keys):
        d.text((ww + pad + j * (t + pad), 4), 'zoom1: ' + FW_LABELS[k], fill=(0, 0, 0), font=fb)
    for i, (m, thumbs, where) in enumerate(results):
        y = cap + i * (t + pad)
        im.paste(fit(where, ww, t, (255, 255, 255)), (0, y))
        z1 = thumbs.get('zoom1', thumbs['full'])
        for j, k in enumerate(keys):
            im.paste(Image.fromarray(z1[k]).resize((t, t), Image.LANCZOS), (ww + pad + j * (t + pad), y))
        d.text((ww + pad + len(keys) * (t + pad) + 6, y + 6),
               'cand%02d  %s\nthin-band acc %+.1f pts vs baseline\nRSR changed %d stride-4 cells\n'
               'OHEM picks in boundary band:\n  BPM %.0f%%  vs  stock %.0f%%\n(band = %.1f%% of pixels)' % (
                   m['candidate'], m['image'], m['thin_band_acc_gain_vs_baseline_pts'], m['rsr_changed_stride4_cells'],
                   m['pct_bpm_selected_in_band'], m['pct_stock_selected_in_band'], m['pct_valid_pixels_in_band']),
               fill=(0, 0, 0), font=fs)
    im.save(path)


THIN_CS = common.THIN
THIN_OF = {ds: common.DATASETS[ds]['thin'] for ds in RUNS}


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------
def write_guide(rows, fw, strategy, mask_ignore):
    L = ['# Qualitative figures -- what every file is and how to assemble them', '',
         'Generated %s by `gen_qualitative.py`. All paths are relative to `resultData/qualitative/`.' % common.now(),
         'Models: baseline and "Ours" (= HI: RSR + BPM) only. STDC-Seg: I0 / HI1 (Cityscapes), Synthia / '
         'Synthia-HI1, RUGD / RUGD-HI1. HRNet-W48: baseline / HI1 (Cityscapes only -- the only dataset HRNet was '
         'trained on).', '',
         '## Folder map', '',
         '```',
         'fig_qualitative/                  Figure A (cf. ref_images/qualitative/QualitativeData.png)',
         '  PREVIEW_qualitative.png         auto-assembled preview in the recommended two-block layout (reference only)',
         '  legend_cityscapes_synthia.png   colour legend for Cityscapes + SYNTHIA rows (same classes)',
         '  legend_rugd.png                 colour legend for RUGD rows',
         '  rowN_<dataset>__<image>/        one folder per figure row',
         '    1_rgb.png 2_gt.png 3_stdc.png 4_stdc_ours.png [5_hrnet.png 6_hrnet_ours.png]   (nothing else;',
         '    --extras adds boxed/, zoom/ and green-fixed/red-broken diagnostics if you ever want them)',
         '  candidates/candidates_<dataset>.png   top-10 alternative images per dataset, with scores',
         'fig_framework/                    Figure B (cf. ref_images/qualitative/duet_framework.png)',
         '  INDEX_candidates.png            one row per candidate image -- choose here',
         '  candNN__<image>/                everything for one candidate (full/, zoom1/, zoom2/; see Figure B)',
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
          '2. HRNet exists for Cityscapes only, so SYNTHIA/RUGD rows have no HRNet panels. Recommended layout '
          '(the preview): **two blocks**. Top block = the Cityscapes rows x 6 columns (RGB, GT, STDC-Seg, '
          'STDC-Seg + Ours, HRNet-W48, HRNet-W48 + Ours). Dashed separator (like the reference figure). Bottom block '
          '= the SYNTHIA and RUGD rows x 4 columns (RGB, GT, STDC-Seg, STDC-Seg + Ours) with their own column titles; '
          'the 4 panels can be widened to span the full figure width. No empty cells, and the figure reads as '
          '"two hosts on Cityscapes" + "one host across domains".',
          '   - Alternative A: split into two figures (Cityscapes both hosts; cross-dataset STDC-Seg only).',
          '   - Alternative B: one 6-column grid with the HRNet cells on SYNTHIA/RUGD rows marked "--" and a caption '
          'note. Simplest, but empty cells look like missing results.',
          '3. Insert each row\'s images left to right in file order 1..6 (1..4 for SYNTHIA/RUGD).'
          '4. Aspect ratios differ: Cityscapes 2:1, SYNTHIA 1280x760 (1.68:1), RUGD 688x550 (1.25:1). Either crop '
          'every image in a row identically (Picture Format -> Crop, then copy the crop to the other panels of the '
          'row) or let rows have different heights. Always crop all panels of a row the same way.',
          '5. Put a row label on the left (Cityscapes / SYNTHIA / RUGD), column titles on top (table above), and the '
          'legend strip(s) at the bottom: `legend_cityscapes_synthia.png` and `legend_rugd.png`.',
          '6. Other image choices: look at `candidates/candidates_<dataset>.png`, then re-run with '
          '`--pick <dataset>:<image>`.', '',
          'Caption suggestion: "Qualitative results on Cityscapes (top: STDC-Seg and HRNet-W48, each without and '
          'with Ours) and on SYNTHIA and RUGD (bottom: STDC-Seg; HRNet-W48 was trained on Cityscapes only)." If the '
          'rows are best-case picks (strategy `top`), say they were "selected for visible differences"; '
          '`--strategy median` gives typical images instead.', '']

    # figure B
    if fw:
        m0 = fw[0]
        L += ['## Figure B -- framework diagram (STDC-Seg + Ours = HI1, Cityscapes val)', '',
              '%d candidate images, each rendered completely. Pick the one you like and fill the diagram from that '
              'folder. Start with `fig_framework/INDEX_candidates.png` (one row per candidate), then open '
              '`candNN__<image>/zoom1/SHEET_zoom1.png` for every panel with its diagram label.' % len(fw), '']
        L += common.md_table(
            ['Candidate folder', 'Image', 'Thin-band acc. vs baseline', 'RSR changed (stride-4 cells)',
             'OHEM picks in boundary band: BPM vs stock', 'Band share of pixels'],
            [['`%s/`' % m['folder'], m['image'], '%+.1f pts' % m['thin_band_acc_gain_vs_baseline_pts'],
              str(m['rsr_changed_stride4_cells']),
              '%.0f%% vs %.0f%%' % (m['pct_bpm_selected_in_band'], m['pct_stock_selected_in_band']),
              '%.1f%%' % m['pct_valid_pixels_in_band']] for m in fw]) + ['']
        L += ['Candidates = Cityscapes val images where Ours most improves thin structures near boundaries '
              '(per-image thin-band accuracy, HI1 vs the I0 baseline), at most %d per city.' % (-(-len(fw) // 3)), '',
              '### Inside each candidate folder', '',
              '```',
              'candNN__<image>/',
              '  where_the_zooms_are.png   the image with the zoom1 / zoom2 squares drawn in',
              '  full/     F01..F10 + extras/X1..X8 at full GT resolution (2048x1024) + SHEET_full.png',
              '  zoom1/    the same panels cropped to square window 1, 1536x1536   <- use these for the diagram boxes',
              '  zoom2/    a second, non-overlapping window (alternative crop)',
              '  meta.json numbers for the caption',
              '```', '',
              '**Accuracy and resolution.** Every panel is a real tensor of the trained HI1 model on that image (the '
              'model sees %dx%d, i.e. scale 0.75, exactly as in evaluation). *Z* is a %dx%d grid (stride 8) and *Z\'* '
              'a %dx%d grid (stride 4). In the zooms, each window is 512 GT px = 384 input px = 48 stride-8 cells = '
              '96 stride-4 cells, placed on a lattice aligned to the grid cells. All enlargements are exact integers '
              '(GT x3, stride-4 x16, stride-8 x32), so every block you see is one real grid cell, with no resampling '
              'artifacts. Label maps and grids use nearest-neighbour enlargement (crisp edges); RGB and heatmaps use '
              'Lanczos. Colour scales (loss, weight, Delta, PCA features) are computed on the whole image, so the full '
              'frame and both zooms share one scale. Each candidate\'s prediction was checked against the evaluation '
              'pipeline\'s cached prediction for the same image (agreement in `meta.json`).'
              % (m0['model_input_hw'][0], m0['model_input_hw'][1], m0['Z_hw'][0], m0['Z_hw'][1], m0['Zprime_hw'][0],
                 m0['Zprime_hw'][1]), '',
              '**Tip:** when you scale these PNGs down in PowerPoint, set the picture\'s resampling to keep edges '
              'sharp: File -> Options -> Advanced -> Image Size and Quality -> "Do not compress images in file", '
              'with default resolution "High fidelity".', '',
              '### Main slots (one per hatched box in duet_framework.png)', '']
        L += common.md_table(['Diagram box (label in the figure)', 'File (in full/ or zoom1/)', 'What it shows',
                              'How it was made'], [
            ['Input image *I*', 'F01_input_image.png', 'the val image', 'RGB as loaded'],
            ['context feat.', 'F02_context_feat.png', 'context-path features fed to the decoder (FFM)',
             'feat_cp8 (128 ch, stride 8; built from stride-16/32 context, so it looks coarse) -> first 3 PCA '
             'components as RGB'],
            ['*Z* (stride rho)', 'F03_Z_stride8_logits.png', 'host logits before RSR, as a label map',
             'argmax of conv_out output, stride 8 (rho = 8 for STDC-Seg)'],
            ['*F* (stride rho/2)', 'F04_F_stride4_features.png', 'high-res detail features RSR consumes',
             'feat_res4 (64 ch, stride 4) -> first 3 PCA components as RGB'],
            ["*Z'* (stride rho/2)", 'F05_Zprime_stride4_refined.png', 'logits after RSR, as a label map',
             "argmax of Z' = Z~' + gamma*Delta (verified equal to the module's output), stride 4"],
            ['Prediction *Y^*', 'F06_prediction.png', 'final full-resolution prediction',
             "Z' upsampled x4 to the input then to 1024x2048, argmax; black where the GT category is not one of "
             "the 19 evaluated classes (same black as F07; unmasked version = extras/X9)"],
            ['Ground truth *Y*', 'F07_ground_truth.png', 'GT labels', 'black = ignore'],
            ['*w* (Eq. 9) -- "weight map"', 'F08_weight_map_w.png', 'BPM boundary weights',
             'yellow = w_bnd (%.0f) within r=%d px of a GT boundary, blue = 1, black = ignore'
             % (m0['w_bnd'], m0['bnd_radius'])],
            ['*l_i* -- "loss map"', 'F09_loss_map.png', 'per-pixel cross-entropy', 'log(1+CE), magma colour map'],
            ['bottom-left box ("Output image *I*")', 'F10_selected_pixels_BPM.png',
             'pixels BPM\'s OHEM keeps (S_w)', 'red = kept and inside the boundary band, yellow = kept elsewhere, '
             'on a darkened RGB; OHEM rule of loss.py with n_min = 1/16 of the pixels, as in train.py']]) + ['']
        L += ['**The bottom-left box is mislabeled in the current diagram.** It reads "Output image *I*" with an '
              '"input image" placeholder, but BPM outputs a loss, not an image. Relabel it "OHEM-selected pixels '
              '*S_w*" and fill it with F10. That makes the BPM row end in a picture of what BPM does. Stronger '
              'option: put `extras/X8_selected_pixels_stock_OHEM.png` next to it as "stock OHEM" for contrast. '
              'Otherwise delete the box and let the arrow end at *L_BPM*.', '',
              'Caption numbers are per candidate in the table above and in `meta.json`, e.g. "the boundary band is '
              '%.1f%% of the pixels; stock OHEM spends %.0f%% of its selection there, BPM %.0f%%" (candidate 1).'
              % (m0['pct_valid_pixels_in_band'], m0['pct_stock_selected_in_band'], m0['pct_bpm_selected_in_band']),
              'Illustration caveat: the loss map and OHEM selection are computed on this val image with the trained '
              'model (per-image selection); during training they are computed on 512x1024 crops in batches of 16.', '',
              '### Optional extras (`extras/`, tensors the diagram names but has no box for)', '']
        L += common.md_table(['Where it would go', 'File', 'What it shows'], [
            ["*Z~'* (after \"bilinear x2\")", 'X1_Ztilde_bilinear_x2.png',
             'Z upsampled x2 before correction: compare with F05 to see what RSR adds'],
            ['*P* (after ConvBNReLU)', 'X2_P_projected_features.png', 'projected F inside RSR (PCA RGB)'],
            ['*Delta* (before x gamma)', 'X3_Delta_correction_magnitude.png',
             '|gamma*Delta| per stride-4 cell: where the correction acts (inferno)'],
            ["next to *Z'*, or as an inset", 'X4_pixels_changed_by_RSR.png',
             "cyan = stride-4 cells whose label differs between Z~' and Z' (what RSR changes)"],
            ['boundary set *B* (Eq. 8)', 'X5_boundary_set_B.png',
             'GT boundary pixels (full/: drawn 3 px wide so it survives downscaling; zooms: true width x3)'],
            ['dilate to *B_r*', 'X6_boundary_band_Br.png', 'band of radius r (white)'],
            ['after the (x) node, *w_i l_i*', 'X7_weighted_loss_w_times_l.png', 'weighted loss map (magma)'],
            ['contrast panel for F10', 'X8_selected_pixels_stock_OHEM.png', 'what stock OHEM (w = 1) would keep'],
            ['instead of F06, if you want it', 'X9_prediction_unmasked.png',
             'the raw prediction everywhere, incl. regions whose category is not evaluated']])
        L += ['', '**Black in F06/F07.** Black marks pixels whose Cityscapes category is not one of the 19 '
              'evaluated classes: parking, ground, static and dynamic objects, the ego vehicle, image borders. They '
              'are real, annotated image content, but excluded from training and from every metric (standard '
              'Cityscapes protocol), and the model cannot output those categories. The prediction is masked with '
              'exactly the GT\'s black region, so it hides nothing that is scored. Say so once in the legend or '
              'caption, e.g. "black: not among the 19 evaluated classes" (or an "n/a" legend entry).',
              '', '### How to fill the diagram in PowerPoint', '',
              '1. Choose a candidate (INDEX_candidates.png), then a crop (zoom1 or zoom2; see where_the_zooms_are.png).',
              '2. Open `ref_images/qualitative/duet_framework.pptx`. For each hatched placeholder: right-click -> '
              'Change Picture, using the same-named file from that candidate\'s `zoom1/` (or `zoom2/`). Use the '
              'same candidate and the same crop for every box, so the panels line up as one example.',
              '3. Keep the label text above each box. The images replace only the hatching. For the input-image box '
              '(a 1:1 box too), use zoom1/F01; if you want the whole scene there instead, use full/F01 and crop it in '
              'PowerPoint.',
              '4. Optional: add `extras/X4_pixels_changed_by_RSR.png` as a small inset beside *Z\'*, and swap the '
              'bottom-left box for F10 + X8 as described above.',
              '5. Label maps (Z, Z\', prediction, GT) use the Cityscapes palette; the legend is '
              '`fig_qualitative/legend_cityscapes_synthia.png` if the figure needs one.', '']
    L += ['## Regenerating', '',
          '```',
          '/home/husky/anaconda3/envs/stdcseg/bin/python gen_qualitative.py                 # everything',
          '... --pick cityscapes:<image> --pick rugd:<image>                                   # choose rows',
          '... --framework_image <img> [<img> ...] --framework_n 8                         # framework candidates',
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
    ap.add_argument('--framework_image', nargs='+', default=[],
                    help='Cityscapes val image(s) to include as framework candidates (always kept)')
    ap.add_argument('--framework_n', type=int, default=FW_N, help='number of framework candidates')
    ap.add_argument('--width', type=int, default=1024, help='max width of saved full-frame panels')
    ap.add_argument('--no_mask', action='store_true', help='do not black out GT-ignore pixels in predictions')
    ap.add_argument('--extras', action='store_true',
                    help='also write boxed/, zoom/ and extra/ diagnostics per row (off: RGB + predictions only)')
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
            r = render_row(len(rows) + 1, ds, d, args.width, not args.no_mask, figA, args.extras)
            r.update(dataset=ds, name=d['name'], rank=d['rank'], n=d['n'], score=d['score'], gains=d['gains'])
            rows.append(r)
            print('  row %d  %-10s %-32s rank %d/%d' % (len(rows), ds, d['name'], d['rank'], d['n']))
    preview(rows, osp.join(figA, 'PREVIEW_qualitative.png'))
    legend(LEGEND_NAMES_CS + ['n/a'], CITYSCAPES_PAL + [(0, 0, 0)],
           osp.join(figA, 'legend_cityscapes_synthia.png'), per_row=10)
    rugd_cls = sorted(set(c for r in rows if r['dataset'] == 'rugd' for c in r['present']))
    if rugd_cls:
        legend([common.RUGD_CLASSES[c] for c in rugd_cls] + ['n/a'], [RUGD_PAL[c] for c in rugd_cls] + [(0, 0, 0)],
               osp.join(figA, 'legend_rugd.png'), cell_w=130, per_row=8)

    fw = None
    if 'cityscapes' in args.datasets:
        names, info = framework_candidates(args.framework_n, args.framework_image)
        if not args.render_only:
            ensure_framework(names, args.force)
        names = [n for n in names if osp.isfile(fw_npz(n))]
        if names:
            figB = osp.join(QROOT, 'fig_framework')
            if osp.isdir(figB):
                shutil.rmtree(figB)
            mkdir(figB)
            results = []
            for i, n in enumerate(names, 1):
                results.append(render_framework_one(i, n, info, figB))
                print('  framework cand%02d %s' % (i, n))
            framework_index(results, osp.join(figB, 'INDEX_candidates.png'))
            fw = [m for m, _, _ in results]
    with open(osp.join(QROOT, 'manifest.json'), 'w') as fh:
        json.dump(dict(generated=common.now(), strategy=args.strategy, rows=rows, framework=fw), fh, indent=2)
    write_guide(rows, fw, args.strategy, not args.no_mask)
    print('done -> %s' % osp.relpath(QROOT, common.STDC_ROOT))


if __name__ == '__main__':
    main()
