#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Print-quality image assets for the DUET framework figure, from ONE Cityscapes
image pushed through an RSR (Arm H, --use_brh) checkpoint.

  img_input   RGB input, original 2048x1024 pixels (Lanczos for the enlargement)
  img_Z       argmax of Z  = output of net.conv_out   (stride-8 logits, BEFORE RSR)
  img_Zprime  argmax of Z' = output of net.brh        (stride-4 logits, AFTER RSR)
  img_weight  BPM weight map w = loss.boundary_weight_map(label, r, lambda)
              1 (interior) -> white, lambda (band) -> dark blue, ignore -> gray
  img_loss    per-pixel CE of the final prediction vs GT, log10 over a FIXED
              range, magma; ignore -> gray (magma's low end is already black)
  img_pred    final prediction = argmax(net(im)[0]) at GT resolution (what is
              scored); masked black where the GT is ignore (--no_mask_pred to keep it)
  img_gt      GT trainIds, ignore -> black   (reference only)
  each as .png (lossless) and .pdf (same pixels, Flate-compressed, no resampling),
  plus ref_crop_overview.png (whole frame, crop outlined) and assets_meta.json.

Geometry. Everything is cut from the full-resolution frame. A stride-s tensor
of n cells over N full-res pixels is drawn as a partition: cell i covers
full-res [i*N/n, (i+1)*N/n). At scale 0.75 a Z cell is 32/3 full-res px and a Z'
cell 16/3 px, so the output is the full-res crop enlarged by an integer k that is
a multiple of 3 (smallest with long side >= --min_long_side, i.e. 3 for a 544 px
crop). Then every full-res pixel is a k x k block, every Z' cell (16k/3)^2 and every
Z cell (32k/3)^2, all exact. Crops on the 32-px lattice also start and end on
whole Z cells. Label and scalar maps are nearest-neighbour, no other resampling.

Consistency. Colours, the loss range (default 0.01..3, fixed, not per image), the
enlargement factor and the crop are all deterministic. Auto-crop depends on the
checkpoint (it looks for Z vs Z' changes), so pin it for later runs with --crop,
or with --crop_from <previous assets_meta.json>.

Inference is evaluation.evaluate_boundary's path: evaluation.load_net (same
key-mismatch and variant checks), eval(), no_grad, ImageNet normalisation as in
cityscapes.py, bilinear resize to scale 0.75, net(im)[0] bilinearly to GT size.
Z and Z' come from forward hooks. No model code is touched.

USAGE (repo root, stdcseg env)
    python tools/make_figure_assets.py \\
        --ckpt ./checkpoints/train_STDC2-Seg-HI1/pths/model_maxmIOU75.pth --use_brh \\
        --image data/leftImg8bit/val/munster/munster_000016_000019_leftImg8bit.png \\
        --label data/gtFine/val/munster/munster_000016_000019_gtFine_labelTrainIds.png \\
        --out resultData/figure_assets/ --auto_crop
    # same crop/settings with another checkpoint:
    python tools/make_figure_assets.py --ckpt <other.pth> --use_brh --image ... --label ... \\
        --out resultData/figure_assets_other/ --crop_from resultData/figure_assets/assets_meta.json

--label accepts *_labelTrainIds.png (used as is) or *_labelIds.png (converted
with cityscapes_info.json, as the val loader does).
"""
import argparse
import json
import os
import os.path as osp
import sys

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import torch.nn.functional as F                     # noqa: E402
import torchvision.transforms as transforms         # noqa: E402
from PIL import Image, ImageDraw                    # noqa: E402

from evaluation import load_net, gt_boundary, dilate, CLASS_NAMES, THIN   # noqa: E402
from loss.loss import boundary_weight_map           # noqa: E402
from gen_qualitative import CITYSCAPES_PAL          # noqa: E402

N_CLASSES = 19
IGNORE = 255
LATTICE = 32                         # = 3 Z cells at scale 0.75
CROP_THIN = [5, 6, 7]                # pole, traffic light, traffic sign (auto-crop targets)
WHITE, DARK_BLUE, GRAY, BLACK = (255, 255, 255), (8, 48, 107), (128, 128, 128), (0, 0, 0)
LOSS_CMAP = 'magma'
PDF_DPI = 600                        # PDF page = pixels / 600 in (1632 px -> 2.72 in); pixels are unchanged
ASSETS = ('img_input', 'img_Z', 'img_Zprime', 'img_weight', 'img_loss', 'img_pred', 'img_gt')


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def load_label(path):
    lb = np.array(Image.open(path)).astype(np.int64)
    if '_labelIds' in osp.basename(path):           # raw ids -> trainIds, as cityscapes.CityScapes does
        with open(osp.join(REPO, 'cityscapes_info.json')) as f:
            lb_map = {el['id']: el['trainId'] for el in json.load(f)}
        out = np.full_like(lb, IGNORE)
        for k, v in lb_map.items():
            out[lb == k] = v
        lb = out
    return lb


def to_tensor(rgb_img):
    # identical to cityscapes.CityScapes.to_tensor (cityscapes.py:68-71)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return t(rgb_img).unsqueeze(0)


def require_rsr(ckpt):
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)
    if not any(k.replace('module.', '', 1).startswith('brh.') for k in sd):
        sys.exit('ERROR: %s has no RSR (brh.*) weights. It was trained without --use_brh, so the '
                 'refined logits Z\' do not exist for this model. Use an H1/HI1-style checkpoint.' % ckpt)


# ---------------------------------------------------------------------------
# geometry: exact partition mapping, integer arithmetic only
# ---------------------------------------------------------------------------
def cell_index(out_len, start, k, n, N):
    """output pixels 0..out_len-1 of a crop starting at full-res `start`, enlarged by k ->
    index of the covering cell of an n-cell grid over N full-res px:
    floor((start + (u + 0.5)/k) * n / N), computed exactly in integers."""
    u = np.arange(out_len, dtype=np.int64)
    return np.clip(((2 * (k * start + u) + 1) * n) // (2 * k * N), 0, n - 1)


def render_grid(arr, region, k, full_hw):
    """arr [n_h, n_w] covering the whole frame -> crop at k x full-res, nearest"""
    x, y, w, h = region
    iy = cell_index(h * k, y, k, arr.shape[0], full_hw[0])
    ix = cell_index(w * k, x, k, arr.shape[1], full_hw[1])
    return arr[iy[:, None], ix[None, :]]


def to_full(arr, full_hw):
    """same partition mapping onto every full-res pixel (k = 1)"""
    return render_grid(arr, (0, 0, full_hw[1], full_hw[0]), 1, full_hw)


def block_size(k, n, N):
    s = k * N / float(n)
    return s, abs(s - round(s)) < 1e-9


def choose_k(region, min_long):
    """smallest multiple of 3 (whole-pixel Z / Z' cells at scale 0.75) reaching min_long"""
    k = 3
    while k * max(region[2], region[3]) < min_long:
        k += 3
    return k


def auto_crop(label, changed, size, max_ignore=0.15):
    """lattice-aligned window with the most Z!=Z' pixels near GT pole / light / sign"""
    cw, ch = size
    H, W = label.shape
    near = dilate(torch.from_numpy(np.isin(label, CROP_THIN).astype(np.float32)).unsqueeze(0), 3)[0]
    score = (changed & near.numpy().astype(bool)).astype(np.int64)
    ign = (label == IGNORE).astype(np.int64)

    def box_sums(m):
        ii = np.zeros((H + 1, W + 1), np.int64)
        ii[1:, 1:] = m.cumsum(0).cumsum(1)
        ys = np.arange(0, H - ch + 1, LATTICE)[:, None]
        xs = np.arange(0, W - cw + 1, LATTICE)[None, :]
        return ii[ys + ch, xs + cw] - ii[ys, xs + cw] - ii[ys + ch, xs] + ii[ys, xs]

    s, g = box_sums(score), box_sums(ign)
    s = np.where(g <= max_ignore * cw * ch, s, -1)
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    return (int(ix * LATTICE), int(iy * LATTICE), cw, ch), int(s[iy, ix])


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------
def colorize(lbl):
    lut = np.zeros((256, 3), np.uint8)          # 255 (ignore) -> black, Cityscapes 'void'
    lut[:len(CITYSCAPES_PAL)] = CITYSCAPES_PAL
    return lut[lbl]


def color_weight(w, valid, lam):
    t = np.clip((w - 1.0) / max(lam - 1.0, 1e-6), 0, 1)[..., None]
    rgb = (1 - t) * np.array(WHITE) + t * np.array(DARK_BLUE)
    rgb[~valid] = GRAY
    return np.round(rgb).astype(np.uint8)


def color_loss(ce, valid, lo, hi):
    import matplotlib.cm as cm
    t = (np.log10(np.clip(ce, lo, hi)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    rgb = np.round(cm.get_cmap(LOSS_CMAP)(t)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = GRAY
    return rgb


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def save_pdf(rgb, path):
    """exactly these pixels on a page of the same aspect: no axes, no padding, and
    interpolation='none' makes the PDF backend embed the raw array (FlateDecode)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / float(PDF_DPI), h / float(PDF_DPI)), dpi=PDF_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(rgb, interpolation='none', aspect='auto')
    fig.savefig(path, dpi=PDF_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# metrics: evaluation.evaluate_boundary's accumulation on one image
# ---------------------------------------------------------------------------
def image_metrics(pred, label):
    pred, label = torch.from_numpy(pred).unsqueeze(0), torch.from_numpy(label).unsqueeze(0)
    valid = label != IGNORE
    bnd = gt_boundary(label, IGNORE)
    out = {}
    for name, mask in (('full', valid), ('bnd_r1', dilate(bnd, 1) & valid), ('bnd_r3', dilate(bnd, 3) & valid)):
        I = np.array([float(((pred == c) & (label == c) & mask).sum()) for c in range(N_CLASSES)])
        U = np.array([float((((pred == c) | (label == c)) & mask).sum()) for c in range(N_CLASSES)])
        iou = I / (U + 1e-6)
        out[name] = float(iou[U > 0].mean())
        if name == 'bnd_r1':
            present = [c for c in THIN if U[c] > 0]
            out['thin_r1'] = float(iou[present].mean()) if present else float('nan')
            out['thin_present'] = [CLASS_NAMES[c] for c in present]
    return out


def registry_lookup(ckpt, stem):
    """(key, cached eval-path prediction or None, val-set metrics or None) for a registered checkpoint"""
    try:
        from paper import common
        for key in common.STDC_RUNS:
            if osp.abspath(common.ckpt_path('stdc', key)) != osp.abspath(ckpt):
                continue
            pred = val = None
            p = osp.join(common.RESULT_DIR, 'qualitative', 'cache', 'preds', 'stdc__' + key, stem + '.png')
            if osp.isfile(p):
                pred = np.array(Image.open(p)).astype(np.int64)
            e = osp.join(common.RESULT_DIR, 'cache', 'eval', 'stdc__%s.json' % key)
            if osp.isfile(e):
                with open(e) as f:
                    val = json.load(f)
            return key, pred, val
    except Exception:
        pass
    return None, None, None


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--use_brh', action='store_true', help='required: Z\' only exists with RSR')
    ap.add_argument('--brh_variant', default='full')
    ap.add_argument('--image', required=True, help='*_leftImg8bit.png, full resolution')
    ap.add_argument('--label', required=True, help='*_labelTrainIds.png or *_labelIds.png')
    ap.add_argument('--out', default=osp.join(REPO, 'resultData', 'figure_assets'))
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--crop', default=None, help='x,y,w,h in full-resolution pixels')
    g.add_argument('--crop_from', default=None, help='reuse the crop of a previous assets_meta.json')
    g.add_argument('--auto_crop', action='store_true', help='(default when no crop is given)')
    ap.add_argument('--crop_size', default='544,480', help='auto-crop window w,h (multiples of 32)')
    ap.add_argument('--min_long_side', type=int, default=1600)
    ap.add_argument('--scale', type=float, default=0.75)
    ap.add_argument('--bnd_radius', type=int, default=3)
    ap.add_argument('--bnd_weight', type=float, default=3.0)
    ap.add_argument('--loss_range', default='0.01,3', help='fixed CE range lo,hi for the log colour scale')
    ap.add_argument('--no_mask_pred', action='store_true', help='keep predictions inside GT-ignore regions')
    ap.add_argument('--mask_internal', action='store_true', help='also mask Z and Z\' (off: internal tensors)')
    ap.add_argument('--no_pdf', action='store_true')
    args = ap.parse_args()

    if not args.use_brh:
        sys.exit('ERROR: pass --use_brh. img_Zprime needs the RSR module (and the checkpoint must have it).')
    require_rsr(args.ckpt)
    torch.backends.cudnn.benchmark = False
    lam, radius = args.bnd_weight, args.bnd_radius
    lo, hi = (float(v) for v in args.loss_range.split(','))
    stem = osp.basename(args.image).replace('_leftImg8bit.png', '')

    print('\n[1] checkpoint (evaluation.load_net)')
    net = load_net(args.ckpt, 'STDCNet1446', True, True, 64, N_CLASSES, args.brh_variant)   # raises on any mismatch
    net = net.cuda().eval()
    print('    OK: no missing / unexpected keys')

    print('\n[2] RSR residual scale')
    rs = net.brh.res_scale
    gamma = float(rs) if rs is not None else None
    if gamma is None:
        print('    variant %r has no res_scale (fixed gate)' % args.brh_variant)
    elif abs(gamma) < 1e-3:
        print('    [WARN] brh.res_scale = %+.2e ~ 0: RSR never trained, Z\' == upsampled Z' % gamma)
    else:
        print('    OK: brh.res_scale = %+.4f' % gamma)

    rgb_img = Image.open(args.image).convert('RGB')
    label = load_label(args.label)
    H, W = label.shape
    assert rgb_img.size == (W, H), 'image %s and label %s sizes differ' % (rgb_img.size, (W, H))
    img = to_tensor(rgb_img).cuda()

    grab = {}
    hooks = [net.conv_out.register_forward_hook(lambda m, i, o: grab.__setitem__('Z', o)),
             net.brh.register_forward_hook(lambda m, i, o: grab.__setitem__('brh', (i[0], o)))]
    try:
        with torch.no_grad():
            im = F.interpolate(img, [int(H * args.scale), int(W * args.scale)], mode='bilinear', align_corners=True)
            out = net(im)[0]
            logits = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
            lt = torch.from_numpy(label).unsqueeze(0).cuda()
            ce = F.cross_entropy(logits, lt, ignore_index=IGNORE, reduction='none')[0]
            w = boundary_weight_map(lt, radius=radius, w_bnd=lam, ignore_lb=IGNORE)[0]
            Z, (brh_in, Zp) = grab['Z'], grab['brh']
            z_is_brh_input = bool(torch.equal(Z, brh_in))
            zp_feeds_output = bool(torch.allclose(
                F.interpolate(Zp, im.shape[-2:], mode='bilinear', align_corners=True), out, atol=1e-5))
    finally:
        for h in hooks:
            h.remove()

    a = lambda t: t.cpu().numpy()
    Z_arg, Zp_arg = a(Z.argmax(1)[0]), a(Zp.argmax(1)[0])
    pred, ce, w = a(logits.argmax(1)[0]), a(ce), a(w)
    valid = label != IGNORE
    Z_full, Zp_full = to_full(Z_arg, (H, W)), to_full(Zp_arg, (H, W))

    print('\n[3] Z\' is the tensor behind the prediction')
    print('    input %dx%d -> Z %dx%d (stride 8), Z\' %dx%d (stride 4)' % (
        im.shape[-1], im.shape[-2], Z.shape[-1], Z.shape[-2], Zp.shape[-1], Zp.shape[-2]))
    print('    net.conv_out output == net.brh input               : %s' % z_is_brh_input)
    print('    bilinear(Z\') reproduces net(im)[0] (atol 1e-5)     : %s' % zp_feeds_output)
    key, ck_pred, val = registry_lookup(args.ckpt, stem)
    if ck_pred is not None:
        print('    prediction == eval-cache prediction (stdc__%s)    : %.4f of pixels' % (key, float((ck_pred == pred).mean())))
    agree = float((Zp_full == pred)[valid].mean())
    print('    nearest-upsampled argmax(Z\') == prediction         : %.4f of valid pixels' % agree)
    print('      (argmax and bilinear upsampling do not commute: the rest is a thin strip along predicted edges)')
    print('    argmax(Z) != argmax(Z\') (what RSR changed)          : %.2f%% of valid pixels' % (
        100 * float((Z_full != Zp_full)[valid].mean())))
    assert z_is_brh_input and zp_feeds_output, 'hooks did not capture the tensors on the prediction path'
    assert agree >= 0.98, 'argmax(Z\') agrees with the prediction on only %.2f%% of valid pixels' % (100 * agree)

    print('\n[4] BPM weight map (r=%d, lambda=%g)' % (radius, lam))
    vals = sorted(set(np.unique(w[valid]).tolist()))
    eff_ignore = float(np.abs(ce[~valid] * w[~valid]).max()) if (~valid).any() else 0.0
    print('    values on valid pixels: %s, lambda on %.2f%% of them' % (vals, 100.0 * float((w[valid] == lam).mean())))
    print('    CE*w on ignore pixels: max %g (CE uses ignore_index; drawn gray)' % eff_ignore)
    assert set(vals) <= {1.0, lam}, 'unexpected weight values %s' % vals
    assert eff_ignore == 0.0, 'ignore pixels carry loss'

    print('\n[5] is this image representative? (evaluation.gt_boundary / dilate, scale %.2f)' % args.scale)
    met = image_metrics(pred, label)
    print('    %-12s %8s %8s %8s %8s' % ('', 'mIoU', 'bnd r=1', 'bnd r=3', 'thin r=1'))
    print('    %-12s %8.4f %8.4f %8.4f %8.4f   (thin over present: %s)' % (
        'this image', met['full'], met['bnd_r1'], met['bnd_r3'], met['thin_r1'], ', '.join(met['thin_present'])))
    if val is not None:
        print('    %-12s %8.4f %8.4f %8.4f %8.4f' % ('val (%s)' % key, val['full'], val['bnd_r1'], val['bnd_r3'], val['thin_r1']))
    print('    (one image: classes absent from it are skipped, so expect spread around the val mean)')

    # ---- crop -------------------------------------------------------------
    if args.crop_from:
        with open(args.crop_from) as f:
            prev = json.load(f)
        region = tuple(prev['crop_xywh'])
        lo, hi = prev['loss_range']
        print('\n[crop] from %s: --crop %s (loss range %g..%g)' % (args.crop_from, ','.join(map(str, region)), lo, hi))
    elif args.crop:
        region = tuple(int(v) for v in args.crop.split(','))
        assert len(region) == 4, '--crop wants x,y,w,h'
        print('\n[crop] --crop %s' % args.crop)
    else:
        size = tuple(int(v) for v in args.crop_size.split(','))
        region, sc = auto_crop(label, Z_full != Zp_full, size)
        print('\n[crop] auto: %d px where Z != Z\' within 3 px of a GT pole/light/sign -> pin with  --crop %s'
              % (sc, ','.join(map(str, region))))
    x, y, cw, ch = region
    assert x >= 0 and y >= 0 and x + cw <= W and y + ch <= H, 'crop %s outside %dx%d' % (region, W, H)
    if any(v % LATTICE for v in region):
        print('    note: crop is off the %d-px lattice, so Z cells at the crop border are partial' % LATTICE)
    k = choose_k(region, args.min_long_side)
    for nm, t in (('Z', Z_arg), ('Z\'', Zp_arg)):
        sx, okx = block_size(k, t.shape[1], W)
        sy, oky = block_size(k, t.shape[0], H)
        print('    %-3s cell = %gx%g output px%s' % (nm, sx, sy, '' if (okx and oky) else '  [WARN] not whole pixels'))
    print('    enlargement k = %d -> %dx%d px (full-res pixel = %dx%d block)' % (k, cw * k, ch * k, k, k))

    # ---- render -----------------------------------------------------------
    hw = (H, W)
    v_out = render_grid(valid, region, k, hw)
    imgs = {
        'img_input': np.array(rgb_img.crop((x, y, x + cw, y + ch)).resize((cw * k, ch * k), Image.LANCZOS)),
        'img_Z': colorize(np.where(v_out | (not args.mask_internal), render_grid(Z_arg, region, k, hw), IGNORE)),
        'img_Zprime': colorize(np.where(v_out | (not args.mask_internal), render_grid(Zp_arg, region, k, hw), IGNORE)),
        'img_weight': color_weight(render_grid(w, region, k, hw), v_out, lam),
        'img_loss': color_loss(render_grid(ce, region, k, hw), v_out, lo, hi),
        'img_pred': colorize(np.where(v_out | args.no_mask_pred, render_grid(pred, region, k, hw), IGNORE)),
        'img_gt': colorize(render_grid(label, region, k, hw)),
    }
    os.makedirs(args.out, exist_ok=True)
    for n in ASSETS:
        Image.fromarray(imgs[n]).save(osp.join(args.out, n + '.png'))
        if not args.no_pdf:
            save_pdf(imgs[n], osp.join(args.out, n + '.pdf'))

    ov = rgb_img.resize((W // 2, H // 2), Image.LANCZOS)
    ImageDraw.Draw(ov).rectangle([x // 2, y // 2, (x + cw) // 2 - 1, (y + ch) // 2 - 1], outline=WHITE, width=3)
    ov.save(osp.join(args.out, 'ref_crop_overview.png'))

    c = ce[y:y + ch, x:x + cw][valid[y:y + ch, x:x + cw]]
    lstat = dict(p1=float(np.percentile(c, 1)), median=float(np.median(c)), p99=float(np.percentile(c, 99)),
                 below_lo=float((c < lo).mean()), above_hi=float((c > hi).mean()))
    print('\n[out] %s' % args.out)
    print('    %s  (.png%s, %dx%d)  + ref_crop_overview.png, assets_meta.json' % (
        ', '.join(ASSETS), '' if args.no_pdf else ' + .pdf', cw * k, ch * k))
    print('    prediction %smasked; Z/Z\' %smasked' % ('NOT ' if args.no_mask_pred else '', '' if args.mask_internal else 'not '))
    print('    loss: log10, %s, fixed range [%g, %g]; crop CE p1 %.3g / median %.3g / p99 %.3g; '
          '%.1f%% below lo, %.1f%% above hi (saturate)' % (
              LOSS_CMAP, lo, hi, lstat['p1'], lstat['median'], lstat['p99'],
              100 * lstat['below_lo'], 100 * lstat['above_hi']))

    meta = dict(ckpt=osp.abspath(args.ckpt), registry_key=key, image=osp.abspath(args.image),
                label=osp.abspath(args.label), scale=args.scale, res_scale=gamma,
                bnd_radius=radius, bnd_weight=lam, crop_xywh=list(region), k=k,
                size_wh=[cw * k, ch * k], z_cell_px=block_size(k, Z_arg.shape[1], W)[0],
                zprime_cell_px=block_size(k, Zp_arg.shape[1], W)[0],
                loss_range=[lo, hi], loss_cmap=LOSS_CMAP, loss_in_crop=lstat,
                mask_pred=not args.no_mask_pred, mask_internal=args.mask_internal,
                colors=dict(weight_interior=WHITE, weight_band=DARK_BLUE, ignore_scalar=GRAY,
                            ignore_label=BLACK, palette='Cityscapes (gen_qualitative.CITYSCAPES_PAL)'),
                checks=dict(z_is_brh_input=z_is_brh_input, zp_feeds_output=zp_feeds_output,
                            zprime_pred_agreement_valid=agree, weight_values=vals),
                metrics=met, val_metrics=None if val is None else {m: val[m] for m in
                                                                   ('full', 'bnd_r1', 'bnd_r3', 'thin_r1')})
    with open(osp.join(args.out, 'assets_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)


if __name__ == '__main__':
    main()
