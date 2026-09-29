#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Worker for gen_qualitative.py (one host per process, in that host's env).

Predictions mode: runs a checkpoint over its dataset's val split and saves
  resultData/qualitative/cache/preds/<host>__<key>/<name>.png   argmax label map (uint8, GT resolution)
  resultData/qualitative/cache/stats/<host>__<key>.json         per-image counts for image selection
  resultData/qualitative/cache/gt/<dataset>/<name>.png          GT trainIds (255 = ignore), once
  resultData/qualitative/cache/gt/<dataset>/index.json          name -> RGB path

    <env python> -m paper.qual_worker --host stdc  --keys I0 HI1
    <env python> -m paper.qual_worker --host hrnet --keys baseline HI1

Framework mode (STDC only): one image through an RSR+BPM model, saving the
internal tensors the framework figure shows (context features, F, Z, Z~', P,
Delta, Z', prediction, boundary set, weight map, loss map, OHEM selection) to
resultData/qualitative/cache/framework_<key>_<name>.npz

    <env python> -m paper.qual_worker --host stdc --framework frankfurt_000001_054640 --keys HI1
"""
import argparse
import json
import os
import os.path as osp
import sys

_host = sys.argv[sys.argv.index('--host') + 1] if '--host' in sys.argv else None
if _host == 'hrnet':
    from paper import host_hrnet as H
else:
    from paper import host_stdc as H

import numpy as np                             # noqa: E402
import torch                                   # noqa: E402
import torch.nn.functional as F                # noqa: E402
from PIL import Image                          # noqa: E402

from paper import common                       # noqa: E402

QCACHE = osp.join(common.RESULT_DIR, 'qualitative', 'cache')


def stem(name):
    """dataset-independent image id: no extension, no Cityscapes GT suffix"""
    return osp.splitext(name)[0].replace('_gtFine_labelIds', '')


def mkdir(d):
    if not osp.isdir(d):
        os.makedirs(d)
    return d


def image_stats(pred, label, thin):
    """per-image counts: valid, boundary band r=1, thin classes inside the band."""
    valid = label != 255
    band = common.dilate(common.gt_boundary(label), 1) & valid
    thin_m = torch.zeros_like(valid)
    for c in thin:
        thin_m = thin_m | (label == c)
    thin_m = thin_m & band
    ok = pred == label
    out = []
    for i in range(label.shape[0]):
        out.append([int(valid[i].sum()), int((ok[i] & valid[i]).sum()),
                    int(band[i].sum()), int((ok[i] & band[i]).sum()),
                    int(thin_m[i].sum()), int((ok[i] & thin_m[i]).sum())])
    return out


def save_png(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
def run_predictions(host, key):
    dsname = common.run_dataset(host, key)
    thin = common.DATASETS[dsname]['thin']
    pdir = mkdir(osp.join(QCACHE, 'preds', '%s__%s' % (host, key)))
    gdir = mkdir(osp.join(QCACHE, 'gt', dsname))
    index = {}
    stats = {}
    print('\n=== %s / %s [%s] ===' % (host, key, dsname))
    with torch.no_grad():
        if host == 'hrnet':
            net, cfg = H.load_eval_net(key)
            dl = torch.utils.data.DataLoader(H._dataset(cfg, train=False), batch_size=1, shuffle=False,
                                             num_workers=4, pin_memory=True)
            for image, label, _, name in dl:
                label = label.long().cuda()
                pred = H.predict(net, cfg, image.cuda(), label.shape[-2:])
                n = stem(name[0])
                stats[n] = image_stats(pred, label, thin)[0]
                save_png(pred[0].cpu().numpy(), osp.join(pdir, n + '.png'))
        else:
            net = H.load_eval_net(key)
            ds = H.val_dataset(dsname)
            bs = common.PROTOCOL['stdc']['batchsize']
            for b, (imgs, label) in enumerate(H.val_loader(dsname, ds=ds)):
                label = label.squeeze(1).cuda()
                pred = H.predict(net, imgs.cuda(), label.shape[-2:])
                names = ds.imnames[b * bs: b * bs + label.shape[0]]
                st = image_stats(pred, label, thin)
                for i, nm in enumerate(names):
                    n = stem(nm)
                    stats[n] = st[i]
                    save_png(pred[i].cpu().numpy(), osp.join(pdir, n + '.png'))
                    gp = osp.join(gdir, n + '.png')
                    if not osp.isfile(gp):
                        save_png(label[i].cpu().numpy(), gp)
                    index[n] = osp.abspath(ds.imgs[nm])
    if index:
        with open(osp.join(gdir, 'index.json'), 'w') as f:
            json.dump(index, f)
    with open(osp.join(mkdir(osp.join(QCACHE, 'stats')), '%s__%s.json' % (host, key)), 'w') as f:
        json.dump(dict(fingerprint=common.fingerprint(host, key), dataset=dsname,
                       columns=['n_valid', 'ok_valid', 'n_band_r1', 'ok_band_r1', 'n_thin_band', 'ok_thin_band'],
                       images=stats), f)
    print('  %d images' % len(stats))
    del net
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
def pca_rgb(feat):
    """[1,C,h,w] -> [h,w,3] float in [0,1]: first 3 principal components,
    each stretched to its 1-99 percentile (standard feature visualisation)."""
    C, h, w = feat.shape[1:]
    X = feat[0].reshape(C, -1).t().float()
    X = X - X.mean(0, keepdim=True)
    U, S, V = torch.svd(X)
    Y = (X @ V[:, :3]).cpu().numpy()
    out = np.zeros_like(Y)
    for k in range(3):
        lo, hi = np.percentile(Y[:, k], 1), np.percentile(Y[:, k], 99)
        out[:, k] = np.clip((Y[:, k] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out.reshape(h, w, 3).astype(np.float16)


def ohem_select(loss, valid, n_min, thresh):
    """exactly BoundaryOhemCELoss's selection rule on one image's (weighted) loss map"""
    flat = loss[valid]
    srt, _ = torch.sort(flat, descending=True)
    k = min(n_min, srt.numel() - 1)
    if srt[k] > thresh:
        return (loss > thresh) & valid
    return (loss >= srt[k - 1]) & valid


def run_framework(key, name, radius=3, w_bnd=3.0):
    from loss.loss import boundary_weight_map
    dsname = common.run_dataset('stdc', key)
    ds = H.val_dataset(dsname)
    names = [stem(n) for n in ds.imnames]
    idx = names.index(name)
    img, label = ds[idx]
    img = img.unsqueeze(0).cuda()
    label = torch.from_numpy(label).cuda()          # [1,H,W]
    net = H.load_eval_net(key)
    assert net.use_brh, 'framework figure needs an RSR (use_brh) checkpoint'
    grab = {}
    h1 = net.cp.register_forward_hook(lambda m, i, o: grab.__setitem__('cp', o))
    h2 = net.brh.register_forward_hook(lambda m, i, o: grab.__setitem__('brh', (i, o)))
    with torch.no_grad():
        N, C, Hh, Ww = img.shape
        s = common.PROTOCOL['stdc']['scale']
        im = F.interpolate(img, [int(Hh * s), int(Ww * s)], mode='bilinear', align_corners=True)
        logits_in = net(im)[0]
        logits = F.interpolate(logits_in, size=label.shape[-2:], mode='bilinear', align_corners=True)
        h1.remove()
        h2.remove()
        feat_res4, feat_cp8 = grab['cp'][1], grab['cp'][4]
        (logits8, feat_hr), z_prime = grab['brh']
        brh = net.brh
        up = F.interpolate(logits8, feat_hr.size()[2:], mode='bilinear', align_corners=True)
        P = brh.proj(feat_hr)
        delta = brh.delta(brh.fuse(torch.cat([up, P], dim=1)))
        gamma = float(brh.res_scale)
        recon = up + gamma * delta
        assert torch.allclose(recon, z_prime, atol=1e-4), 'RSR reconstruction mismatch'

        valid = label[0] != 255
        ce = F.cross_entropy(logits, label, ignore_index=255, reduction='none')[0]
        w = boundary_weight_map(label, radius, w_bnd)[0]
        bnd = common.gt_boundary(label)[0] > 0.5
        band = common.dilate(common.gt_boundary(label), radius)[0] & valid
        thresh = -float(np.log(0.7))
        n_min = int(label.numel() // 16)       # train.py: n_img*crop_h*crop_w//16, per image
        sel_w = ohem_select(ce * w, valid, n_min, thresh)
        sel_u = ohem_select(ce, valid, n_min, thresh)

    a = lambda t: t.cpu().numpy()
    out = dict(
        name=name, key=key, rgb_path=osp.abspath(ds.imgs[ds.imnames[idx]]), gamma=gamma,
        radius=radius, w_bnd=w_bnd, n_min=n_min, input_hw=list(im.shape[-2:]),
        label=a(label[0]).astype(np.uint8), pred=a(logits.argmax(1)[0]).astype(np.uint8),
        Z=a(logits8.argmax(1)[0]).astype(np.uint8), Z_tilde=a(up.argmax(1)[0]).astype(np.uint8),
        Z_prime=a(z_prime.argmax(1)[0]).astype(np.uint8),
        ctx_pca=pca_rgb(feat_cp8), F_pca=pca_rgb(feat_res4), P_pca=pca_rgb(P),
        delta_mag=a((gamma * delta).norm(dim=1)[0]).astype(np.float32),
        rsr_changed=a(up.argmax(1)[0] != z_prime.argmax(1)[0]).astype(np.uint8),
        loss=a(ce).astype(np.float16), weight=a(w).astype(np.float16),
        boundary=a(bnd).astype(np.uint8), band=a(band).astype(np.uint8), valid=a(valid).astype(np.uint8),
        sel_bpm=a(sel_w).astype(np.uint8), sel_stock=a(sel_u).astype(np.uint8),
    )
    p = osp.join(mkdir(QCACHE), 'framework_%s_%s.npz' % (key, name))
    np.savez_compressed(p, **{k: (np.array(v) if not isinstance(v, np.ndarray) else v) for k, v in out.items()})
    print('  gamma(res_scale)=%+.3f  OHEM kept: BPM %d px (%.0f%% in boundary band), stock %d px (%.0f%%)' % (
        gamma, int(sel_w.sum()), 100.0 * float((sel_w & band).sum()) / max(int(sel_w.sum()), 1),
        int(sel_u.sum()), 100.0 * float((sel_u & band).sum()) / max(int(sel_u.sum()), 1)))
    print('  wrote %s' % p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True, choices=('stdc', 'hrnet'))
    ap.add_argument('--keys', nargs='+', required=True)
    ap.add_argument('--framework', default=None, help='image id for framework-mode (stdc only)')
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    if args.framework:
        assert args.host == 'stdc'
        run_framework(args.keys[0], args.framework)
        return
    for key in args.keys:
        run_predictions(args.host, key)


if __name__ == '__main__':
    main()
