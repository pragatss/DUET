#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Score any STDC-Seg checkpoint on the val split of Cityscapes, SYNTHIA or RUGD
with the paper's metrics: full-image mIoU, boundary-band mIoU at r=1 / r=3,
and thin-class IoU inside the band. Same model build, inference path and
metric code as gen_perclass.py (paper/host_stdc.py + paper/common.SegMeter),
so the numbers match the paper tables; this script just takes a path instead
of a registry key.

USAGE (run from the repo root)
    python eval_checkpoint.py --dataset cityscapes --ckpt checkpoints/train_STDC2-Seg-HI1/pths/model_maxmIOU75.pth
    python eval_checkpoint.py --dataset synthia    --ckpt <pth>
    python eval_checkpoint.py --dataset rugd       --ckpt <pth> --per_class
    ... --ckpt base.pth ours.pth    # several checkpoints, deltas vs the first
    ... --size32                    # 32-divisible input instead of scale 0.75
    ... --out result.json           # full per-class breakdown

Whether a checkpoint has the RSR head (train.py --use_brh) and which ablation
variant it is are read from its weights, so baseline and Ours checkpoints can
be mixed in one call. BPM is loss-only and needs nothing. --use_brh / --no_brh
force it; a forced value that does not match the weights raises instead of
silently building the wrong model.

Checkpoints saved by torch >= 1.6 (zip format) need a torch >= 1.6 env.
"""
import argparse
import json

import torch
from tqdm import tqdm

from paper import common
from paper import host_stdc as H


def detect_brh(path):
    """-> (use_brh, brh_variant) from the state dict (variant stamp added by the ablation patch;
    older RSR checkpoints have none and are 'full')."""
    from models.model_stages import BRH_VARIANTS
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    keys = [k.replace('module.', '', 1) for k in sd]
    if not any(k.startswith('brh.') for k in keys):
        return False, 'full'
    for k, v in sd.items():
        if k.endswith('brh.variant_code'):
            return True, BRH_VARIANTS[int(v.flatten()[0])]
    return True, 'full'


def evaluate(path, dataset, use_brh, brh_variant, scale, size32):
    from evaluation import load_net
    ds = common.DATASETS[dataset]
    found = detect_brh(path)
    use_brh = found[0] if use_brh is None else use_brh
    brh_variant = brh_variant or found[1]
    print('  RSR head: %s%s' % ('yes, variant %s' % brh_variant if use_brh else 'no',
                               '' if (use_brh, brh_variant) == found else '  (forced)'))
    net = load_net(path, H.BACKBONE, True, use_brh, 64, len(ds['classes']), brh_variant)
    net.cuda().eval()
    meter = common.SegMeter(n_classes=len(ds['classes']), thin=ds['thin'],
                            class_names=ds['classes'], min_support=ds['min_support'])
    input_hw = ds['size32'] if size32 else None
    with torch.no_grad():
        for imgs, label in tqdm(H.val_loader(dataset)):
            label = label.squeeze(1).cuda()
            meter.update(H.predict(net, imgs.cuda(), label.shape[-2:], scale=scale, input_hw=input_hw), label)
    del net
    torch.cuda.empty_cache()
    return meter.result()


def fmt(x):
    return '%9s' % ('-' if x is None else '%.2f' % (100 * x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, choices=list(common.DATASETS))
    ap.add_argument('--ckpt', nargs='+', required=True, help='one or more .pth files')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--use_brh', dest='use_brh', action='store_const', const=True, default=None,
                   help='force the RSR head on (default: detect from the checkpoint)')
    g.add_argument('--no_brh', dest='use_brh', action='store_const', const=False,
                   help='force the RSR head off')
    ap.add_argument('--brh_variant', default=None, help='force the RSR ablation variant (default: detect)')
    ap.add_argument('--scale', type=float, default=common.PROTOCOL['stdc']['scale'])
    ap.add_argument('--size32', action='store_true', help="use the dataset's 32-divisible input size")
    ap.add_argument('--per_class', action='store_true', help='print per-class IoU (full and r=1 band)')
    ap.add_argument('--out', default=None, help='write all results (incl. per-class) to this JSON')
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = False

    ds = common.DATASETS[args.dataset]
    results = []
    for p in args.ckpt:
        print('\n=== %s  [%s] ===' % (p, args.dataset))
        results.append((p, evaluate(p, args.dataset, args.use_brh, args.brh_variant, args.scale, args.size32)))

    print('\nIoU in %%. Means over %s.' % (
        'classes with >= %.1f%% of val GT pixels' % (100 * ds['min_support']) if ds['min_support']
        else 'classes present in val'))
    print('Thin classes: %s' % ', '.join(ds['classes'][c] for c in ds['thin']))
    hdr = ('full', 'bnd_r1', 'bnd_r3', 'thin_r1', 'thin_r3')
    print('%-50s %s' % ('checkpoint', ' '.join('%9s' % h for h in hdr)))
    for p, r in results:
        print('%-50s %s' % (p[-50:], ' '.join(fmt(r[h]) for h in hdr)))
    for p, r in results[1:]:
        b = results[0][1]
        print('%-50s %s' % ('  d vs first', ' '.join(
            '%+9.2f' % (100 * (r[h] - b[h])) if r[h] is not None and b[h] is not None else '%9s' % '-'
            for h in hdr)))
    if ds['min_support']:
        print('(mIoU over every present class, unfiltered: %s)' % ', '.join(
            '%.2f' % (100 * r['full_all']) for _, r in results))

    if args.per_class:
        for p, r in results:
            print('\nper-class IoU  %s' % p)
            full, band = r['regions']['full'], r['regions']['bnd_r1']
            for c, name in enumerate(ds['classes']):
                if full['present'][c]:
                    print('  %-14s full %6.2f   bnd r1 %6.2f   support %6.3f%%%s' % (
                        name, 100 * full['iou'][c], 100 * band['iou'][c], 100 * r['support'][c],
                        '' if full['used'][c] else '   (excluded from means)'))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({p: r for p, r in results}, f, indent=2)
        print('\nwrote %s' % args.out)


if __name__ == '__main__':
    main()
