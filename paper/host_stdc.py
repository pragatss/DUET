#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""STDC-Seg adapter: build/load, the evaluation.py eval loop, one train.py
training step. Run with cwd = repo root (CityScapes opens ./data and
./cityscapes_info.json relatively)."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper.common import DATASETS, PROTOCOL, STDC_RUNS, ckpt_path

BACKBONE = 'STDCNet1446'
N_CLASSES = 19
TRAIN_BATCH = 16            # train.py --n_img_per_gpu default; never overridden in commands.txt
TRAIN_CROP = [1024, 512]    # train.py dscfg['cityscapes']['cropsize'] (W, H)
RANDOMSCALE = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)


def load_eval_net(key):
    """Loaded, eval-mode net. Goes through evaluation.load_net so the
    variant-stamp and key-mismatch checks are the same as every other eval."""
    from evaluation import load_net
    run = STDC_RUNS[key]
    n_classes = len(DATASETS[run.get('dataset', 'cityscapes')]['classes'])
    net = load_net(ckpt_path('stdc', key), BACKBONE, True, run['use_brh'], 64,
                   n_classes, run['brh_variant'])
    return net.cuda().eval()


def build_net(use_brh):
    from models.model_stages import BiSeNet
    return BiSeNet(backbone=BACKBONE, n_classes=N_CLASSES, pretrain_model='',
                   use_boundary_2=False, use_boundary_4=False,
                   use_boundary_8=True, use_boundary_16=False,
                   use_conv_last=False, use_brh=use_brh, brh_mid=64)


def val_dataset(dataset='cityscapes'):
    """val split; .imnames gives the image order, .imgs[name] the RGB path"""
    root = DATASETS[dataset]['root']
    if dataset == 'synthia':
        from synthia import Synthia
        return Synthia(root, mode='val')
    if dataset == 'rugd':
        from rugd import RUGD
        return RUGD(root, mode='val')
    from cityscapes import CityScapes
    return CityScapes(root, mode='val')


def val_loader(dataset='cityscapes', n_workers=2, ds=None):
    return DataLoader(ds if ds is not None else val_dataset(dataset),
                      batch_size=PROTOCOL['stdc']['batchsize'], shuffle=False,
                      num_workers=n_workers, drop_last=False)


def predict(net, imgs, out_size, scale=PROTOCOL['stdc']['scale'], input_hw=None):
    """Exactly evaluation.evaluate_boundary's inference path (input_hw
    overrides the scale with an exact size, for the 32-divisible check)."""
    N, C, H, W = imgs.size()
    hw = list(input_hw) if input_hw else [int(H * scale), int(W * scale)]
    im = F.interpolate(imgs, hw, mode='bilinear', align_corners=True)
    logits = net(im)[0]
    logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=True)
    return torch.argmax(logits, dim=1)


def iterate_val(net, dataset='cityscapes', input_hw=None):
    """yield (pred, label) at GT resolution, [N,H,W] long cuda."""
    for imgs, label in val_loader(dataset):
        label = label.squeeze(1).cuda()
        yield predict(net, imgs.cuda(), label.shape[-2:], input_hw=input_hw), label


# ---------------------------------------------------------------------------
# runtime helpers
# ---------------------------------------------------------------------------
def infer_input_shape():
    s = PROTOCOL['stdc']['scale']
    return (1, 3, int(1024 * s), int(2048 * s))


def infer_fn(net):
    """full-res image -> label map, batch 1: the deployed path."""
    def f(x_full):
        return predict(net, x_full, x_full.shape[-2:])
    return f


def train_batches(n_batches, n_workers=8):
    """A few real augmented training batches (train.py's transforms), kept on
    the GPU so timing excludes data loading."""
    from cityscapes import CityScapes
    ds = CityScapes('./data', cropsize=TRAIN_CROP, mode='train', randomscale=RANDOMSCALE)
    dl = DataLoader(ds, batch_size=TRAIN_BATCH, shuffle=True, num_workers=n_workers,
                    drop_last=True)
    out = []
    for im, lb in dl:
        out.append((im.cuda(), torch.squeeze(lb, 1).cuda()))
        if len(out) == n_batches:
            break
    return out


def build_train_step(use_brh, bnd_weight):
    """-> (step(im, lb), modules) replicating one iteration of train.py's
    loop (criteria_p/16/32, detail BCE+dice, SGD over net + detail-loss
    params). No DDP wrapper: single process, so it only adds bookkeeping."""
    from loss.loss import OhemCELoss, BoundaryOhemCELoss
    from loss.detail_loss import DetailAggregateLoss
    net = build_net(use_brh).cuda().train()
    n_min = TRAIN_BATCH * TRAIN_CROP[0] * TRAIN_CROP[1] // 16
    criteria_p = BoundaryOhemCELoss(thresh=0.7, n_min=n_min, ignore_lb=255,
                                    radius=3, w_bnd=bnd_weight, ohem_mode='weighted_rank')
    criteria_16 = OhemCELoss(thresh=0.7, n_min=n_min, ignore_lb=255)
    criteria_32 = OhemCELoss(thresh=0.7, n_min=n_min, ignore_lb=255)
    detail_loss = DetailAggregateLoss()
    params = list(net.parameters()) + list(detail_loss.parameters())
    optim = torch.optim.SGD(params, lr=1e-2, momentum=0.9, weight_decay=5e-4)

    def step(im, lb):
        optim.zero_grad()
        out, out16, out32, detail8 = net(im)
        loss = criteria_p(out, lb) + criteria_16(out16, lb) + criteria_32(out32, lb)
        bce, dice = detail_loss(detail8, lb)
        loss = loss + bce + dice
        loss.backward()
        optim.step()
    return step, (net, detail_loss, optim)
