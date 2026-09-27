#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""HRNet adapter. Imports the HRNet checkout's lib/ in place (HRNET_ROOT in
paper/common.py), so nothing new has to live in that repo. Must be imported
before anything pulls in STDC's `models` package."""
import argparse
import os.path as osp
import sys

from paper.common import HRNET_ROOT, HRNET_RUNS, ckpt_path

sys.path.insert(0, osp.join(HRNET_ROOT, 'lib'))
if 'models' in sys.modules and HRNET_ROOT not in (getattr(sys.modules['models'], '__file__', '') or ''):
    raise ImportError("STDC's models package is already imported; run the hrnet host in its own process")

import torch                                   # noqa: E402
import torch.nn.functional as F                # noqa: E402
from torch.utils.data import DataLoader        # noqa: E402

from config import config as _base_cfg         # noqa: E402  (HRNet's lib/config)
from config import update_config               # noqa: E402
import datasets                                # noqa: E402  (HRNet's lib/datasets)
import models.seg_hrnet as seg_hrnet           # noqa: E402  (HRNet's lib/models)

N_CLASSES = 19


def load_cfg(key):
    cfg = _base_cfg.clone()
    update_config(cfg, argparse.Namespace(cfg=osp.join(HRNET_ROOT, HRNET_RUNS[key]['cfg']), opts=[]))
    cfg.defrost()
    cfg.DATASET.ROOT = osp.join(HRNET_ROOT, 'data') + '/'   # yaml has a cwd-relative 'data/'
    cfg.MODEL.PRETRAINED = ''                               # never needed: we load or time only
    cfg.freeze()
    return cfg


def build_net(key):
    """Fresh model for this run's config (tools/test.py's torch>=1 BN swap)."""
    cfg = load_cfg(key)
    seg_hrnet.BatchNorm2d_class = seg_hrnet.BatchNorm2d = torch.nn.BatchNorm2d
    return seg_hrnet.get_seg_model(cfg), cfg


def load_eval_net(key):
    net, cfg = build_net(key)
    sd = torch.load(ckpt_path('hrnet', key), map_location='cpu')
    sd = sd.get('state_dict', sd)
    # best.pth is FullModel.state_dict(): 'model.*' + 'loss.*'
    sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    net.load_state_dict(sd, strict=True)       # raises if USE_BRH doesn't match the checkpoint
    print('  checkpoint loaded cleanly (all keys matched)')
    for k, v in sd.items():
        if 'res_scale' in k:
            print('  %s = %+.4f' % (k, float(v.flatten()[0])))
    return net.cuda().eval(), cfg


def _dataset(cfg, train):
    if train:
        crop = (cfg.TRAIN.IMAGE_SIZE[1], cfg.TRAIN.IMAGE_SIZE[0])
        return datasets.cityscapes(root=cfg.DATASET.ROOT, list_path=cfg.DATASET.TRAIN_SET,
                                   num_samples=None, num_classes=N_CLASSES,
                                   multi_scale=cfg.TRAIN.MULTI_SCALE, flip=cfg.TRAIN.FLIP,
                                   ignore_label=cfg.TRAIN.IGNORE_LABEL,
                                   base_size=cfg.TRAIN.BASE_SIZE, crop_size=crop,
                                   downsample_rate=cfg.TRAIN.DOWNSAMPLERATE,
                                   scale_factor=cfg.TRAIN.SCALE_FACTOR)
    crop = (cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])
    return datasets.cityscapes(root=cfg.DATASET.ROOT, list_path=cfg.DATASET.TEST_SET,
                               num_samples=None, num_classes=N_CLASSES,
                               multi_scale=False, flip=False,
                               ignore_label=cfg.TRAIN.IGNORE_LABEL,
                               base_size=cfg.TEST.BASE_SIZE, crop_size=crop,
                               downsample_rate=1)


def predict(net, cfg, image, out_size):
    """HRNet's validate() path: full-res forward, bilinear to GT size."""
    logits = net(image)
    logits = F.interpolate(logits, size=out_size, mode='bilinear',
                           align_corners=cfg.MODEL.ALIGN_CORNERS)
    return torch.argmax(logits, dim=1)


def iterate_val(net, cfg, n_workers=4):
    dl = DataLoader(_dataset(cfg, train=False), batch_size=1, shuffle=False,
                    num_workers=n_workers, pin_memory=True)
    for image, label, _, _ in dl:
        label = label.long().cuda()
        yield predict(net, cfg, image.cuda(), label.shape[-2:]), label


# ---------------------------------------------------------------------------
# runtime helpers
# ---------------------------------------------------------------------------
def infer_input_shape(cfg):
    return (1, 3, cfg.TEST.IMAGE_SIZE[1], cfg.TEST.IMAGE_SIZE[0])


def infer_fn(net, cfg):
    def f(x_full):
        return predict(net, cfg, x_full, x_full.shape[-2:])
    return f


def train_batches(cfg, n_batches, n_workers=4):
    dl = DataLoader(_dataset(cfg, train=True), batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
                    shuffle=True, num_workers=n_workers, drop_last=True)
    out = []
    for images, labels, _, _ in dl:
        out.append((images.cuda(), labels.long().cuda()))
        if len(out) == n_batches:
            break
    return out


def build_train_step(key):
    """-> (step(im, lb), modules, cfg) replicating tools/train.py: FullModel
    (model + criterion chosen by the yaml), SGD. Single GPU, so the
    nn.DataParallel wrapper train.py adds would just call the module."""
    from core.criterion import OhemCrossEntropy, BoundaryOhemCrossEntropy
    from utils.utils import FullModel
    net, cfg = build_net(key)
    class_weights = _dataset(cfg, train=False).class_weights   # same constant tensor as train set
    if cfg.LOSS.USE_BOUNDARY_OHEM:
        crit = BoundaryOhemCrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL, thres=cfg.LOSS.OHEMTHRES,
                                        min_kept=cfg.LOSS.OHEMKEEP, weight=class_weights,
                                        bnd_radius=cfg.LOSS.BND_RADIUS, bnd_weight=cfg.LOSS.BND_WEIGHT)
    else:
        assert cfg.LOSS.USE_OHEM
        crit = OhemCrossEntropy(ignore_label=cfg.TRAIN.IGNORE_LABEL, thres=cfg.LOSS.OHEMTHRES,
                                min_kept=cfg.LOSS.OHEMKEEP, weight=class_weights)
    model = FullModel(net, crit).cuda().train()
    optim = torch.optim.SGD([{'params': list(model.parameters()), 'lr': cfg.TRAIN.LR}],
                            lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                            weight_decay=cfg.TRAIN.WD, nesterov=cfg.TRAIN.NESTEROV)

    def step(im, lb):
        losses, _ = model(im, lb)
        loss = losses.mean()
        model.zero_grad()
        loss.backward()
        optim.step()
    return step, (model, optim), cfg
