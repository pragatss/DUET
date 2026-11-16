#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Traversability maps for STDC-Seg and STDC-Seg + Ours, following TADA
(https://github.com/Masrur02/TADA, mmseg/models/utils/trav.py): each pixel of
the predicted seg map gets its class's traversability (1 = drivable, 0 = obstacle).

    /home/husky/anaconda3/envs/stdcseg/bin/python gen_trav.py --dataset rugd
    ... --dataset rugd --images path/to/frames      # own images instead of the val split

OUTPUT  resultData/traversability/<dataset>/<model>/{seg,trav,color}/<image>.png
  seg = trainIds, trav = traversability x 255 (gray), color = TADA's jet colours (red = traversable)
"""
import argparse
import os
import os.path as osp

import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import cm
from PIL import Image

from paper import common
from paper import host_stdc as H

# (baseline key, ours key) per dataset; Cityscapes baseline is a fresh-clone run, as in the digest (not I0)
MODELS = dict(cityscapes=('Baseline', 'HI1'), synthia=('Synthia', 'Synthia-HI1'), rugd=('RUGD', 'RUGD-HI1'))

# TADA CITYSCAPES_TRAV_PRIOR, Cityscapes trainId order (also used for SYNTHIA); other classes 0
CITY_TRAV = dict(road=1.00, sidewalk=0.75, terrain=0.50, veg=0.25)
# TADA FOREST_TRAV_PRIOR (RUGD), by class name; other classes 0
RUGD_TRAV = {'dirt': 0.65, 'sand': 0.45, 'grass': 0.80, 'container': 0.15, 'asphalt': 1.00,
             'gravel': 0.95, 'mulch': 0.40, 'rock-bed': 0.55, 'log': 0.10, 'bush': 0.10,
             'bridge': 0.70, 'concrete': 1.00}


def trav_lut(dataset):
    """trainId -> traversability; 255 (ignore) and unlisted classes -> 0"""
    prior = RUGD_TRAV if dataset == 'rugd' else CITY_TRAV
    lut = np.zeros(256, np.float32)
    for i, c in enumerate(common.DATASETS[dataset]['classes']):
        lut[i] = prior.get(c, 0.0)
    return lut


def save(arr, path):
    if not osp.isdir(osp.dirname(path)):
        os.makedirs(osp.dirname(path))
    Image.fromarray(np.ascontiguousarray(arr)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cityscapes', choices=list(MODELS))
    ap.add_argument('--images', default=None, help='folder of RGB images (default: the val split)')
    args = ap.parse_args()

    if args.images:
        files = sorted(osp.join(args.images, f) for f in os.listdir(args.images)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    else:
        ds = H.val_dataset(args.dataset)
        files = [ds.imgs[n] for n in ds.imnames]
    lut = trav_lut(args.dataset)
    to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    out_root = osp.join(common.RESULT_DIR, 'traversability', args.dataset)

    for model, key in zip(('stdc', 'stdc_ours'), MODELS[args.dataset]):
        print('%s (%s): %d images' % (model, key, len(files)))
        net = H.load_eval_net(key)
        for f in files:
            x = to_tensor(Image.open(f).convert('RGB')).unsqueeze(0).cuda()
            with torch.no_grad():
                seg = H.predict(net, x, x.shape[-2:])[0].cpu().numpy().astype(np.uint8)
            trav = lut[seg]
            name = osp.splitext(osp.basename(f))[0]
            save(seg, osp.join(out_root, model, 'seg', name + '.png'))
            save(np.round(trav * 255).astype(np.uint8), osp.join(out_root, model, 'trav', name + '.png'))
            # TADA mmseg/apis/test.py: jet(1 - trav) written as BGR, i.e. channel-reversed in RGB
            save((cm.jet(1.0 - trav)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1],
                 osp.join(out_root, model, 'color', name + '.png'))
        del net
        torch.cuda.empty_cache()
    print('wrote %s' % out_root)


if __name__ == '__main__':
    main()
