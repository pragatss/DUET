#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Lay out SYNTHIA-RAND-CITYSCAPES in the train/val/test split used for the paper.

SYNTHIA-RAND-CITYSCAPES (9,400 frames) ships without a split. The paper uses a
fixed split at Cityscapes' proportions, 5593 train / 940 val / 2867 test
(test is never used), listed in splits/synthia/{train,val,test}.txt. This
script links (or copies) the raw files into the layout synthia.py reads:

    data/SYNTHIA/RGB/<split>/<id>.png
    data/SYNTHIA/GT/LABELS/<split>/<id>.png    (raw 16-bit labels, decoded by synthia.py)

USAGE
    python prepare_synthia.py --raw /path/to/RAND_CITYSCAPES          # symlinks
    python prepare_synthia.py --raw /path/to/RAND_CITYSCAPES --copy   # real copies

--raw is the extracted download, i.e. the folder that contains RGB/ and GT/LABELS/.
"""

import argparse
import os
import os.path as osp
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True, help='extracted SYNTHIA-RAND-CITYSCAPES folder (has RGB/ and GT/LABELS/)')
    ap.add_argument('--out', default='./data/SYNTHIA')
    ap.add_argument('--splits', default=osp.join(osp.dirname(osp.abspath(__file__)), 'splits', 'synthia'))
    ap.add_argument('--copy', action='store_true', help='copy files instead of symlinking')
    args = ap.parse_args()

    for split in ('train', 'val', 'test'):
        with open(osp.join(args.splits, split + '.txt')) as f:
            ids = [l.strip() for l in f if l.strip()]
        for sub in ('RGB', osp.join('GT', 'LABELS')):
            src_dir = osp.abspath(osp.join(args.raw, sub))
            dst_dir = osp.join(args.out, sub, split)
            if not osp.isdir(dst_dir):
                os.makedirs(dst_dir)
            missing = 0
            for i in ids:
                src, dst = osp.join(src_dir, i + '.png'), osp.join(dst_dir, i + '.png')
                if not osp.isfile(src):
                    missing += 1
                    continue
                if osp.lexists(dst):
                    continue
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(src, dst)
            print('%-5s %-10s %d files%s' % (split, sub, len(ids) - missing,
                                            ' (%d MISSING from %s)' % (missing, src_dir) if missing else ''))


if __name__ == '__main__':
    main()
