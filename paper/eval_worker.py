#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Worker: score checkpoints of ONE host into resultData/cache/eval/.
Launched by paper.common.ensure_evaluated with the host's conda env:

    <env python> -m paper.eval_worker --host stdc  --keys I0 HI1
    <env python> -m paper.eval_worker --host hrnet --keys baseline HI1
"""
import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from paper import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True, choices=('stdc', 'hrnet'))
    ap.add_argument('--keys', nargs='+', required=True)
    args = ap.parse_args()

    if args.host == 'hrnet':
        from paper import host_hrnet as H     # must come before anything imports STDC's models
    else:
        from paper import host_stdc as H
    torch.backends.cudnn.benchmark = False
    if not os.path.isdir(common.EVAL_CACHE):
        os.makedirs(common.EVAL_CACHE)

    for key in args.keys:
        print('\n=== %s / %s ===\n%s' % (args.host, key, common.ckpt_path(args.host, key)))
        t0 = time.time()
        meter = common.SegMeter()
        with torch.no_grad():
            if args.host == 'hrnet':
                net, cfg = H.load_eval_net(key)
                it = H.iterate_val(net, cfg)
            else:
                net = H.load_eval_net(key)
                it = H.iterate_val(net)
            for pred, label in tqdm(it):
                meter.update(pred, label)
        res = meter.result()
        res['fingerprint'] = common.fingerprint(args.host, key)
        res['evaluated_at'] = common.now()
        res['eval_seconds'] = round(time.time() - t0, 1)
        res['gpu'] = torch.cuda.get_device_name(0)
        res['torch'] = torch.__version__

        print('  full %.4f | bnd r1 %.4f r3 %.4f | thin r1 %.4f r3 %.4f'
              % tuple(res[m] for m in common.METRICS))
        if args.host == 'hrnet':
            lb = common.HRNET_RUNS[key]['log_best']
            res['log_best_miou'] = lb
            if abs(res['full'] - lb) > 0.002:
                print('  [warn] full mIoU %.4f differs from training-log best %.4f' % (res['full'], lb))
        with open(common.cache_file(args.host, key), 'w') as f:
            json.dump(res, f, indent=2)
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
