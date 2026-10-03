#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Worker: training/inference throughput, peak GPU memory, params and GMACs for
ONE host. Launched by gen_runtime_memory.py with the host's conda env:

    <env python> -m paper.runtime_worker --host stdc --out resultData/runtime/raw/stdc.json

Protocol:
  * Timing: inference nets are built once and kept resident; training
    configs are rebuilt per block (they don't fit together). The GPU is
    heat-soaked for --soak_s so it reaches its throttled steady state (the
    T4 sits at 83-84C and 825-1545 MHz under sustained load). Then R rounds
    run; each round visits every config in a fresh random order for a SHORT
    block (a few iterations), so all configs sample the same clock
    distribution. Reported: pooled throughput (total iters / total time), and
    the median of per-round ratios to the host baseline (paired, which
    cancels the clock state common to a round).
  * Each measurement: only that config is on the GPU. The allocator is
    emptied and the peak counters reset first, so peak memory is that
    config's weights + grads + optimizer state + activations, nothing else.
    Memory comes from a separate pass with cudnn.benchmark OFF (autotuning
    scratch workspaces are large and run-dependent); timing uses it ON.
  * Inference: batch 1, full-res 1024x2048 input -> label map (each host's
    eval path incl. its resize and the final upsample + argmax), blocks
    bracketed by cuda.synchronize.
  * Training: one real train.py / tools/train.py iteration (forward, every
    loss term, backward, SGD step) at the host's training batch and crop, on
    real augmented Cityscapes batches pre-loaded to host memory (copied to the
    GPU each step, as the training loops do). Data loading itself is
    excluded, so this is the model+loss cost, not the as-trained wall clock
    (gen_runtime_memory.py reports that separately from the training logs).
  * Weights do not affect speed or memory, so training uses a fresh init;
    inference loads the real checkpoint (which also proves the arch matches).
"""
import argparse
import json
import os
import os.path as osp
import random
import statistics
import subprocess
import sys
import time
from collections import OrderedDict

# host adapters must be imported before torch pulls anything else in
_host = sys.argv[sys.argv.index('--host') + 1] if '--host' in sys.argv else None
if _host == 'hrnet':
    from paper import host_hrnet as H
else:
    from paper import host_stdc as H

import torch                                   # noqa: E402

from paper import common                       # noqa: E402

CONFIGS = {
    # name, eval checkpoint key, and the training knobs of that arm
    'stdc': [
        dict(name='STDC-Seg',            key='I0',  use_brh=False, bnd_weight=1.0),
        dict(name='STDC-Seg + BPM only', key='I1',  use_brh=False, bnd_weight=3.0),
        dict(name='STDC-Seg + RSR only', key='H1',  use_brh=True,  bnd_weight=1.0),
        dict(name='STDC-Seg + Ours',     key='HI1', use_brh=True,  bnd_weight=3.0),
    ],
    'hrnet': [
        dict(name='HRNet-W48',        key='baseline'),
        dict(name='HRNet-W48 + Ours', key='HI1'),
    ],
}
GB = 1024.0 ** 3


# ---------------------------------------------------------------------------
# torch-version-agnostic memory counters (1.1 has *_cached, 2.x *_reserved)
# ---------------------------------------------------------------------------
def reset_peak():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()


def peak():
    torch.cuda.synchronize()
    res = getattr(torch.cuda, 'max_memory_reserved', None) or torch.cuda.max_memory_cached
    return torch.cuda.max_memory_allocated() / GB, res() / GB


def gpu_state():
    try:
        s = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,power.draw',
             '--format=csv,noheader,nounits']).decode().strip().split('\n')[0]
        t, clk, pw = [x.strip() for x in s.split(',')]
        return dict(temp_c=float(t), sm_mhz=float(clk), power_w=float(pw))
    except Exception:
        return {}


def gpu_busy():
    try:
        s = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                     '--format=csv,noheader']).decode().strip()
        return [l for l in s.split('\n') if l.strip()]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# params / MACs
# ---------------------------------------------------------------------------
def count_macs(net, x):
    """Conv2d + Linear multiply-accumulates for one forward (batch 1). This
    is the dominant cost and the quantity thop reports as 'flops'; BN/ReLU/
    interpolation are not counted."""
    total = [0]

    def conv_hook(m, inp, out):
        total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, inp, out):
        total[0] += out.numel() * m.in_features

    hooks = []
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        net(x)
    for h in hooks:
        h.remove()
    return total[0]


def thop_gflops(net, x):
    try:
        from thop import profile
        flops, _ = profile(net, inputs=(x,), verbose=False)
    except Exception:           # not installed in the hrnet env, or incompatible
        return None
    return flops / 1e9


# ---------------------------------------------------------------------------
# per-config measurements
# ---------------------------------------------------------------------------
def load_infer(host, cfg):
    if host == 'hrnet':
        net, hcfg = H.load_eval_net(cfg['key'])
        return net, H.infer_fn(net, hcfg)
    net = H.load_eval_net(cfg['key'])
    return net, H.infer_fn(net)


def measure_infer(host, cfg, warmup, iters):
    reset_peak()
    net, f = load_infer(host, cfg)
    x = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(warmup):
            f(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            f(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
    alloc, reserved = peak()
    del net, f, x
    return dict(img_s=iters / dt, ms=1000.0 * dt / iters, peak_alloc_gb=alloc, peak_reserved_gb=reserved)


def build_train(host, cfg):
    if host == 'hrnet':
        step, mods, _ = H.build_train_step(cfg['key'])
    else:
        step, mods = H.build_train_step(cfg['use_brh'], cfg['bnd_weight'])
    return step, mods


def measure_train(host, cfg, batches, warmup, iters):
    reset_peak()
    step, mods = build_train(host, cfg)
    n = len(batches)
    for i in range(warmup):
        im, lb = batches[i % n]
        step(im.cuda(), lb.cuda())
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        im, lb = batches[i % n]
        step(im.cuda(), lb.cuda())
    torch.cuda.synchronize()
    dt = time.time() - t0
    alloc, reserved = peak()
    del step, mods
    return dict(it_s=iters / dt, img_s=iters * batches[0][0].shape[0] / dt,
                peak_alloc_gb=alloc, peak_reserved_gb=reserved)


def static_cost(host, cfg):
    """params (M) and GMACs at the host's eval input size."""
    if host == 'hrnet':
        net, hcfg = H.build_net(cfg['key'])
        shape = H.infer_input_shape(hcfg)
    else:
        net = H.build_net(cfg['use_brh'])
        shape = H.infer_input_shape()
    net = net.cuda().eval()
    x = torch.randn(*shape).cuda()
    out = dict(params_m=sum(p.numel() for p in net.parameters()) / 1e6,
               gmacs=count_macs(net, x) / 1e9, input=list(shape))
    out['thop_gflops'] = thop_gflops(net, x)
    del net, x
    torch.cuda.empty_cache()
    return out


def summarize(xs):
    return dict(mean=statistics.mean(xs), std=statistics.stdev(xs) if len(xs) > 1 else 0.0,
                median=statistics.median(xs), min=min(xs), max=max(xs), n=len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True, choices=('stdc', 'hrnet'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--soak_s', type=int, default=90, help='heat-soak seconds before timing')
    ap.add_argument('--infer_warmup', type=int, default=10)
    ap.add_argument('--infer_iters', type=int, default=10, help='per config per round')
    ap.add_argument('--train_warmup', type=int, default=2, help='untimed steps after each rebuild')
    ap.add_argument('--train_iters', type=int, default=4, help='per config per round')
    ap.add_argument('--train_batches', type=int, default=4,
                    help='distinct real batches cycled through during timing')
    ap.add_argument('--force', action='store_true', help='run even if the GPU is busy')
    args = ap.parse_args()

    busy = gpu_busy()
    if busy and not args.force:
        raise SystemExit('GPU busy (%s); timing would be meaningless. Use --force to override.' % busy)
    torch.backends.cudnn.benchmark = True
    random.seed(0)
    host, cfgs = args.host, CONFIGS[args.host]
    cfgs = [c for c in cfgs if common.available(host, c['key'])]

    print('[%s] loading %d real training batches' % (host, args.train_batches))
    if host == 'hrnet':
        _, hcfg = H.build_net(cfgs[0]['key'])
        batches = H.train_batches(hcfg, args.train_batches)
    else:
        batches = H.train_batches(args.train_batches)
    batches = [(im.cpu().pin_memory(), lb.cpu().pin_memory()) for im, lb in batches]
    torch.cuda.empty_cache()
    train_shape = list(batches[0][0].shape)

    # memory: one deterministic pass per config. cudnn.benchmark autotuning
    # allocates large, run-dependent scratch workspaces that land in the
    # peak counters (HRNet HI1 infer read 11.6 GB with it vs ~6 GB baseline),
    # so memory is taken with it off and timing with it on.
    torch.backends.cudnn.benchmark = False
    mem = {}
    for c in cfgs:
        inf = measure_infer(host, c, 1, 2)
        tr = measure_train(host, c, batches, 1, 2)
        mem[c['name']] = dict(infer=inf, train=tr)
        print('  memory   %-22s infer %5.2f GB (reserved %5.2f) | train %5.2f GB (reserved %5.2f)'
              % (c['name'], inf['peak_alloc_gb'], inf['peak_reserved_gb'],
                 tr['peak_alloc_gb'], tr['peak_reserved_gb']))
    torch.backends.cudnn.benchmark = True

    # timing: every config resident at once, heat-soaked, then many short
    # blocks in a fresh random order each round. The T4 throttles to 825-1545
    # MHz at 83-84C and wanders within that range, so coarse per-config blocks
    # (the first version of this script) put each config at a different clock.
    # Fine interleaving makes every config sample the same clock distribution,
    # and the per-round ratio to the baseline (paired) cancels what remains.
    torch.backends.cudnn.benchmark = True
    x = torch.randn(1, 3, 1024, 2048).cuda()
    nb = len(batches)
    # Inference nets are small and stay resident. Training configs do NOT fit
    # together (4 x STDC at batch 16 OOMs the T4), so each train block builds
    # its config fresh, takes train_warmup untimed steps, times train_iters,
    # and frees it. cudnn's autotune cache is global, so a rebuild does not
    # re-autotune, and weights don't affect speed.
    run = OrderedDict()
    for c in cfgs:
        net, f = load_infer(host, c)
        with torch.no_grad():
            for _ in range(args.infer_warmup):
                f(x)
        run[c['name']] = dict(f=f, keep=net)
    torch.cuda.synchronize()

    def train_block(c, r):
        step, mods = build_train(host, c)
        for k in range(args.train_warmup):
            im, lb = batches[k % nb]
            step(im.cuda(), lb.cuda())
        torch.cuda.synchronize()
        t0 = time.time()
        for k in range(args.train_iters):
            im, lb = batches[(r + k) % nb]
            step(im.cuda(), lb.cuda())
        torch.cuda.synchronize()
        dt = time.time() - t0
        del step, mods
        torch.cuda.empty_cache()
        return dt

    print('[%s] heat soak %ds (reach steady-state clocks before timing)' % (host, args.soak_s))
    t_end, i = time.time() + args.soak_s, 0
    while time.time() < t_end:
        train_block(cfgs[i % len(cfgs)], i)
        i += 1

    raw = {c['name']: dict(infer=[], train=[]) for c in cfgs}
    gpu_log = []
    for r in range(args.rounds):
        order = list(cfgs)
        random.shuffle(order)
        gpu_log.append(gpu_state())
        for c in order:
            R = run[c['name']]
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(args.infer_iters):
                    R['f'](x)
                torch.cuda.synchronize()
                dt_i = time.time() - t0
            dt_t = train_block(c, r)
            raw[c['name']]['infer'].append(dict(round=r, iters=args.infer_iters, sec=dt_i,
                                                img_s=args.infer_iters / dt_i))
            raw[c['name']]['train'].append(dict(round=r, iters=args.train_iters, sec=dt_t,
                                                it_s=args.train_iters / dt_t))
        print('  round %2d/%d  %s  %s' % (r + 1, args.rounds, gpu_log[-1], '  '.join(
            '%s %.2f img/s %.3f it/s' % (c['name'], raw[c['name']]['infer'][-1]['img_s'],
                                         raw[c['name']]['train'][-1]['it_s']) for c in cfgs)))
    del run
    torch.cuda.empty_cache()

    def pooled(xs, key):
        return sum(v['iters'] for v in xs) / sum(v['sec'] for v in xs)

    def paired(name, phase, key):
        """median over rounds of (config rate / baseline rate) in the same round"""
        base = raw[cfgs[0]['name']][phase]
        return statistics.median(a[key] / b[key] for a, b in zip(raw[name][phase], base))

    results = []
    for c in cfgs:
        rr = raw[c['name']]
        inf_s = summarize([v['img_s'] for v in rr['infer']])
        inf_s['pooled'] = pooled(rr['infer'], 'img_s')
        tr_s = summarize([v['it_s'] for v in rr['train']])
        tr_s['pooled'] = pooled(rr['train'], 'it_s')
        results.append(dict(
            name=c['name'], key=c['key'], config={k: v for k, v in c.items() if k not in ('name',)},
            static=static_cost(host, c),
            infer=dict(img_s=inf_s, ms=1000.0 / inf_s['pooled'],
                       ratio_vs_baseline=paired(c['name'], 'infer', 'img_s'),
                       peak_alloc_gb=mem[c['name']]['infer']['peak_alloc_gb'],
                       peak_reserved_gb=mem[c['name']]['infer']['peak_reserved_gb']),
            train=dict(it_s=tr_s, img_s=tr_s['pooled'] * train_shape[0],
                       ratio_vs_baseline=paired(c['name'], 'train', 'it_s'),
                       peak_alloc_gb=mem[c['name']]['train']['peak_alloc_gb'],
                       peak_reserved_gb=mem[c['name']]['train']['peak_reserved_gb']),
            memory_pass=mem[c['name']], raw=rr))

    meta = dict(host=host, measured_at=common.now(), gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__, train_batch_shape=train_shape,
                cudnn_benchmark='timing: on; memory: off (deterministic pass)',
                infer_input='1x3x1024x2048 -> label map; ' + common.PROTOCOL[host]['input'],
                gpu_per_round=gpu_log, settings=vars(args))
    d = osp.dirname(args.out)
    if d and not osp.isdir(d):
        os.makedirs(d)
    with open(args.out, 'w') as f:
        json.dump(dict(meta=meta, results=results), f, indent=2)
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
