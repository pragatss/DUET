#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
ROS node: camera image -> STDC-Seg (or STDC-Seg + Ours) seg map -> TADA
traversability map. It publishes the same topics as TADA's tools/in_ros.py
(https://github.com/Masrur02/TADA), so TADA's projection.py and the planner
run unchanged on top of it.

  subscribes  /D435i/color/image_raw   (--image_topic)
  publishes   /trav_map      32FC1  traversability in [0, 1] (read by projection.py)
              /trav_map_pov  8UC1   traversability x 255
              /trav_vis      bgr8   RGB blended with the jet trav colours (red = traversable)
              /seg_raw       bgr8   colour seg map
              /rgb_proc      bgr8   the resized input

All outputs are --width x --height (default 640x480, as in_ros.py), which must
match the point cloud projection.py pairs /trav_map with.

    python trav_ros.py --dataset rugd --model stdc         # baseline
    python trav_ros.py --dataset rugd --model stdc_ours    # + RSR + BPM
    ... --ckpt /path/to/model.pth                          # checkpoint path on this machine

Needs rospy, torch, cv2 and matplotlib in one env.
"""
import argparse
import time

import cv2
import numpy as np
import rospy
import torch
from matplotlib import cm
from sensor_msgs.msg import Image

from evaluation import load_net
from gen_qualitative import PALETTE
from gen_trav import MODELS, trav_lut
from paper import common
from paper.host_stdc import BACKBONE, predict

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def msg_to_rgb(msg):
    ch = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)[:, :, :3]
    return img[:, :, ::-1] if msg.encoding.startswith('bgr') else img


def to_msg(arr, encoding, header):
    arr = np.ascontiguousarray(arr)
    m = Image()
    m.header = header
    m.height, m.width = arr.shape[:2]
    m.encoding = encoding
    m.is_bigendian = 0
    m.step = arr.strides[0]
    m.data = arr.tobytes()
    return m


class TravNode(object):
    def __init__(self, args):
        key = MODELS[args.dataset][1 if args.model == 'stdc_ours' else 0]
        run = common.STDC_RUNS[key]
        ckpt = args.ckpt or common.ckpt_path('stdc', key)
        self.net = load_net(ckpt, BACKBONE, True, run['use_brh'], 64,
                            len(common.DATASETS[args.dataset]['classes']), run['brh_variant']).cuda().eval()
        self.lut = trav_lut(args.dataset)
        pal = np.zeros((256, 3), np.uint8)
        pal[:len(PALETTE[args.dataset])] = PALETTE[args.dataset]
        self.pal = pal
        self.args = args
        self.count = 0
        self.pubs = {t: rospy.Publisher(t, Image, queue_size=10)
                     for t in ('trav_map', 'trav_map_pov', 'trav_vis', 'seg_raw', 'rgb_proc')}
        rospy.Subscriber(args.image_topic, Image, self.callback, queue_size=1, buff_size=2 ** 24)
        rospy.loginfo('trav_ros: %s on %s, ckpt %s' % (args.model, args.dataset, ckpt))

    def callback(self, msg):
        self.count += 1
        if self.count % self.args.every:
            return
        t0 = time.time()
        a = self.args
        rgb = cv2.resize(msg_to_rgb(msg), (a.width, a.height), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(((rgb / 255.0 - MEAN) / STD).astype(np.float32).transpose(2, 0, 1))
        with torch.no_grad():
            seg = predict(self.net, x.unsqueeze(0).cuda(), (a.height, a.width), scale=a.scale)[0]
        seg = seg.cpu().numpy().astype(np.uint8)
        trav = self.lut[seg]
        color = (cm.jet(trav)[:, :, :3] * 255).astype(np.uint8)
        vis = cv2.addWeighted(rgb, 0.6, color, 0.4, 0)

        h = msg.header
        self.pubs['trav_map'].publish(to_msg(trav.astype(np.float32), '32FC1', h))
        self.pubs['trav_map_pov'].publish(to_msg(np.round(trav * 255).astype(np.uint8), '8UC1', h))
        self.pubs['trav_vis'].publish(to_msg(vis[:, :, ::-1], 'bgr8', h))
        self.pubs['seg_raw'].publish(to_msg(self.pal[seg][:, :, ::-1], 'bgr8', h))
        self.pubs['rgb_proc'].publish(to_msg(rgb[:, :, ::-1], 'bgr8', h))
        rospy.logdebug('trav_ros: %.1f FPS' % (1.0 / (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='rugd', choices=list(MODELS), help='whose checkpoints and trav priors')
    ap.add_argument('--model', default='stdc_ours', choices=('stdc', 'stdc_ours'))
    ap.add_argument('--ckpt', default=None, help='checkpoint path (default: the registered one)')
    ap.add_argument('--image_topic', default='/D435i/color/image_raw')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--scale', type=float, default=common.PROTOCOL['stdc']['scale'],
                    help='network input scale (evaluation uses 0.75)')
    ap.add_argument('--every', type=int, default=1, help='process every Nth frame (in_ros.py uses 5)')
    args = ap.parse_args(rospy.myargv()[1:])
    rospy.init_node('stdc_trav', anonymous=True)
    TravNode(args)
    rospy.spin()


if __name__ == '__main__':
    main()
