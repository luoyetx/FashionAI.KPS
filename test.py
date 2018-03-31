from __future__ import print_function, division

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import time
import shutil
import logging
import argparse
import cv2
import mxnet as mx
from mxnet import nd, autograd as ag, gluon as gl
from mxnet.gluon import nn
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from tensorboardX import SummaryWriter
from dataset import FashionAIKPSDataSet, process_cv_img
from model import PoseNet
from config import cfg


def draw(img, ht):
    h, w = img.shape[:2]
    ht = cv2.resize(ht, (w, h))
    ht[ht < 0] = 0
    ht[ht > 1] = 1
    ht = (ht * 255).astype(np.uint8)
    ht = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
    drawed = cv2.addWeighted(img, 0.5, ht, 0.5, 0)
    return drawed

def connect_score():
    pass


def detect_kps(img, heatmap, paf, category):
    h, w = img.shape[:2]
    heatmap = cv2.resize(heatmap.transpose((1, 2, 0)), (w, h))
    paf = cv2.resize(paf.transpose((1, 2, 0)), (w, h))
    num_ldm = len(cfg.LANDMARK_IDX[category])
    num_limb = len(cfg.PAF_LANDMARK_PAIR[category])
    sigma = 1
    # peaks
    peaks = []
    for i in range(num_ldm):
        ht_ori = heatmap[: , :, i]
        ht = gaussian_filter(ht_ori, sigma=sigma)
        ht_left = np.zeros(ht.shape)
        ht_left[1:,:] = ht[:-1,:]
        ht_right = np.zeros(ht.shape)
        ht_right[:-1,:] = ht[1:,:]
        ht_up = np.zeros(ht.shape)
        ht_up[:,1:] = ht[:,:-1]
        ht_down = np.zeros(ht.shape)
        ht_down[:,:-1] = ht[:,1:]
        peak_binary = np.logical_and.reduce((ht>ht_left, ht>ht_right, ht>ht_up, ht>ht_down, ht > 0.1))
        peak = zip(np.nonzero(peak_binary)[1], np.nonzero(peak_binary)[0]) # note reverse
        peak_with_score = [x + (ht_ori[x[1],x[0]],) for x in peak]
        peaks.append(peak_with_score)
    # # connection
    # connection = np.zeros((num_ldm, num_ldm))
    # pairs = cfg.PAF_LANDMARK_PAIR[category]
    # for i in range(num_ldm):
    #     for j in range(i + 1, num_ldm):
    #         paf_idx = -1
    #         if (i, j) in pairs:
    #             paf_idx = pairs.index((i, j))
    #         elif (j, i) in pairs:
    #             paf_idx = pairs.index((i, j))
    #         else:
    #             paf_idx = -1
    #         if paf_idx == -1:
    #             continue
    #         score = paf[:, : 2*paf_idx:2*paf_idx+2]

    #         connection[j, i] = connection[i, j]
    # detect kps
    kps = np.zeros((num_ldm, 3), dtype='int32')
    kps[:, :] = -1
    for i in range(num_ldm):
        if len(peaks[i]) >= 1:
            idx = 0
            max_score = peaks[i][0][2]
            for j in range(1, len(peaks[i])):
                if peaks[i][j][2] > max_score:
                    max_score = peaks[i][j][2]
                    idx = j
            kps[i, 0] = peaks[i][idx][0]
            kps[i, 1] = peaks[i][idx][1]
            kps[i, 2] = 1
    return kps


def draw_kps(im, kps):
    im = im.copy()
    for i, (x, y) in enumerate(kps):
        if x != -1 and y != -1:
            x, y = int(x), int(y)
            cv2.circle(im, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(im, '%d'%i, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='-1')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--category', type=str, default='skirt', choices=['blouse', 'skirt', 'outwear', 'dress', 'trousers'])
    parser.add_argument('--cpm-stages', type=int, default=5)
    parser.add_argument('--cpm-channels', type=int, default=64)
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--backbone', type=str, default='vgg19', choices=['vgg19'])
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    print(args)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    cpm_stages = args.cpm_stages
    cpm_channels = args.cpm_channels
    epoch = args.epoch
    optim = args.optim
    category = args.category
    data_dir = args.data_dir
    backbone = args.backbone
    base_name = '%s-%s-S%d-C%d-%s' % (category, backbone, cpm_stages, cpm_channels, optim)
    show = args.show
    # model
    num_kps = len(cfg.LANDMARK_IDX[category])
    num_limb = len(cfg.PAF_LANDMARK_PAIR[category])
    net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=cpm_stages, channels=cpm_channels)
    creator, featname, fixed = cfg.BACKBONE[backbone]
    net.init_backbone(creator, featname, fixed)
    net.load_params('./output/%s-%04d.params' % (base_name, epoch), mx.cpu(0))
    net.collect_params().reset_ctx(ctx)
    # data
    df = pd.read_csv(os.path.join(data_dir, 'test/test.csv'))
    num = len(df)
    result = []
    for idx, row in df.iterrows():
        if row['image_category'] == category:
            print('process', idx)
            path = os.path.join(data_dir, 'test', row['image_id'])
            img = cv2.imread(path)
            data = process_cv_img(img)
            batch = mx.nd.array(data[np.newaxis], ctx)
            out = net(batch)
            heatmap = out[-1][0][0].asnumpy()
            paf = out[-1][1][0].asnumpy()
            kps = detect_kps(img, heatmap, paf, category)
            all_kps = np.zeros((24,3), dtype='int32')
            all_kps[:, :] = -1
            assert len(kps) == len(cfg.LANDMARK_IDX[category])
            for i, ldm_idx in enumerate(cfg.LANDMARK_IDX[category]):
                all_kps[ldm_idx][0] = kps[i][0]
                all_kps[ldm_idx][1] = kps[i][1]
                all_kps[ldm_idx][2] = kps[i][2]
            result.append((row['image_id'], all_kps))

            if show:
                heatmap = heatmap[::-1].max(axis=0)
                n, h, w = paf.shape
                paf = paf.reshape((n // 2, 2, h, w))
                paf = np.sqrt(np.square(paf[:, 0]) + np.square(paf[:, 1]))
                paf = paf.max(axis=0)

                dr1 = draw(img, heatmap)
                dr2 = draw(img, paf)
                dr3 = draw_kps(img, all_kps[:, :2])

                cv2.imshow('heatmap', dr1)
                cv2.imshow('paf', dr2)
                cv2.imshow('detect', dr3)
                key = cv2.waitKey(0)
                if key == 27:
                    break

    with open('./%s.csv'%category, 'w') as fout:
        for img_id, kps in result:
            fout.write(img_id)
            fout.write(',%s'%category)
            for p in kps:
                s=',%d_%d_%d'%(p[0], p[1], p[2])
                fout.write(s)
            fout.write('\n')
