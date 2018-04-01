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
    thres1 = 0.1
    num_mid = 10
    thres2 = 0.05
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
        peak_binary = np.logical_and.reduce((ht>ht_left, ht>ht_right, ht>ht_up, ht>ht_down, ht > thres1))
        peak = zip(np.nonzero(peak_binary)[1], np.nonzero(peak_binary)[0]) # note reverse
        peak_with_score_links = [[x[0], x[1], ht_ori[x[1], x[0]], 0, 0] for x in peak]
        peaks.append(peak_with_score_links)
    # connection
    for idx, (ldm1, ldm2) in enumerate(cfg.PAF_LANDMARK_PAIR[category]):
        candA = peaks[ldm1]
        candB = peaks[ldm2]
        nA = len(candA)
        nB = len(candB)
        if nA != 0 and nB != 0:
            connection_candidate = []
            score = paf[:, :, 2*idx: 2*idx+2]
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    norm = max(norm, 1e-5)
                    vec = np.divide(vec, norm)
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=num_mid), np.linspace(candA[i][1], candB[j][1], num=num_mid))
                    vec_x = np.array([score[int(round(startend[k][1])), int(round(startend[k][0])), 0] for k in range(len(startend))])
                    vec_y = np.array([score[int(round(startend[k][1])), int(round(startend[k][0])), 1] for k in range(len(startend))])
                    score_mid = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_mid) / len(score_mid) + min(0.5*h/norm - 1, 0)
                    c1 = (score_mid > thres2).sum() > 0.8 * len(score_mid)
                    c2 = score_with_dist_prior > 0
                    if c1 and c2:
                        connection_candidate.append([i, j, score_with_dist_prior])
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 3))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c]
                if i not in connection[:, 0] and j not in connection[:, 1]:
                    connection = np.vstack([connection, [i, j, s]])
                    if len(connection) >= min(nA, nB):
                        break
            for i, j, s in connection:
                i, j = int(i), int(j)
                candA[i][3] += 1
                candB[j][3] += 1
                candA[i][4] += s
                candB[j][4] += s
    # detect kps
    kps = np.zeros((num_ldm, 3), dtype='int32')
    kps[:, :] = -1
    for i in range(num_ldm):
        cand = peaks[i]
        if len(cand) >= 1:
            idx = 0
            max_links = cand[0][3]
            max_score = cand[0][2]
            for j in range(1, len(cand)):
                if cand[j][3] > max_links or (cand[j][3] == max_links and cand[j][2] > max_score):
                    max_links = cand[j][3]
                    max_score = cand[j][2]
                    idx = j
            # if len(cand) > 1:
            #     print(i, 'select with', cand[idx][3], 'links and', cand[idx][2], 'score')
            kps[i, 0] = cand[idx][0]
            kps[i, 1] = cand[idx][1]
            kps[i, 2] = 1
    # cheat
    keep = np.logical_and(kps[:, 0] != -1, kps[:, 1] != -1)
    if keep.sum() != 0:
        xmin = kps[keep, 0].min()
        xmax = kps[keep, 0].max()
        ymin = kps[keep, 1].min()
        ymax = kps[keep, 1].max()
        xc = (xmin + xmax) // 2
        yc = (ymin + ymax) // 2
    else:
        xc = w // 2
        yc = h // 2
    miss = np.logical_not(keep)
    kps[miss, 0] = xc
    kps[miss, 1] = yc
    kps[miss, 2] = 0
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
    cc = 0
    for idx, row in df.iterrows():
        if row['image_category'] == category:
            cc += 1
            if cc % 50 == 0:
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
                ht = heatmap[-1]
                heatmap = heatmap[::-1].max(axis=0)
                n, h, w = paf.shape
                paf = paf.reshape((n // 2, 2, h, w))
                paf = np.sqrt(np.square(paf[:, 0]) + np.square(paf[:, 1]))
                paf = paf.max(axis=0)

                dr1 = draw(img, heatmap)
                dr2 = draw(img, paf)
                dr3 = draw_kps(img, all_kps[:, :2])
                dr4 = draw(img, ht)

                cv2.imshow('heatmap', dr1)
                cv2.imshow('paf', dr2)
                cv2.imshow('detect', dr3)
                #cv2.imshow('front', dr4)
                key = cv2.waitKey(0)
                if key == 27:
                    break

    with open('./result/%s.csv'%category, 'w') as fout:
        for img_id, kps in result:
            fout.write(img_id)
            fout.write(',%s'%category)
            for p in kps:
                s=',%d_%d_%d'%(p[0], p[1], p[2])
                fout.write(s)
            fout.write('\n')
