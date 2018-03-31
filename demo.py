from __future__ import print_function, division

import os
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
from tensorboardX import SummaryWriter
from dataset import FashionAIKPSDataSet
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='-1')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--category', type=str, default='skirt', choices=['blouse', 'skirt', 'outwear', 'dress', 'trousers'])
    parser.add_argument('--cpm-stages', type=int, default=5)
    parser.add_argument('--cpm-channels', type=int, default=64)
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--backbone', type=str, default='vgg19', choices=['vgg19'])
    parser.add_argument('--test-idx', type=int, default=0)
    args = parser.parse_args()
    print(args)
    # seed
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    # hyper parameters
    ctx = mx.cpu(0) if args.gpu == -1 else mx.gpu(args.gpu)
    cpm_stages = args.cpm_stages
    cpm_channels = args.cpm_channels
    epoch = args.epoch
    optim = args.optim
    category = args.category
    data_dir = args.data_dir
    backbone = args.backbone
    test_idx = args.test_idx
    base_name = '%s-%s-S%d-C%d-%s' % (category, backbone, cpm_stages, cpm_channels, optim)
    # model
    num_kps = len(cfg.LANDMARK_IDX[category])
    num_limb = len(cfg.PAF_LANDMARK_PAIR[category])
    net = PoseNet(num_kps=num_kps, num_limb=num_limb, stages=cpm_stages, channels=cpm_channels)
    creator, featname, fixed = cfg.BACKBONE[backbone]
    net.init_backbone(creator, featname, fixed)
    net.load_params('./output/%s-%04d.params' % (base_name, epoch), mx.cpu(0))
    net.collect_params().reset_ctx(ctx)
    # data
    df = pd.read_csv(os.path.join(data_dir, 'train/Annotations/train.csv'))
    df = df.sample(frac=1)
    train_num = int(len(df) * 0.9)
    df_train = df[:train_num]
    df_test = df[train_num:]
    traindata = FashionAIKPSDataSet(df_train, category, True)
    testdata = FashionAIKPSDataSet(df_test, category, False)
    # render
    mean = np.array(cfg.PIXEL_MEAN, dtype='float32').reshape((3, 1, 1))
    std = np.array(cfg.PIXEL_STD, dtype='float32').reshape((3, 1, 1))
    data, heatmap, paf = testdata[test_idx]
    heatmap = heatmap.max(axis=0)
    n, h, w = paf.shape
    paf = paf.reshape((n // 2, 2, h, w))
    paf = np.sqrt(np.square(paf[:, 0]) + np.square(paf[:, 1]))
    paf = paf.max(axis=0)
    im = (data * std + mean).astype('uint8').transpose((1, 2, 0))[:, :, ::-1]

    out = net(nd.array(data[np.newaxis], ctx))
    out_heatmap = out[-1][0][0].asnumpy()
    out_paf = out[-1][1][0].asnumpy()
    out_heatmap = out_heatmap[::-1].max(axis=0)
    n, h, w = out_paf.shape
    out_paf = out_paf.reshape((n // 2, 2, h, w))
    out_paf = np.sqrt(np.square(out_paf[:, 0]) + np.square(out_paf[:, 1]))
    out_paf = out_paf.max(axis=0)

    dr1 = draw(im, heatmap)
    dr2 = draw(im, paf)
    dr3 = draw(im, out_heatmap)
    dr4 = draw(im, out_paf)

    cv2.imwrite('./tmp/%s_%d_ori_heatmap.jpg' % (category, test_idx), dr1)
    cv2.imwrite('./tmp/%s_%d_ori_paf.jpg' % (category, test_idx), dr2)
    cv2.imwrite('./tmp/%s_%d_pred_heatmap.jpg' % (category, test_idx), dr3)
    cv2.imwrite('./tmp/%s_%d_pred_paf.jpg' % (category, test_idx), dr4)
    cv2.imshow('ori_heatmap', dr1)
    cv2.imshow('ori_paf', dr2)
    cv2.imshow('pred_heatmap', dr3)
    cv2.imshow('pred_paf', dr4)
    cv2.waitKey(0)
